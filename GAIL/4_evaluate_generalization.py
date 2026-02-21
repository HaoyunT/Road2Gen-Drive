import argparse
import os
from typing import Any, Dict, List, Tuple

import cv2
import gymnasium as gym
import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv
from panda3d.core import loadPrcFileData
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GAIL policy generalization on unseen MetaDrive scenarios.")
    parser.add_argument("--model-path", type=str, default="/root/autodl-tmp/GAIL/checkpoints/gail_policy.zip", help="Path to saved PPO model.")
    parser.add_argument("--start-seed", type=int, default=5000, help="Start seed for unseen scenarios.")
    parser.add_argument("--num-scenarios", type=int, default=100, help="Number of unseen scenarios.")
    parser.add_argument("--horizon", type=int, default=1200, help="Episode horizon.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--deterministic", dest="deterministic", action="store_true", help="Use deterministic policy action for evaluation (default).")
    mode_group.add_argument("--stochastic", dest="deterministic", action="store_false", help="Use stochastic policy action for evaluation.")
    parser.set_defaults(deterministic=True)
    parser.add_argument("--print-every", type=int, default=20, help="Print one episode summary every N episodes.")
    parser.add_argument("--render", action="store_true", help="Enable MetaDrive visualization during evaluation.")
    parser.add_argument("--num-runs", type=int, default=5, help="Number of evaluation runs for confidence intervals.")
    parser.add_argument("--run-seed-stride", type=int, default=100, help="Seed offset per run; run_k starts at start-seed + k*stride.")
    parser.add_argument("--vecnorm-path", type=str, default="", help="Path to VecNormalize stats pickle for observation normalization.")
    parser.add_argument("--action-smoothing-alpha", type=float, default=0.6, help="Low-pass smoothing factor for action in [0,1).")
    record_group = parser.add_mutually_exclusive_group()
    record_group.add_argument("--record-success-mp4", dest="record_success_mp4", action="store_true", help="Save successful episodes as MP4 videos (default: enabled).")
    record_group.add_argument("--no-record-success-mp4", dest="record_success_mp4", action="store_false", help="Disable saving successful episodes as MP4 videos.")
    parser.set_defaults(record_success_mp4=True)
    parser.add_argument("--video-dir", type=str, default="/root/autodl-tmp/gail_videos", help="Directory to save recorded MP4 files.")
    offscreen_group = parser.add_mutually_exclusive_group()
    offscreen_group.add_argument("--offscreen-render", dest="offscreen_render", action="store_true", help="Force Panda3D offscreen rendering (default: enabled).")
    offscreen_group.add_argument("--no-offscreen-render", dest="offscreen_render", action="store_false", help="Disable Panda3D offscreen rendering.")
    parser.set_defaults(offscreen_render=True)
    parser.add_argument("--video-fps", type=int, default=20, help="FPS of saved MP4 videos.")
    parser.add_argument("--max-recorded-successes", type=int, default=-1, help="Maximum number of successful episodes to save per run (<=0 means save all).")
    return parser.parse_args()


def resolve_model_path(path: str) -> str:
    if os.path.exists(path):
        return path

    candidate = os.path.join("checkpoints", os.path.basename(path))
    if os.path.exists(candidate):
        print(f"[Info] model-path not found at '{path}', using '{candidate}'")
        return candidate

    return path


def resolve_vecnorm_path(model_path: str, vecnorm_path: str) -> str:
    if vecnorm_path and os.path.exists(vecnorm_path):
        return vecnorm_path

    candidates = [
        model_path.replace(".zip", "_vecnormalize.pkl"),
        os.path.splitext(model_path)[0] + "_vecnormalize.pkl",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            print(f"[Info] Auto-detected VecNormalize stats: {candidate}")
            return candidate
    return vecnorm_path


class ActionSmoothingWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, alpha: float):
        super().__init__(env)
        self.alpha = float(alpha)
        self._prev_action = None

    def reset(self, **kwargs):
        self._prev_action = None
        return self.env.reset(**kwargs)

    def action(self, action):
        action_arr = np.asarray(action, dtype=np.float32)
        if self._prev_action is None:
            smoothed = action_arr
        else:
            smoothed = self.alpha * self._prev_action + (1.0 - self.alpha) * action_arr
        self._prev_action = smoothed
        return smoothed


def unpack_reset(reset_ret: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    if isinstance(reset_ret, tuple) and len(reset_ret) == 2:
        return reset_ret
    return reset_ret, {}


def unpack_step(step_ret: Any) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
    if len(step_ret) == 5:
        obs, reward, terminated, truncated, info = step_ret
        return obs, float(reward), bool(terminated or truncated), info
    obs, reward, done, info = step_ret
    return obs, float(reward), bool(done), info


def extract_speed(info: Dict[str, Any]) -> float:
    for key in ["velocity", "speed", "ego_speed", "vehicle_speed"]:
        if key in info:
            try:
                return float(info[key])
            except (TypeError, ValueError):
                pass
    return 0.0


def has_collision(info: Dict[str, Any]) -> bool:
    collision_keys = ["crash", "crash_vehicle", "crash_object", "crash_sidewalk", "out_of_road", "crash_human"]
    return any(bool(info.get(k, False)) for k in collision_keys)


def is_success(info: Dict[str, Any]) -> bool:
    return bool(info.get("arrive_dest", False) or info.get("success", False))


def compute_steering_jerk(actions: List[np.ndarray]) -> float:
    """Compute mean absolute jerk of steering signal (smoothness metric)."""
    if len(actions) < 3:
        return 0.0
    steers = [float(a[0]) if a.size > 0 else 0.0 for a in actions]
    jerk = [abs(steers[i + 2] - 2 * steers[i + 1] + steers[i]) for i in range(len(steers) - 2)]
    return float(np.mean(jerk))


def capture_rgb_frame(env: gym.Env) -> np.ndarray | None:
    render_env = env.unwrapped if hasattr(env, "unwrapped") else env

    render_candidates = [
        {"mode": "topdown", "film_size": (1280, 720)},
        {"mode": "rgb_array"},
        {},
    ]
    for kwargs in render_candidates:
        try:
            frame = render_env.render(**kwargs)
        except Exception:
            continue

        if isinstance(frame, tuple) and len(frame) > 0:
            frame = frame[0]
        if isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.shape[-1] in (3, 4):
            if frame.shape[-1] == 4:
                frame = frame[..., :3]
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            return frame
    return None


def save_episode_mp4(frames: List[np.ndarray], output_path: str, fps: int) -> bool:
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(max(1, fps)),
        (w, h),
    )
    if not writer.isOpened():
        return False
    try:
        for frame in frames:
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return True


def run_single_evaluation(
    model: PPO,
    env: gym.Env,
    args: argparse.Namespace,
    obs_rms=None,
    run_idx: int = 0,
    run_start_seed: int | None = None,
) -> Dict[str, Any]:
    """Run one full evaluation pass over all scenarios."""
    success_count = 0
    collision_count = 0
    out_of_road_count = 0
    speed_values: List[float] = []
    episode_lengths: List[int] = []
    episode_steering_jerks: List[float] = []
    route_completions: List[float] = []
    saved_success_videos = 0
    if args.record_success_mp4:
        os.makedirs(args.video_dir, exist_ok=True)

    seed_base = args.start_seed if run_start_seed is None else int(run_start_seed)

    for ep in range(args.num_scenarios):
        eval_seed = seed_base + ep
        try:
            reset_ret = env.reset(seed=eval_seed)
        except TypeError:
            reset_ret = env.reset()
        obs, _ = unpack_reset(reset_ret)
        done = False
        ep_collision = False
        ep_success = False
        ep_out_of_road = False
        ep_steps = 0
        ep_actions: List[np.ndarray] = []
        ep_route_completion = 0.0
        ep_frames: List[np.ndarray] = []

        if args.record_success_mp4:
            first_frame = capture_rgb_frame(env)
            if first_frame is not None:
                ep_frames.append(first_frame)

        while not done:
            obs_for_pred = obs
            if obs_rms is not None:
                obs_for_pred = np.clip(
                    (obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8),
                    -10.0, 10.0,
                ).astype(np.float32)

            action, _ = model.predict(obs_for_pred, deterministic=args.deterministic)
            step_ret = env.step(action)
            obs, _, done, info = unpack_step(step_ret)

            speed_values.append(extract_speed(info))
            ep_collision = ep_collision or has_collision(info)
            ep_success = ep_success or is_success(info)
            ep_out_of_road = ep_out_of_road or bool(info.get("out_of_road", False))
            ep_actions.append(np.asarray(action, dtype=np.float32))
            ep_steps += 1

            rc = info.get("route_completion", info.get("progress", -1.0))
            if isinstance(rc, (int, float)) and rc >= 0:
                ep_route_completion = max(ep_route_completion, float(rc))

            if args.record_success_mp4:
                frame = capture_rgb_frame(env)
                if frame is not None:
                    ep_frames.append(frame)

        if ep_success:
            success_count += 1
            ep_route_completion = 1.0
        if ep_collision:
            collision_count += 1
        if ep_out_of_road:
            out_of_road_count += 1

        episode_lengths.append(ep_steps)
        episode_steering_jerks.append(compute_steering_jerk(ep_actions))
        route_completions.append(ep_route_completion)

        if (
            args.record_success_mp4
            and ep_success
            and (args.max_recorded_successes <= 0 or saved_success_videos < args.max_recorded_successes)
            and len(ep_frames) > 0
        ):
            video_path = os.path.join(
                args.video_dir,
                f"eval_run{run_idx + 1:02d}_ep{ep + 1:03d}_seed{eval_seed}_success.mp4",
            )
            ok = save_episode_mp4(ep_frames, video_path, args.video_fps)
            if ok:
                saved_success_videos += 1
                print(f"[Eval] Saved success video: {video_path}")
            else:
                print(f"[Warn] Failed to save video: {video_path}")

        if (ep + 1) % args.print_every == 0 or (ep + 1) == args.num_scenarios:
            print(
                f"[Eval] Episode {ep + 1}/{args.num_scenarios}: "
                f"seed={eval_seed}, "
                f"success={ep_success}, collision={ep_collision}, out_of_road={ep_out_of_road}, "
                f"steps={ep_steps}, route={ep_route_completion:.2%}"
            )

    n = args.num_scenarios
    return {
        "success_rate": success_count / n,
        "collision_rate": collision_count / n,
        "out_of_road_rate": out_of_road_count / n,
        "avg_speed": float(np.mean(speed_values)) if speed_values else 0.0,
        "avg_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "std_episode_length": float(np.std(episode_lengths)) if episode_lengths else 0.0,
        "avg_route_completion": float(np.mean(route_completions)) if route_completions else 0.0,
        "avg_steering_jerk": float(np.mean(episode_steering_jerks)) if episode_steering_jerks else 0.0,
        "saved_success_videos": saved_success_videos,
    }


def main() -> None:
    args = parse_args()
    if args.offscreen_render:
        loadPrcFileData("", "window-type offscreen")

    if not os.environ.get("DISPLAY"):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    if args.print_every <= 0:
        raise ValueError("print-every must be positive")
    if args.video_fps <= 0:
        raise ValueError("video-fps must be positive")
    if args.num_runs <= 0:
        raise ValueError("num-runs must be positive")
    if args.run_seed_stride < 0:
        raise ValueError("run-seed-stride must be non-negative")
    if not (0.0 <= args.action_smoothing_alpha < 1.0):
        raise ValueError("action-smoothing-alpha must be in [0,1)")

    args.video_dir = os.path.abspath(os.path.expanduser(args.video_dir or "/root/autodl-tmp/gail_videos"))

    model_path = resolve_model_path(args.model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    args.vecnorm_path = resolve_vecnorm_path(model_path, args.vecnorm_path)

    obs_rms = None
    if args.vecnorm_path and os.path.exists(args.vecnorm_path):
        dummy_env_config = {
            "use_render": False, "image_observation": False, "manual_control": False,
            "start_seed": 0, "num_scenarios": 1, "horizon": 100,
        }
        from stable_baselines3.common.env_util import make_vec_env
        dummy_venv = make_vec_env(lambda: MetaDriveEnv(dummy_env_config), n_envs=1)
        vn = VecNormalize.load(args.vecnorm_path, dummy_venv)
        obs_rms = vn.obs_rms
        dummy_venv.close()
        print(f"[Info] Loaded VecNormalize obs stats from: {args.vecnorm_path}")

    if args.offscreen_render and args.render:
        print("[Info] offscreen-render is enabled; ignore --render to avoid opening a physical window.")

    env_config = {
        "use_render": (args.render or args.record_success_mp4) and (not args.offscreen_render),
        "image_observation": False,
        "manual_control": False,
        "start_seed": args.start_seed,
        "num_scenarios": args.num_scenarios,
        "horizon": args.horizon,
    }
    if args.offscreen_render:
        env_config["_render_mode"] = "offscreen"

    try:
        model = PPO.load(model_path)

        all_run_results: List[Dict[str, Any]] = []
        for run_idx in range(args.num_runs):
            run_start_seed = args.start_seed + run_idx * args.run_seed_stride
            run_env_config = dict(env_config)
            run_env_config["start_seed"] = run_start_seed
            run_env_config["num_scenarios"] = args.num_scenarios

            if args.num_runs > 1:
                print(f"\n{'='*80}\n[Eval] Run {run_idx + 1}/{args.num_runs} (seed range: {run_start_seed}-{run_start_seed + args.num_scenarios - 1})\n{'='*80}")

            env = None
            try:
                base_env = MetaDriveEnv(run_env_config)
                if args.action_smoothing_alpha > 0.0:
                    env = ActionSmoothingWrapper(base_env, args.action_smoothing_alpha)
                else:
                    env = base_env

                result = run_single_evaluation(
                    model,
                    env,
                    args,
                    obs_rms=obs_rms,
                    run_idx=run_idx,
                    run_start_seed=run_start_seed,
                )
                all_run_results.append(result)
            finally:
                if env is not None:
                    env.close()

        print("\n" + "=" * 80)
        print(f"Generalization evaluation on unseen seeds [{args.start_seed}, {args.start_seed + args.num_scenarios - 1}]")

        if args.num_runs == 1:
            r = all_run_results[0]
            print(f"  Success Rate:          {r['success_rate']:.2%}")
            print(f"  Collision Rate:        {r['collision_rate']:.2%}")
            print(f"  Out-of-road Rate:      {r['out_of_road_rate']:.2%}")
            print(f"  Average Speed:         {r['avg_speed']:.3f}")
            print(f"  Avg Episode Length:    {r['avg_episode_length']:.1f} +/- {r['std_episode_length']:.1f}")
            print(f"  Avg Route Completion:  {r['avg_route_completion']:.2%}")
            print(f"  Avg Steering Jerk:     {r['avg_steering_jerk']:.6f}")
            if args.record_success_mp4:
                print(f"  Saved Success Videos:  {int(r['saved_success_videos'])}")
        else:
            metric_keys = ["success_rate", "collision_rate", "out_of_road_rate", "avg_speed",
                           "avg_episode_length", "avg_route_completion", "avg_steering_jerk", "saved_success_videos"]
            for key in metric_keys:
                vals = [r[key] for r in all_run_results]
                mean_v = float(np.mean(vals))
                std_v = float(np.std(vals))
                if "rate" in key or "completion" in key:
                    print(f"  {key:28s}: {mean_v:.2%} +/- {std_v:.2%}")
                elif key == "saved_success_videos":
                    print(f"  {key:28s}: {mean_v:.2f} +/- {std_v:.2f}")
                else:
                    print(f"  {key:28s}: {mean_v:.4f} +/- {std_v:.4f}")

        print("=" * 80)

    finally:
        pass


if __name__ == "__main__":
    main()
