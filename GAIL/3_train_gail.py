import argparse
import os
import pickle
from typing import Any, Callable, Dict, List

import gymnasium as gym
from imitation.algorithms.adversarial.gail import GAIL
from imitation.algorithms.bc import BC
from imitation.data import rollout
from imitation.data.wrappers import BufferingWrapper
from imitation.data.types import Trajectory
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from metadrive.envs.metadrive_env import MetaDriveEnv
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear learning rate decay: lr = initial_value * progress_remaining."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


class MetaDriveResetSeedAdapter(gym.Wrapper):
    def __init__(self, env: MetaDriveEnv, start_seed: int, num_scenarios: int):
        super().__init__(env)
        self._start_seed = int(start_seed)
        self._num_scenarios = max(1, int(num_scenarios))

    def reset(self, *, seed=None, options=None):
        mapped_seed = None
        if seed is not None:
            mapped_seed = self._start_seed + (int(seed) % self._num_scenarios)

        try:
            if mapped_seed is None:
                return self.env.reset()
            return self.env.reset(seed=mapped_seed)
        except TypeError:
            return self.env.reset()


class SmoothActionWrapper(gym.ActionWrapper):
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


class LaneKeepingRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        lane_center_penalty_coef: float,
        out_of_road_penalty: float,
        action_jerk_penalty_coef: float,
        steer_jerk_penalty_coef: float,
    ):
        super().__init__(env)
        self.lane_center_penalty_coef = float(lane_center_penalty_coef)
        self.out_of_road_penalty = float(out_of_road_penalty)
        self.action_jerk_penalty_coef = float(action_jerk_penalty_coef)
        self.steer_jerk_penalty_coef = float(steer_jerk_penalty_coef)
        self._prev_action: np.ndarray | None = None

    def reset(self, **kwargs):
        self._prev_action = None
        return self.env.reset(**kwargs)

    def _extract_center_offset(self, info: Dict[str, Any]) -> float:
        for key in ["distance_to_center", "dist_to_center", "lateral_dist", "lateral_distance"]:
            if key in info:
                try:
                    return abs(float(info[key]))
                except (TypeError, ValueError):
                    continue
        return 0.0

    def step(self, action):
        step_ret = self.env.step(action)
        if len(step_ret) == 5:
            obs, reward, terminated, truncated, info = step_ret
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = step_ret
            terminated, truncated = bool(done), False

        info_dict = info if isinstance(info, dict) else {}
        shaped_reward = float(reward)

        center_offset = self._extract_center_offset(info_dict)
        lane_penalty = self.lane_center_penalty_coef * center_offset

        out_of_road = bool(info_dict.get("out_of_road", False) or info_dict.get("out_of_lane", False))
        offroad_penalty = self.out_of_road_penalty if out_of_road else 0.0

        action_arr = np.asarray(action, dtype=np.float32)
        jerk_penalty = 0.0
        steer_jerk_penalty = 0.0
        if self._prev_action is not None and action_arr.shape == self._prev_action.shape:
            delta = action_arr - self._prev_action
            jerk_penalty = self.action_jerk_penalty_coef * float(np.linalg.norm(delta, ord=1))
            if action_arr.size > 0:
                steer_jerk_penalty = self.steer_jerk_penalty_coef * float(abs(delta[0]))
        self._prev_action = action_arr.copy()

        shaped_reward -= (lane_penalty + offroad_penalty + jerk_penalty + steer_jerk_penalty)

        if isinstance(info_dict, dict):
            info_dict["lane_center_penalty"] = lane_penalty
            info_dict["offroad_penalty"] = offroad_penalty
            info_dict["action_jerk_penalty"] = jerk_penalty
            info_dict["steer_jerk_penalty"] = steer_jerk_penalty
            info_dict["shaped_reward"] = shaped_reward

        if len(step_ret) == 5:
            return obs, shaped_reward, terminated, truncated, info_dict
        return obs, shaped_reward, done, info_dict


class ProgressRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, survival_reward: float, progress_reward_coef: float):
        super().__init__(env)
        self.survival_reward = float(survival_reward)
        self.progress_reward_coef = float(progress_reward_coef)
        self._last_progress: float = 0.0

    def reset(self, **kwargs):
        self._last_progress = 0.0
        return self.env.reset(**kwargs)

    def _extract_progress(self, info: Dict[str, Any]) -> float:
        value = info.get("route_completion", info.get("progress", 0.0)) if isinstance(info, dict) else 0.0
        try:
            progress = float(value)
        except (TypeError, ValueError):
            return self._last_progress
        if progress > 1.0:
            progress *= 0.01
        return float(np.clip(progress, 0.0, 1.0))

    def step(self, action):
        step_ret = self.env.step(action)
        if len(step_ret) == 5:
            obs, reward, terminated, truncated, info = step_ret
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = step_ret
            terminated, truncated = bool(done), False

        info_dict = info if isinstance(info, dict) else {}
        progress_now = self._extract_progress(info_dict)
        progress_delta = max(0.0, progress_now - self._last_progress)
        self._last_progress = progress_now

        shaped_reward = float(reward) + self.survival_reward + self.progress_reward_coef * progress_delta
        if isinstance(info_dict, dict):
            info_dict["survival_reward_bonus"] = self.survival_reward
            info_dict["progress_reward_bonus"] = self.progress_reward_coef * progress_delta
            info_dict["progress_value"] = progress_now
            info_dict["progress_delta"] = progress_delta
            info_dict["progress_shaped_reward"] = shaped_reward

        if len(step_ret) == 5:
            return obs, shaped_reward, terminated, truncated, info_dict
        return obs, shaped_reward, done, info_dict


ActionSmoothingWrapper = SmoothActionWrapper


class HybridBufferingWrapper(BufferingWrapper):
    def __init__(
        self,
        venv,
        reward_net,
        alpha_gail: float,
        beta_env: float,
        error_on_premature_reset: bool = True,
    ):
        super().__init__(venv=venv, error_on_premature_reset=error_on_premature_reset)
        self.reward_net = reward_net
        self.alpha_gail = float(alpha_gail)
        self.beta_env = float(beta_env)
        self._last_obs = None

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self._last_obs = np.asarray(obs, dtype=np.float32)
        return obs

    def step_wait(self):
        assert self._init_reset
        assert self._saved_acts is not None
        acts, self._saved_acts = self._saved_acts, None
        obs, env_rews, dones, infos = self.venv.step_wait()

        prev_obs = self._last_obs
        if prev_obs is None:
            prev_obs = np.asarray(obs, dtype=np.float32)

        obs_arr = np.asarray(obs, dtype=np.float32)
        acts_arr = np.asarray(acts, dtype=np.float32)
        dones_arr = np.asarray(dones, dtype=bool)
        env_rews_arr = np.asarray(env_rews, dtype=np.float32).reshape(-1)

        gail_rews = self.reward_net.predict_processed(prev_obs, acts_arr, obs_arr, dones_arr)
        gail_rews_arr = np.asarray(gail_rews, dtype=np.float32).reshape(-1)
        hybrid_rews = self.alpha_gail * gail_rews_arr + self.beta_env * env_rews_arr

        if isinstance(infos, list):
            for idx, info in enumerate(infos):
                if isinstance(info, dict):
                    info["gail_reward"] = float(gail_rews_arr[idx])
                    info["env_reward"] = float(env_rews_arr[idx])
                    info["hybrid_reward"] = float(hybrid_rews[idx])

        self.n_transitions += self.num_envs
        self._timesteps += 1
        ep_lens = self._timesteps[dones]
        if len(ep_lens) > 0:
            self._ep_lens += list(ep_lens)
        self._timesteps[dones] = 0

        finished_trajs = self._traj_accum.add_steps_and_auto_finish(
            acts_arr,
            obs,
            hybrid_rews,
            dones,
            infos,
        )
        self._trajectories.extend(finished_trajs)
        self._last_obs = obs_arr.copy()

        return obs, hybrid_rews, dones, infos


class HybridGAIL(GAIL):
    def __init__(
        self,
        *,
        alpha_gail: float,
        beta_env: float,
        **kwargs,
    ):
        self.alpha_gail = float(alpha_gail)
        self.beta_env = float(beta_env)
        super().__init__(**kwargs)

        self.venv_buffering = HybridBufferingWrapper(
            self.venv,
            reward_net=self.reward_train,
            alpha_gail=self.alpha_gail,
            beta_env=self.beta_env,
        )
        self.venv_wrapped = self.venv_buffering
        self.gen_callback = None
        self.venv_train = self.venv_wrapped
        self.gen_algo.set_env(self.venv_train)
        self.gen_algo.set_logger(self.logger)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GAIL on MetaDrive expert trajectories.")
    parser.add_argument("--demo-path", type=str, default="/root/autodl-tmp/GAIL/data/gail_expert_trajs_train.pkl", help="Path to expert trajectory pickle.")
    parser.add_argument("--save-dir", type=str, default="/root/autodl-tmp/GAIL/checkpoints/gail_policy", help="Directory to save PPO policy.")
    parser.add_argument("--total-timesteps", type=int, default=1000000, help="Total adversarial training timesteps.")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of vectorized environments. Use 1 for MetaDrive stability.")
    parser.add_argument("--train-start-seed", type=int, default=0, help="Training scenario start seed.")
    parser.add_argument("--train-num-scenarios", type=int, default=8000, help="Number of training scenarios.")
    parser.add_argument("--horizon", type=int, default=1200, help="Episode horizon in training env.")
    parser.add_argument("--ppo-n-steps", type=int, default=512, help="PPO rollout horizon per environment.")
    parser.add_argument("--ppo-batch-size", type=int, default=256, help="PPO mini-batch size.")
    parser.add_argument("--ppo-learning-rate", type=float, default=1e-4, help="PPO initial learning rate (linearly decayed).")
    parser.add_argument("--ppo-gamma", type=float, default=0.99, help="PPO discount factor.")
    parser.add_argument("--ppo-gae-lambda", type=float, default=0.95, help="PPO GAE lambda.")
    parser.add_argument("--ppo-ent-coef", type=float, default=0.03, help="PPO entropy coefficient.")
    parser.add_argument("--ppo-target-kl", type=float, default=0.03, help="PPO target KL for early stopping updates.")
    parser.add_argument("--demo-batch-size", type=int, default=32, help="Discriminator demo batch size.")
    parser.add_argument("--gen-replay-buffer-capacity", type=int, default=65536, help="Generator replay buffer size for GAIL.")
    parser.add_argument("--n-disc-updates-per-round", type=int, default=12, help="Discriminator updates per adversarial round.")
    parser.add_argument("--disc-weight-decay", type=float, default=1e-3, help="Discriminator optimizer weight decay (L2 regularization).")
    parser.add_argument("--disc-lr", type=float, default=1e-4, help="Discriminator optimizer learning rate.")
    parser.add_argument("--log-dir", type=str, default="/root/autodl-tmp/GAIL/logs/tensorboard/gail", help="TensorBoard log directory for PPO.")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2], help="Console verbosity for PPO/GAIL.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--checkpoint-freq", type=int, default=50000, help="Save a checkpoint every N timesteps (0 to disable).")
    parser.add_argument("--vec-normalize", action="store_true", default=True, help="Enable VecNormalize for observation normalization.")
    parser.add_argument("--no-vec-normalize", dest="vec_normalize", action="store_false", help="Disable VecNormalize.")
    parser.add_argument("--no-lr-decay", action="store_true", default=True, help="Disable linear learning rate decay (use constant LR).")
    parser.add_argument("--lr-decay", dest="no_lr_decay", action="store_false", help="Enable linear learning rate decay.")
    parser.add_argument("--bc-pretrain-batches", type=int, default=2000, help="BC warm-up batches before GAIL (0 to disable).")
    parser.add_argument("--ppo-finetune-timesteps", type=int, default=150000, help="PPO fine-tuning timesteps after GAIL (0 to disable).")
    parser.add_argument("--hybrid-alpha-gail", type=float, default=0.5, help="Weight alpha for discriminator reward in hybrid PPO fine-tuning.")
    parser.add_argument("--hybrid-beta-env", type=float, default=1.0, help="Weight beta for environment reward in hybrid PPO fine-tuning.")
    parser.add_argument("--action-smoothing-alpha", type=float, default=0.7, help="Low-pass smoothing factor for actions in [0,1).")
    parser.add_argument("--lane-center-penalty-coef", type=float, default=1.2, help="Reward penalty coefficient for lane center offset.")
    parser.add_argument("--out-of-road-penalty", type=float, default=8.0, help="Extra terminal penalty when out-of-road or out-of-lane occurs.")
    parser.add_argument("--action-jerk-penalty-coef", type=float, default=0.03, help="Reward penalty coefficient for action delta L1 norm.")
    parser.add_argument("--steer-jerk-penalty-coef", type=float, default=0.12, help="Reward penalty coefficient for steering delta magnitude.")
    parser.add_argument("--survival-reward", type=float, default=0.01, help="Small per-step survival reward bonus in environment shaping.")
    parser.add_argument("--progress-reward-coef", type=float, default=0.2, help="Reward coefficient for positive route-progress delta.")
    parser.add_argument(
        "--resume-model-path",
        type=str,
        default="",
        help="Optional PPO checkpoint path (.zip) to resume generator training from.",
    )

    parser.add_argument("--stage1-timesteps", type=int, default=300000, help="Timesteps for stage-1 warm-up in two-stage mode.")
    parser.add_argument("--stage2-timesteps", type=int, default=400000, help="Timesteps for stage-2 curriculum in two-stage mode.")
    parser.add_argument("--stage2-train-num-scenarios", type=int, default=8000, help="Training scenarios for stage-2.")
    parser.add_argument("--stage2-horizon", type=int, default=1200, help="Episode horizon for stage-2.")
    parser.add_argument("--stage2-ppo-n-steps", type=int, default=1024, help="PPO n-steps for stage-2.")
    parser.add_argument("--stage2-ppo-batch-size", type=int, default=256, help="PPO batch size for stage-2.")
    parser.add_argument("--stage2-ppo-learning-rate", type=float, default=5e-5, help="PPO learning rate for stage-2.")
    parser.add_argument("--stage2-ppo-ent-coef", type=float, default=0.02, help="PPO entropy coefficient for stage-2.")
    parser.add_argument("--stage2-ppo-target-kl", type=float, default=0.02, help="PPO target KL for stage-2.")
    parser.add_argument("--stage2-demo-batch-size", type=int, default=64, help="Discriminator demo batch size for stage-2.")
    parser.add_argument(
        "--stage2-disc-updates-per-round",
        type=int,
        default=12,
        help="Discriminator updates per adversarial round for stage-2.",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--two-stage", dest="two_stage", action="store_true", help="Run built-in two-stage training (default).")
    mode_group.add_argument("--single-stage", dest="two_stage", action="store_false", help="Run single-stage training only.")
    parser.set_defaults(two_stage=False)
    return parser.parse_args()


def make_training_env(
    start_seed: int,
    num_scenarios: int,
    horizon: int,
    action_smoothing_alpha: float,
    lane_center_penalty_coef: float,
    out_of_road_penalty: float,
    action_jerk_penalty_coef: float,
    steer_jerk_penalty_coef: float,
    survival_reward: float,
    progress_reward_coef: float,
) -> gym.Env:
    env_config = {
        "use_render": False,
        "image_observation": False,
        "manual_control": False,
        "start_seed": start_seed,
        "num_scenarios": num_scenarios,
        "horizon": horizon,
    }
    base_env = MetaDriveEnv(env_config)
    wrapped = MetaDriveResetSeedAdapter(base_env, start_seed=start_seed, num_scenarios=num_scenarios)
    wrapped = ProgressRewardWrapper(
        wrapped,
        survival_reward=survival_reward,
        progress_reward_coef=progress_reward_coef,
    )
    wrapped = LaneKeepingRewardWrapper(
        wrapped,
        lane_center_penalty_coef=lane_center_penalty_coef,
        out_of_road_penalty=out_of_road_penalty,
        action_jerk_penalty_coef=action_jerk_penalty_coef,
        steer_jerk_penalty_coef=steer_jerk_penalty_coef,
    )
    if action_smoothing_alpha > 0.0:
        wrapped = SmoothActionWrapper(wrapped, alpha=action_smoothing_alpha)
    return wrapped


def ensure_dir(dir_path: str) -> None:
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


def resolve_path(path: str) -> str:
    if os.path.exists(path):
        return path

    if not os.path.isabs(path):
        script_relative = os.path.join(SCRIPT_DIR, path)
        if os.path.exists(script_relative):
            return script_relative

    return path


def load_demonstrations(path: str) -> List[Trajectory]:
    path = resolve_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Demonstration file not found: {path}")
    with open(path, "rb") as f:
        demos = pickle.load(f)
    if not isinstance(demos, list) or len(demos) == 0:
        raise ValueError("Demonstration file is empty or invalid.")
    return demos


def main() -> None:
    args = parse_args()
    if args.ppo_n_steps <= 0 or args.ppo_batch_size <= 0:
        raise ValueError("ppo-n-steps and ppo-batch-size must be positive")
    if args.ppo_n_steps % args.ppo_batch_size != 0:
        raise ValueError("ppo-n-steps must be divisible by ppo-batch-size for stable minibatch updates")
    if args.stage2_ppo_n_steps <= 0 or args.stage2_ppo_batch_size <= 0:
        raise ValueError("stage2-ppo-n-steps and stage2-ppo-batch-size must be positive")
    if args.stage2_ppo_n_steps % args.stage2_ppo_batch_size != 0:
        raise ValueError("stage2-ppo-n-steps must be divisible by stage2-ppo-batch-size")
    if args.two_stage and (args.stage1_timesteps <= 0 or args.stage2_timesteps <= 0):
        raise ValueError("stage1-timesteps and stage2-timesteps must be positive in two-stage mode")
    if not (0.0 <= args.action_smoothing_alpha < 1.0):
        raise ValueError("action-smoothing-alpha must be in [0,1)")
    if args.bc_pretrain_batches < 0:
        raise ValueError("bc-pretrain-batches must be >= 0")
    if args.ppo_finetune_timesteps < 0:
        raise ValueError("ppo-finetune-timesteps must be >= 0")
    if args.hybrid_alpha_gail < 0 or args.hybrid_beta_env < 0:
        raise ValueError("hybrid reward weights must be >= 0")
    if args.lane_center_penalty_coef < 0 or args.out_of_road_penalty < 0:
        raise ValueError("lane/out-of-road penalty coefficients must be >= 0")
    if args.action_jerk_penalty_coef < 0 or args.steer_jerk_penalty_coef < 0:
        raise ValueError("jerk penalty coefficients must be >= 0")
    if args.survival_reward < 0 or args.progress_reward_coef < 0:
        raise ValueError("survival/progress reward coefficients must be >= 0")

    save_dir = resolve_path(args.save_dir) if os.path.isabs(args.save_dir) else os.path.join(SCRIPT_DIR, args.save_dir)
    log_dir = resolve_path(args.log_dir) if os.path.isabs(args.log_dir) else os.path.join(SCRIPT_DIR, args.log_dir)
    ensure_dir(save_dir)
    ensure_dir(log_dir)

    demos = load_demonstrations(args.demo_path)
    resume_model_path = ""
    if args.resume_model_path:
        resume_model_path = resolve_path(args.resume_model_path)
        if not os.path.exists(resume_model_path):
            raise FileNotFoundError(f"Resume model not found: {resume_model_path}")

    requested_n_envs = max(1, int(args.n_envs))

    def train_one_phase(
        phase_name: str,
        *,
        timesteps: int,
        train_num_scenarios: int,
        horizon: int,
        ppo_n_steps: int,
        ppo_batch_size: int,
        ppo_learning_rate: float,
        ppo_ent_coef: float,
        ppo_target_kl: float,
        demo_batch_size: int,
        disc_updates_per_round: int,
        resume_path: str,
        phase_log_dir: str,
        phase_save_dir: str,
    ) -> str:
        venv = None
        try:
            def _build_venv(n_envs: int):
                raw_venv = make_vec_env(
                    lambda: make_training_env(
                        args.train_start_seed,
                        train_num_scenarios,
                        horizon,
                        args.action_smoothing_alpha,
                        args.lane_center_penalty_coef,
                        args.out_of_road_penalty,
                        args.action_jerk_penalty_coef,
                        args.steer_jerk_penalty_coef,
                        args.survival_reward,
                        args.progress_reward_coef,
                    ),
                    n_envs=n_envs,
                )
                if args.vec_normalize:
                    return VecNormalize(raw_venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
                return raw_venv

            venv = _build_venv(requested_n_envs)
            try:
                venv.reset()
            except AssertionError as err:
                if requested_n_envs > 1:
                    print(
                        f"[Warning] Multi-env reset failed with n_envs={requested_n_envs}: {err}. "
                        "Falling back to n_envs=1 for stable MetaDrive training."
                    )
                    venv.close()
                    venv = _build_venv(1)
                    venv.reset()
                else:
                    raise

            lr = ppo_learning_rate if args.no_lr_decay else linear_schedule(ppo_learning_rate)

            if resume_path:
                print(f"[Info][{phase_name}] Resuming PPO generator from: {resume_path}")
                gen_algo = PPO.load(
                    resume_path,
                    env=venv,
                    seed=args.seed,
                    n_steps=ppo_n_steps,
                    batch_size=ppo_batch_size,
                    gamma=args.ppo_gamma,
                    gae_lambda=args.ppo_gae_lambda,
                    ent_coef=ppo_ent_coef,
                    target_kl=ppo_target_kl,
                    learning_rate=lr,
                    tensorboard_log=phase_log_dir,
                    verbose=args.verbose,
                )
                gen_algo.set_env(venv)

                if gen_algo.rollout_buffer.buffer_size != ppo_n_steps:
                    gen_algo.n_steps = ppo_n_steps
                    gen_algo.batch_size = ppo_batch_size
                    gen_algo.rollout_buffer = gen_algo.rollout_buffer_class(
                        gen_algo.n_steps,
                        gen_algo.observation_space,
                        gen_algo.action_space,
                        device=gen_algo.device,
                        gamma=gen_algo.gamma,
                        gae_lambda=gen_algo.gae_lambda,
                        n_envs=gen_algo.n_envs,
                        **gen_algo.rollout_buffer_kwargs,
                    )
            else:
                gen_algo = PPO(
                    policy="MlpPolicy",
                    env=venv,
                    seed=args.seed,
                    n_steps=ppo_n_steps,
                    batch_size=ppo_batch_size,
                    learning_rate=lr,
                    gamma=args.ppo_gamma,
                    gae_lambda=args.ppo_gae_lambda,
                    ent_coef=ppo_ent_coef,
                    target_kl=ppo_target_kl,
                    policy_kwargs={"net_arch": [256, 256, 256]},
                    tensorboard_log=phase_log_dir,
                    verbose=args.verbose,
                )

            reward_net = BasicShapedRewardNet(
                venv.observation_space,
                venv.action_space,
                normalize_input_layer=RunningNorm,
                use_state=True,
                use_action=True,
                use_next_state=True,
                use_done=False,
            )

            if args.bc_pretrain_batches > 0:
                print(f"[Train][{phase_name}] BC warm-up: batches={args.bc_pretrain_batches}")
                flattened_demos = rollout.flatten_trajectories(demos)
                bc_trainer = BC(
                    observation_space=venv.observation_space,
                    action_space=venv.action_space,
                    demonstrations=flattened_demos,
                    policy=gen_algo.policy,
                    rng=np.random.default_rng(args.seed),
                    batch_size=min(ppo_batch_size, 256),
                )
                bc_trainer.train(n_batches=args.bc_pretrain_batches)
                print(f"[Train][{phase_name}] BC warm-up completed")

            trainer = HybridGAIL(
                demonstrations=demos,
                demo_batch_size=demo_batch_size,
                gen_replay_buffer_capacity=args.gen_replay_buffer_capacity,
                n_disc_updates_per_round=disc_updates_per_round,
                venv=venv,
                gen_algo=gen_algo,
                reward_net=reward_net,
                alpha_gail=args.hybrid_alpha_gail,
                beta_env=args.hybrid_beta_env,
                allow_variable_horizon=True,
                disc_opt_kwargs={"lr": args.disc_lr, "weight_decay": args.disc_weight_decay},
                gen_train_timesteps=ppo_n_steps,
            )

            def _checkpoint_cb(round_no: int) -> None:
                """Called by imitation at end of each round; round_no in range(n_rounds)."""
                if args.checkpoint_freq <= 0:
                    return
                timesteps_so_far = (round_no + 1) * ppo_n_steps
                if timesteps_so_far % args.checkpoint_freq != 0:
                    return
                ckpt_dir = os.path.join(phase_save_dir, "ckpts") if os.path.isdir(phase_save_dir) else f"{phase_save_dir}_ckpts"
                ensure_dir(ckpt_dir)
                path = os.path.join(ckpt_dir, f"gail_{phase_name}_r{round_no + 1}_{timesteps_so_far}_steps.zip")
                gen_algo.save(path)
                if args.vec_normalize and isinstance(venv, VecNormalize):
                    vn_path = path.replace(".zip", "_vecnormalize.pkl")
                    venv.save(vn_path)
                print(f"[Train][{phase_name}] Checkpoint saved: {path}")

            print("=" * 80)
            print(f"[Train][{phase_name}] Start: timesteps={timesteps}, scenarios={train_num_scenarios}, horizon={horizon}")
            print(f"[Train][{phase_name}] VecNormalize={args.vec_normalize}, LR_decay={not args.no_lr_decay}, "
                f"replay_buf={args.gen_replay_buffer_capacity}, disc_lr={args.disc_lr}, disc_wd={args.disc_weight_decay}")
            print(
                f"[Train][{phase_name}] Reward shaping: lane_coef={args.lane_center_penalty_coef}, "
                f"offroad_penalty={args.out_of_road_penalty}, action_jerk_coef={args.action_jerk_penalty_coef}, "
                f"steer_jerk_coef={args.steer_jerk_penalty_coef}, smooth_alpha={args.action_smoothing_alpha}, "
                f"survival_reward={args.survival_reward}, progress_coef={args.progress_reward_coef}"
            )
            print(
                f"[Train][{phase_name}] Hybrid reward: alpha_gail={args.hybrid_alpha_gail}, "
                f"beta_env={args.hybrid_beta_env}"
            )
            print("=" * 80)
            trainer.train(total_timesteps=timesteps, callback=_checkpoint_cb)

            if args.ppo_finetune_timesteps > 0:
                print(f"[Train][{phase_name}] PPO fine-tune (env reward): timesteps={args.ppo_finetune_timesteps}")
                gen_algo.set_env(venv)
                gen_algo.learn(total_timesteps=args.ppo_finetune_timesteps, reset_num_timesteps=False)

            gen_algo.save(phase_save_dir)
            print(f"[Train][{phase_name}] Saved PPO policy to: {phase_save_dir}")

            if args.vec_normalize and isinstance(venv, VecNormalize):
                vec_norm_path = f"{phase_save_dir}_vecnormalize.pkl"
                venv.save(vec_norm_path)
                print(f"[Train][{phase_name}] Saved VecNormalize stats to: {vec_norm_path}")

            return f"{phase_save_dir}.zip"
        finally:
            if venv is not None:
                venv.close()

    if args.two_stage:
        stage1_save_dir = f"{save_dir}_stage1"
        stage1_log_dir = os.path.join(log_dir, "stage1")
        stage2_log_dir = os.path.join(log_dir, "stage2")
        ensure_dir(stage1_log_dir)
        ensure_dir(stage2_log_dir)

        stage1_resume = resume_model_path
        stage1_ckpt = train_one_phase(
            "stage1",
            timesteps=args.stage1_timesteps,
            train_num_scenarios=args.train_num_scenarios,
            horizon=args.horizon,
            ppo_n_steps=args.ppo_n_steps,
            ppo_batch_size=args.ppo_batch_size,
            ppo_learning_rate=args.ppo_learning_rate,
            ppo_ent_coef=args.ppo_ent_coef,
            ppo_target_kl=args.ppo_target_kl,
            demo_batch_size=args.demo_batch_size,
            disc_updates_per_round=args.n_disc_updates_per_round,
            resume_path=stage1_resume,
            phase_log_dir=stage1_log_dir,
            phase_save_dir=stage1_save_dir,
        )

        _ = train_one_phase(
            "stage2",
            timesteps=args.stage2_timesteps,
            train_num_scenarios=args.stage2_train_num_scenarios,
            horizon=args.stage2_horizon,
            ppo_n_steps=args.stage2_ppo_n_steps,
            ppo_batch_size=args.stage2_ppo_batch_size,
            ppo_learning_rate=args.stage2_ppo_learning_rate,
            ppo_ent_coef=args.stage2_ppo_ent_coef,
            ppo_target_kl=args.stage2_ppo_target_kl,
            demo_batch_size=args.stage2_demo_batch_size,
            disc_updates_per_round=args.stage2_disc_updates_per_round,
            resume_path=stage1_ckpt,
            phase_log_dir=stage2_log_dir,
            phase_save_dir=save_dir,
        )
        print("=" * 80)
        print(f"Two-stage GAIL training completed. Final PPO policy saved to: {save_dir}")
        print("=" * 80)
    else:
        _ = train_one_phase(
            "single",
            timesteps=args.total_timesteps,
            train_num_scenarios=args.train_num_scenarios,
            horizon=args.horizon,
            ppo_n_steps=args.ppo_n_steps,
            ppo_batch_size=args.ppo_batch_size,
            ppo_learning_rate=args.ppo_learning_rate,
            ppo_ent_coef=args.ppo_ent_coef,
            ppo_target_kl=args.ppo_target_kl,
            demo_batch_size=args.demo_batch_size,
            disc_updates_per_round=args.n_disc_updates_per_round,
            resume_path=resume_model_path,
            phase_log_dir=log_dir,
            phase_save_dir=save_dir,
        )
        print("=" * 80)
        print(f"Single-stage GAIL training completed. PPO policy saved to: {save_dir}")
        print("=" * 80)


if __name__ == "__main__":
    main()
