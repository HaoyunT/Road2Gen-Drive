import argparse
from collections import defaultdict
import os
import pickle
import random
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect multi-scene expert data from MetaDrive using IDMPolicy (save successful episodes only).")
    parser.add_argument("--output", type=str, default="/root/autodl-tmp/GAIL/data/expert_train.pkl", help="Output pickle path.")
    parser.add_argument("--seed-start", type=int, default=0, help="Start map seed (inclusive).")
    parser.add_argument("--seed-end", type=int, default=799, help="End map seed (inclusive).")
    parser.add_argument("--min-steps-per-scene", type=int, default=360, help="Minimum effective steps per scene.")
    parser.add_argument("--max-steps-per-scene", type=int, default=700, help="Maximum effective steps per scene.")
    parser.add_argument("--target-frames", type=int, default=300000, help="Target total effective frames.")
    parser.add_argument("--horizon", type=int, default=1200, help="Episode horizon.")
    parser.add_argument("--static-speed-threshold", type=float, default=1.0, help="Filter low-speed frames below this threshold.")
    parser.add_argument("--episodes-per-seed", type=int, default=10, help="Run exactly this many episodes per seed.")
    parser.add_argument("--success-episodes-per-seed", type=int, default=8, help="Maximum successful episodes kept per map seed.")
    parser.add_argument("--max-steer-delta", type=float, default=0.35, help="Filter frames if |steer_t-steer_{t-1}| exceeds this threshold (<=0 to disable).")
    parser.add_argument("--progress", action="store_true", help="Enable tqdm progress bars (disabled by default to reduce console load).")
    parser.add_argument("--print-every-seeds", type=int, default=10, help="Print one summary line every N processed seeds.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--env-batch-size", type=int, default=50, help="Number of scenarios per environment instance to amortize init cost.")
    parser.add_argument("--seed-list-file", type=str, default="", help="Optional file with one map seed per line. If provided, overrides seed-start/seed-end.")
    parser.add_argument("--keep-success-episodes", action="store_true", default=True, help="Keep successful episodes (default: enabled).")
    parser.add_argument("--no-keep-success-episodes", dest="keep_success_episodes", action="store_false", help="Disable keeping successful episodes.")
    parser.add_argument("--keep-failure-recovery", action="store_true", default=False, help="Keep recovery windows from failed episodes.")
    parser.add_argument("--no-keep-failure-recovery", dest="keep_failure_recovery", action="store_false", help="Disable keeping recovery windows from failed episodes.")
    parser.add_argument("--recovery-window", type=int, default=100, help="Number of last effective steps to keep from failed episodes.")
    parser.add_argument("--failure-recovery-ratio", type=float, default=1.0, help="Sampling ratio [0,1] for failed-episode recovery windows.")
    parser.add_argument("--target-success-ratio", type=float, default=0.7, help="Target fraction of success samples in [0,1] after auto balancing (default: 0.7 => success:failure=7:3).")
    parser.add_argument("--auto-balance-ratio", action="store_true", default=False, help="Auto rebalance success/failure-recovery samples to target-success-ratio.")
    parser.add_argument("--no-auto-balance-ratio", dest="auto_balance_ratio", action="store_false", help="Disable automatic success/failure ratio balancing.")
    parser.add_argument("--strict-quota-collection", action="store_true", default=True, help="Collect by strict success/failure quotas and stop once both are filled.")
    parser.add_argument("--no-strict-quota-collection", dest="strict_quota_collection", action="store_false", help="Disable strict quota collection and use post-hoc balancing instead.")
    parser.add_argument("--extra-transition-files", type=str, default="", help="Comma-separated list of external transition pickle files (human/perturbation data).")
    parser.add_argument("--extra-max-samples", type=int, default=50000, help="Max sampled transitions to add from external files.")
    return parser.parse_args()


def ensure_parent_dir(file_path: str) -> None:
    parent = os.path.dirname(file_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def load_seed_list(seed_list_file: str) -> List[int]:
    if not seed_list_file:
        return []
    if not os.path.exists(seed_list_file):
        raise FileNotFoundError(f"seed-list-file does not exist: {seed_list_file}")

    seeds: List[int] = []
    with open(seed_list_file, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            seeds.append(int(line))
    if len(seeds) == 0:
        raise ValueError(f"No valid seeds found in seed-list-file: {seed_list_file}")
    return seeds


def normalize_transition_record(trans: Dict[str, Any]) -> Dict[str, Any] | None:
    required_keys = ["obs", "act", "next_obs", "done"]
    if not all(k in trans for k in required_keys):
        return None
    episode_start = bool(trans.get("episode_start", False))
    info = trans.get("info", {}) if isinstance(trans.get("info", {}), dict) else {}
    return {
        "obs": np.asarray(trans["obs"], dtype=np.float32),
        "act": np.asarray(trans["act"], dtype=np.float32),
        "next_obs": np.asarray(trans["next_obs"], dtype=np.float32),
        "done": bool(trans["done"]),
        "episode_start": episode_start,
        "map_seed": int(trans.get("map_seed", -1)),
        "scenario_seed": int(trans.get("scenario_seed", -1)),
        "speed": float(trans.get("speed", 0.0)),
        "info": info,
    }


def load_external_transition_pool(extra_transition_files: str) -> List[Dict[str, Any]]:
    if not extra_transition_files:
        return []
    files = [p.strip() for p in extra_transition_files.split(",") if p.strip()]
    pool: List[Dict[str, Any]] = []

    for file_path in files:
        if not os.path.exists(file_path):
            print(f"[Warning] External transition file not found: {file_path}")
            continue
        try:
            with open(file_path, "rb") as f:
                payload = pickle.load(f)
        except Exception as err:
            print(f"[Warning] Failed to load external transitions from {file_path}: {err}")
            continue

        raw_transitions = payload.get("transitions", []) if isinstance(payload, dict) else payload
        if not isinstance(raw_transitions, list):
            print(f"[Warning] External data format invalid (expect list/dict['transitions']): {file_path}")
            continue

        valid_count = 0
        for trans in raw_transitions:
            if not isinstance(trans, dict):
                continue
            normalized = normalize_transition_record(trans)
            if normalized is not None:
                pool.append(normalized)
                valid_count += 1
        print(f"[Info] Loaded external transitions: {file_path}, valid={valid_count}")

    return pool


def rebalance_transitions_by_success_ratio(
    success_transitions: List[Dict[str, Any]],
    failure_transitions: List[Dict[str, Any]],
    target_success_ratio: float,
    target_total: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    success_n = len(success_transitions)
    failure_n = len(failure_transitions)

    if target_total <= 0:
        return [], {
            "before_success": success_n,
            "before_failure_recovery": failure_n,
            "after_success": 0,
            "after_failure_recovery": 0,
        }

    def _sample_to_count(pool: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        if k <= 0:
            return []
        if len(pool) == 0:
            return []
        if k <= len(pool):
            return random.sample(pool, k)
        return [pool[random.randrange(len(pool))] for _ in range(k)]

    eps = 1e-8
    r = min(max(float(target_success_ratio), eps), 1.0 - eps)

    target_success_n = int(round(target_total * r))
    target_failure_n = target_total - target_success_n

    if success_n == 0 and failure_n > 0:
        target_success_n = 0
        target_failure_n = target_total
    elif failure_n == 0 and success_n > 0:
        target_success_n = target_total
        target_failure_n = 0
    elif success_n == 0 and failure_n == 0:
        target_success_n = 0
        target_failure_n = 0

    sampled_success = _sample_to_count(success_transitions, target_success_n)
    sampled_failure = _sample_to_count(failure_transitions, target_failure_n)
    merged = sampled_success + sampled_failure
    random.shuffle(merged)

    return merged, {
        "before_success": success_n,
        "before_failure_recovery": failure_n,
        "after_success": len(sampled_success),
        "after_failure_recovery": len(sampled_failure),
    }


def unpack_reset(reset_ret: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    if isinstance(reset_ret, tuple) and len(reset_ret) == 2:
        return reset_ret
    return reset_ret, {}


def unpack_step(step_ret: Any) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
    if not isinstance(step_ret, tuple):
        raise RuntimeError("Unexpected step return type from environment.")

    if len(step_ret) == 5:
        next_obs, reward, terminated, truncated, info = step_ret
        done = bool(terminated or truncated)
        return next_obs, float(reward), done, info
    if len(step_ret) == 4:
        next_obs, reward, done, info = step_ret
        return next_obs, float(reward), bool(done), info

    raise RuntimeError(f"Unexpected step return length: {len(step_ret)}")


def extract_speed(info: Dict[str, Any], fallback_obs: np.ndarray) -> float:
    for key in ["velocity", "speed", "ego_speed", "vehicle_speed"]:
        if key in info:
            try:
                return float(info[key])
            except (TypeError, ValueError):
                pass

    if isinstance(fallback_obs, np.ndarray) and fallback_obs.size > 0:
        try:
            return float(np.linalg.norm(fallback_obs[:2]))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def default_action(action_space: Any) -> Any:
    if hasattr(action_space, "shape") and action_space.shape is not None:
        return np.zeros(action_space.shape, dtype=np.float32)
    return action_space.sample()


def extract_executed_action(info: Dict[str, Any], fallback_action: Any) -> np.ndarray:
    raw = info.get("action", info.get("raw_action", fallback_action))
    try:
        arr = np.asarray(raw, dtype=np.float32)
        if arr.size == 0:
            return np.asarray(fallback_action, dtype=np.float32)
        return arr
    except Exception:
        return np.asarray(fallback_action, dtype=np.float32)


def is_success_episode(info: Dict[str, Any]) -> bool:
    arrive_dest = bool(info.get("arrive_dest", False) or info.get("success", False))
    has_crash = bool(
        info.get("crash", False)
        or info.get("crash_vehicle", False)
        or info.get("crash_object", False)
        or info.get("crash_human", False)
        or info.get("crash_sidewalk", False)
    )
    out_of_road = bool(info.get("out_of_road", False))
    return arrive_dest and (not has_crash) and (not out_of_road)


def episode_end_reason(info: Dict[str, Any]) -> str:
    if bool(info.get("arrive_dest", False) or info.get("success", False)):
        return "arrive_dest"
    if bool(info.get("out_of_road", False)):
        return "out_of_road"
    if bool(
        info.get("crash", False)
        or info.get("crash_vehicle", False)
        or info.get("crash_object", False)
        or info.get("crash_human", False)
        or info.get("crash_sidewalk", False)
    ):
        return "crash"
    if bool(info.get("max_step", False)):
        return "max_step"
    return "other"


def speed_bucket(speed: float) -> str:
    if speed < 2.0:
        return "[0,2)"
    if speed < 5.0:
        return "[2,5)"
    if speed < 10.0:
        return "[5,10)"
    if speed < 20.0:
        return "[10,20)"
    return "[20,+inf)"


def steer_bucket(action: np.ndarray) -> str:
    steer_abs = float(abs(action[0])) if action.size > 0 else 0.0
    if steer_abs < 0.05:
        return "[0,0.05)"
    if steer_abs < 0.15:
        return "[0.05,0.15)"
    if steer_abs < 0.30:
        return "[0.15,0.30)"
    if steer_abs < 0.60:
        return "[0.30,0.60)"
    return "[0.60,+inf)"


def _build_env(start_seed: int, num_scenarios: int, horizon: int) -> MetaDriveEnv:
    env_config = {
        "use_render": False,
        "image_observation": False,
        "manual_control": False,
        "agent_policy": IDMPolicy,
        "start_seed": start_seed,
        "num_scenarios": num_scenarios,
        "horizon": horizon,
    }
    return MetaDriveEnv(env_config)


def main() -> None:
    args = parse_args()
    if args.seed_end < args.seed_start:
        raise ValueError("seed-end must be >= seed-start")
    if args.max_steps_per_scene < args.min_steps_per_scene:
        raise ValueError("max-steps-per-scene must be >= min-steps-per-scene")
    if args.episodes_per_seed <= 0:
        raise ValueError("episodes-per-seed must be positive")
    if args.success_episodes_per_seed <= 0:
        raise ValueError("success-episodes-per-seed must be positive")
    if args.recovery_window <= 0:
        raise ValueError("recovery-window must be positive")
    if not (0.0 <= args.failure_recovery_ratio <= 1.0):
        raise ValueError("failure-recovery-ratio must be in [0,1]")
    if not (0.0 <= args.target_success_ratio <= 1.0):
        raise ValueError("target-success-ratio must be in [0,1]")

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    ensure_parent_dir(args.output)

    transitions = []
    success_transition_pool: List[Dict[str, Any]] = []
    failure_recovery_transition_pool: List[Dict[str, Any]] = []
    if args.keep_success_episodes and (not args.keep_failure_recovery):
        target_success_steps = int(args.target_frames)
        target_failure_steps = 0
    elif (not args.keep_success_episodes) and args.keep_failure_recovery:
        target_success_steps = 0
        target_failure_steps = int(args.target_frames)
    else:
        target_success_steps = int(round(args.target_frames * args.target_success_ratio))
        target_failure_steps = int(args.target_frames - target_success_steps)
    total_effective = 0
    static_filtered = 0
    jitter_filtered = 0
    total_episodes = 0
    successful_episodes = 0
    failed_episodes = 0
    kept_episodes = 0
    dropped_episodes = 0
    interrupted = False
    end_reason_counts: Dict[str, int] = defaultdict(int)
    kept_outcome_counts: Dict[str, int] = defaultdict(int)
    speed_bucket_counts: Dict[str, int] = defaultdict(int)
    steer_bucket_counts: Dict[str, int] = defaultdict(int)
    per_seed_stats: Dict[int, Dict[str, Any]] = {}

    all_scenario_seeds = []
    seeds = load_seed_list(args.seed_list_file) if args.seed_list_file else list(range(args.seed_start, args.seed_end + 1))
    random.shuffle(seeds)
    for map_seed in seeds:
        for episode_idx in range(args.episodes_per_seed):
            scenario_seed = int(map_seed * args.episodes_per_seed + episode_idx)
            all_scenario_seeds.append((map_seed, scenario_seed))

    env_batch = args.env_batch_size
    processed_seed_count = 0
    last_reported_seeds = set()
    env = None

    try:
        for batch_start in range(0, len(all_scenario_seeds), env_batch):
            if args.strict_quota_collection:
                if len(success_transition_pool) >= target_success_steps and len(failure_recovery_transition_pool) >= target_failure_steps:
                    break
            elif total_effective >= args.target_frames:
                break

            batch = all_scenario_seeds[batch_start : batch_start + env_batch]
            batch_scenario_seeds = [s for _, s in batch]
            batch_start_seed = min(batch_scenario_seeds)
            batch_num_scenarios = max(batch_scenario_seeds) - batch_start_seed + 1

            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            env = _build_env(batch_start_seed, batch_num_scenarios, args.horizon)

            for map_seed, scenario_seed in batch:
                if args.strict_quota_collection:
                    if len(success_transition_pool) >= target_success_steps and len(failure_recovery_transition_pool) >= target_failure_steps:
                        break
                elif total_effective >= args.target_frames:
                    break

                episode_transitions = []
                last_info: Dict[str, Any] = {}
                try:
                    reset_ret = env.reset(seed=scenario_seed)
                    obs, _ = unpack_reset(reset_ret)
                    done = False
                    episode_start = True
                    prev_expert_action: np.ndarray | None = None

                    while not done:
                        action = default_action(env.action_space)
                        step_ret = env.step(action)
                        next_obs, _, done, info = unpack_step(step_ret)
                        last_info = info
                        expert_action = extract_executed_action(info, action)

                        speed = extract_speed(info, next_obs)
                        keep_by_speed = speed >= args.static_speed_threshold
                        keep_by_jitter = True
                        if args.max_steer_delta > 0:
                            current_steer = float(expert_action[0]) if expert_action.size > 0 else 0.0
                            prev_steer = float(prev_expert_action[0]) if (prev_expert_action is not None and prev_expert_action.size > 0) else None
                            if prev_steer is not None and abs(current_steer - prev_steer) > float(args.max_steer_delta):
                                keep_by_jitter = False

                        if keep_by_speed and keep_by_jitter:
                            episode_transitions.append(
                                {
                                    "obs": np.asarray(obs, dtype=np.float32),
                                    "act": expert_action,
                                    "next_obs": np.asarray(next_obs, dtype=np.float32),
                                    "done": bool(done),
                                    "episode_start": bool(episode_start),
                                    "map_seed": int(map_seed),
                                    "scenario_seed": int(scenario_seed),
                                    "speed": float(speed),
                                    "info": {
                                        "arrive_dest": bool(info.get("arrive_dest", False)),
                                        "crash": bool(info.get("crash", False)),
                                        "crash_vehicle": bool(info.get("crash_vehicle", False)),
                                        "crash_object": bool(info.get("crash_object", False)),
                                        "out_of_road": bool(info.get("out_of_road", False)),
                                    },
                                }
                            )
                            episode_start = False
                        else:
                            if not keep_by_speed:
                                static_filtered += 1
                                if map_seed in per_seed_stats:
                                    per_seed_stats[map_seed]["static_filtered"] += 1
                            if keep_by_speed and (not keep_by_jitter):
                                jitter_filtered += 1
                                if map_seed in per_seed_stats:
                                    per_seed_stats[map_seed]["jitter_filtered"] += 1

                        prev_expert_action = np.asarray(expert_action, dtype=np.float32)

                        obs = next_obs

                    total_episodes += 1
                    success_episode = is_success_episode(last_info)
                    end_reason = episode_end_reason(last_info)
                    end_reason_counts[end_reason] += 1

                    if map_seed not in per_seed_stats:
                        per_seed_stats[map_seed] = {
                            "attempted_episodes": 0,
                            "successful_episodes": 0,
                            "failed_episodes": 0,
                            "effective_frames": 0,
                            "static_filtered": 0,
                            "jitter_filtered": 0,
                            "kept_success_episodes": 0,
                            "end_reason_buckets": defaultdict(int),
                        }
                    per_seed_stats[map_seed]["attempted_episodes"] += 1
                    per_seed_stats[map_seed]["end_reason_buckets"][end_reason] += 1

                    if success_episode:
                        successful_episodes += 1
                        per_seed_stats[map_seed]["successful_episodes"] += 1
                    else:
                        failed_episodes += 1
                        per_seed_stats[map_seed]["failed_episodes"] += 1

                    selected_transitions: List[Dict[str, Any]] = []
                    if success_episode and args.keep_success_episodes and len(episode_transitions) > 0:
                        if per_seed_stats[map_seed]["kept_success_episodes"] < args.success_episodes_per_seed:
                            if args.strict_quota_collection:
                                remaining = max(0, target_success_steps - len(success_transition_pool))
                                if remaining > 0:
                                    selected_transitions = episode_transitions[:remaining]
                            else:
                                selected_transitions = episode_transitions
                        if len(selected_transitions) > 0:
                            kept_outcome_counts["success_full"] += 1
                    elif (not success_episode) and args.keep_failure_recovery and len(episode_transitions) > 0:
                        if random.random() <= args.failure_recovery_ratio:
                            recovery_chunk = episode_transitions[-args.recovery_window :]
                            if args.strict_quota_collection:
                                remaining = max(0, target_failure_steps - len(failure_recovery_transition_pool))
                                if remaining > 0:
                                    selected_transitions = recovery_chunk[:remaining]
                            else:
                                selected_transitions = recovery_chunk
                            if selected_transitions:
                                selected_transitions[0]["episode_start"] = True
                                kept_outcome_counts["failure_recovery"] += 1

                    keep_episode = bool(len(selected_transitions) > 0)
                    if keep_episode:
                        transitions.extend(selected_transitions)
                        total_effective += len(selected_transitions)
                        per_seed_stats[map_seed]["effective_frames"] += len(selected_transitions)
                        kept_episodes += 1
                        if success_episode:
                            per_seed_stats[map_seed]["kept_success_episodes"] += 1
                            success_transition_pool.extend(selected_transitions)
                        else:
                            failure_recovery_transition_pool.extend(selected_transitions)
                        for transition in selected_transitions:
                            speed_bucket_counts[speed_bucket(float(transition["speed"]))] += 1
                            steer_bucket_counts[steer_bucket(np.asarray(transition["act"], dtype=np.float32))] += 1

                    if args.strict_quota_collection:
                        if len(success_transition_pool) >= target_success_steps and len(failure_recovery_transition_pool) >= target_failure_steps:
                            break
                    else:
                        dropped_episodes += 1

                except Exception as err:
                    print(f"[Warning] Failed on scenario_seed {scenario_seed} (map_seed {map_seed}): {err}")

                if map_seed not in last_reported_seeds:
                    last_reported_seeds.add(map_seed)
                    processed_seed_count += 1
                    if processed_seed_count % max(1, args.print_every_seeds) == 0:
                        success_rate = (successful_episodes / total_episodes) if total_episodes > 0 else 0.0
                        print(
                            f"[Collect] Seeds processed: {processed_seed_count}/{len(seeds)}, "
                            f"effective frames: {total_effective}, filtered static: {static_filtered}, filtered jitter: {jitter_filtered}, "
                            f"episode success rate: {success_rate:.2%}, "
                            f"last_seed={map_seed}"
                        )

    except KeyboardInterrupt:
        interrupted = True
        print("[Warning] Collection interrupted by user (KeyboardInterrupt). Saving collected data so far...")
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    rebalance_stats = {
        "before_success": len(success_transition_pool),
        "before_failure_recovery": len(failure_recovery_transition_pool),
        "after_success": len(success_transition_pool),
        "after_failure_recovery": len(failure_recovery_transition_pool),
    }
    if (not args.strict_quota_collection) and args.auto_balance_ratio:
        transitions, rebalance_stats = rebalance_transitions_by_success_ratio(
            success_transition_pool,
            failure_recovery_transition_pool,
            args.target_success_ratio,
            args.target_frames,
        )
        total_effective = len(transitions)
        after_success = rebalance_stats["after_success"]
        after_failure = rebalance_stats["after_failure_recovery"]
        denom = after_success + after_failure
        actual_ratio = (after_success / denom) if denom > 0 else 0.0
        print(
            f"[Info] Auto-balanced transitions: success={after_success}, "
            f"failure_recovery={after_failure}, success_ratio={actual_ratio:.2%}"
        )

    external_added = 0
    external_pool = load_external_transition_pool(args.extra_transition_files)
    if external_pool and args.extra_max_samples > 0:
        sample_k = min(len(external_pool), int(args.extra_max_samples))
        sampled = random.sample(external_pool, sample_k) if sample_k < len(external_pool) else external_pool
        transitions.extend(sampled)
        total_effective += len(sampled)
        external_added = len(sampled)
        print(f"[Info] Added external transitions: {external_added}")

    if args.strict_quota_collection:
        transitions = success_transition_pool + failure_recovery_transition_pool
        random.shuffle(transitions)
        total_effective = len(transitions)
        print(
            f"[Info] Strict quota collection done: success={len(success_transition_pool)}/{target_success_steps}, "
            f"failure_recovery={len(failure_recovery_transition_pool)}/{target_failure_steps}, total={total_effective}"
        )

    if total_effective > args.target_frames:
        transitions = random.sample(transitions, args.target_frames)
        total_effective = len(transitions)
        print(f"[Info] Trimmed transitions to target_frames={args.target_frames}")

    if total_effective == 0:
        raise RuntimeError("No effective transitions collected. Check MetaDrive/IDM setup and thresholds.")

    for seed in per_seed_stats:
        s = per_seed_stats[seed]
        attempted = s["attempted_episodes"]
        s["success_rate"] = float(s["successful_episodes"] / attempted) if attempted > 0 else 0.0
        s["end_reason_buckets"] = dict(s["end_reason_buckets"])

    payload = {
        "transitions": transitions,
        "meta": {
            "seed_range": [args.seed_start, args.seed_end],
            "target_frames": args.target_frames,
            "collected_frames": total_effective,
            "static_filtered": static_filtered,
            "static_speed_threshold": args.static_speed_threshold,
            "jitter_filtered": jitter_filtered,
            "max_steer_delta": args.max_steer_delta,
            "min_steps_per_scene": args.min_steps_per_scene,
            "max_steps_per_scene": args.max_steps_per_scene,
            "episodes_per_seed": args.episodes_per_seed,
            "success_episodes_per_seed": args.success_episodes_per_seed,
            "seed_list_file": args.seed_list_file,
            "interrupted": interrupted,
            "total_episodes": total_episodes,
            "successful_episodes": successful_episodes,
            "failed_episodes": failed_episodes,
            "kept_episodes": kept_episodes,
            "dropped_episodes": dropped_episodes,
            "keep_success_episodes": args.keep_success_episodes,
            "keep_failure_recovery": args.keep_failure_recovery,
            "recovery_window": args.recovery_window,
            "failure_recovery_ratio": args.failure_recovery_ratio,
            "target_success_ratio": args.target_success_ratio,
            "auto_balance_ratio": args.auto_balance_ratio,
            "strict_quota_collection": args.strict_quota_collection,
            "target_success_steps": target_success_steps,
            "target_failure_steps": target_failure_steps,
            "rebalance_stats": rebalance_stats,
            "external_added": external_added,
            "extra_transition_files": args.extra_transition_files,
            "extra_max_samples": args.extra_max_samples,
            "end_reason_buckets": dict(end_reason_counts),
            "kept_outcome_buckets": dict(kept_outcome_counts),
            "speed_buckets": dict(speed_bucket_counts),
            "steer_abs_buckets": dict(steer_bucket_counts),
            "per_seed_stats": per_seed_stats,
        },
    }

    with open(args.output, "wb") as f:
        pickle.dump(payload, f)

    print("=" * 80)
    print(f"Saved expert data to: {args.output}")
    print(f"Collected effective frames: {total_effective}")
    print(f"Filtered static frames: {static_filtered}")
    print(f"Filtered jitter frames: {jitter_filtered}")
    if total_episodes > 0:
        print(
            f"Episode stats - total: {total_episodes}, success: {successful_episodes}, "
            f"failed: {failed_episodes}, kept: {kept_episodes}, dropped: {dropped_episodes}, "
            f"success rate: {successful_episodes / total_episodes:.2%}"
        )
    print(f"End reason buckets: {dict(end_reason_counts)}")
    print(f"Kept outcome buckets: {dict(kept_outcome_counts)}")
    print(f"Speed buckets: {dict(speed_bucket_counts)}")
    print(f"|steer| buckets: {dict(steer_bucket_counts)}")
    print("Per-seed success rates:")
    for seed in sorted(per_seed_stats.keys()):
        seed_info = per_seed_stats[seed]
        print(
            f"  seed={seed}: success={seed_info['successful_episodes']}/{seed_info['attempted_episodes']} "
            f"({seed_info['success_rate']:.2%}), effective_frames={seed_info['effective_frames']}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
