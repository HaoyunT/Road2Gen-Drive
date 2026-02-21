import argparse
import os
import pickle
from typing import Any, Dict, List

import numpy as np
from imitation.data.types import Trajectory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert collected expert data into imitation Trajectory format.")
    parser.add_argument("--input", type=str, default="/root/autodl-tmp/GAIL/data/expert_train.pkl", help="Input expert transition file.")
    parser.add_argument("--output", type=str, default="/root/autodl-tmp/GAIL/data/gail_expert_trajs_train.pkl", help="Output trajectory pickle file.")
    parser.add_argument("--min-traj-len", type=int, default=32, help="Drop trajectories shorter than this length.")
    return parser.parse_args()


def ensure_parent_dir(file_path: str) -> None:
    parent = os.path.dirname(file_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def finalize_trajectory(
    trajs: List[Trajectory],
    obs_buf: List[np.ndarray],
    act_buf: List[np.ndarray],
    info_buf: List[Dict[str, Any]],
    terminal: bool,
) -> None:
    if len(act_buf) == 0:
        return

    if len(obs_buf) != len(act_buf) + 1:
        raise ValueError(
            f"Invalid trajectory lengths: len(obs)={len(obs_buf)}, len(acts)={len(act_buf)}; expected len(obs)=len(acts)+1."
        )

    trajs.append(
        Trajectory(
            obs=np.asarray(obs_buf, dtype=np.float32),
            acts=np.asarray(act_buf, dtype=np.float32),
            infos=info_buf,
            terminal=bool(terminal),
        )
    )


def main() -> None:
    args = parse_args()
    if args.min_traj_len < 2:
        raise ValueError("min-traj-len must be >= 2")

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file does not exist: {args.input}")

    ensure_parent_dir(args.output)

    with open(args.input, "rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict) or "transitions" not in payload:
        raise ValueError("Invalid input format. Expecting a dict with key 'transitions'.")

    transitions = payload["transitions"]
    if not transitions:
        raise ValueError("No transitions found in input file.")

    trajectories: List[Trajectory] = []
    dropped_short = 0
    obs_buf: List[np.ndarray] = []
    act_buf: List[np.ndarray] = []
    info_buf: List[Dict[str, Any]] = []

    for idx, trans in enumerate(transitions):
        required_keys = ["obs", "act", "next_obs", "done"]
        if not all(k in trans for k in required_keys):
            raise KeyError(f"Transition at index {idx} missing required keys: {required_keys}")

        episode_start = bool(trans.get("episode_start", False))
        if episode_start and len(act_buf) > 0:
            if len(act_buf) >= args.min_traj_len:
                finalize_trajectory(trajectories, obs_buf, act_buf, info_buf, terminal=False)
            else:
                dropped_short += 1
            obs_buf, act_buf, info_buf = [], [], []

        obs = np.asarray(trans["obs"], dtype=np.float32)
        act = np.asarray(trans["act"], dtype=np.float32)
        next_obs = np.asarray(trans["next_obs"], dtype=np.float32)
        done = bool(trans["done"])
        info = trans.get("info", {}) if isinstance(trans.get("info", {}), dict) else {}

        if len(obs_buf) == 0:
            obs_buf.append(obs)

        act_buf.append(act)
        info_buf.append(info)
        obs_buf.append(next_obs)

        if done:
            if len(act_buf) >= args.min_traj_len:
                finalize_trajectory(trajectories, obs_buf, act_buf, info_buf, terminal=True)
            else:
                dropped_short += 1
            obs_buf, act_buf, info_buf = [], [], []

    if len(act_buf) > 0:
        if len(act_buf) >= args.min_traj_len:
            finalize_trajectory(trajectories, obs_buf, act_buf, info_buf, terminal=False)
        else:
            dropped_short += 1

    if len(trajectories) == 0:
        raise RuntimeError("No valid trajectories were generated.")

    with open(args.output, "wb") as f:
        pickle.dump(trajectories, f)

    n_steps = int(sum(len(traj.acts) for traj in trajectories))
    print("=" * 80)
    print(f"Saved processed trajectories to: {args.output}")
    print(f"Trajectory count: {len(trajectories)}")
    print(f"Total transition steps: {n_steps}")
    print(f"Dropped short trajectories: {dropped_short}")
    print("=" * 80)


if __name__ == "__main__":
    main()
