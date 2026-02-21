import os
import pickle
import time
import warnings
import numpy as np
import metadrive

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

from metadrive import MetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.version import VERSION as METADRIVE_VERSION


def ensure_headless_asset_stub():
    asset_root = os.path.join(os.path.dirname(metadrive.__file__), "assets")
    version_path = os.path.join(asset_root, "version.txt")
    grass_path = os.path.join(asset_root, "textures", "grass1", "GroundGrassGreen002_COL_1K.jpg")

    if os.path.exists(version_path) and os.path.exists(grass_path):
        return

    os.makedirs(os.path.dirname(grass_path), exist_ok=True)
    with open(version_path, "w", encoding="utf-8") as f:
        f.write(f"{METADRIVE_VERSION}\n")
    open(grass_path, "ab").close()
    print("[INFO] 检测到离线环境，已创建 MetaDrive headless 资源占位文件。")


def collect_idm_expert_data(
    target_steps=30000,
    use_render=False,
    map_name="SCO",
    traffic_density=0.12,
    start_seed=42,
    num_scenarios=1,
    save_path="data/human_expert.pkl",
):
    os.makedirs("data", exist_ok=True)
    save_dir = os.path.dirname(save_path) or "."
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    obs_memmap_path = os.path.join(save_dir, f"human_expert_obs_{run_tag}.memmap")
    action_memmap_path = os.path.join(save_dir, f"human_expert_actions_{run_tag}.memmap")

    config = {
        "use_render": use_render,
        "manual_control": False,
        "agent_policy": IDMPolicy,
        "traffic_density": traffic_density,
        "map": map_name,
        "num_scenarios": num_scenarios,
        "start_seed": start_seed,
    }

    if not use_render:
        ensure_headless_asset_stub()

    env = MetaDriveEnv(config)
    obs_mm = None
    action_mm = None

    print("=" * 68)
    print("开始用 IDM 自动采集专家数据（无需手动驾驶）")
    print(
        f"目标样本: {target_steps} | map={map_name} | traffic_density={traffic_density} | "
        f"start_seed={start_seed} | num_scenarios={num_scenarios}"
    )
    print("=" * 68)

    obs, info = env.reset()
    total_steps = 0
    episode_count = 0

    try:
        while total_steps < target_steps:
            # IDM 接管时，传入占位动作即可，环境内部会使用 IDMPolicy 输出
            next_obs, reward, terminated, truncated, info = env.step([0, 0])

            if hasattr(env, "agent") and hasattr(env.agent, "last_action"):
                real_action = np.asarray(env.agent.last_action, dtype=np.float32)
            else:
                real_action = np.asarray(env.vehicle.last_action, dtype=np.float32)

            if obs_mm is None:
                obs_shape = np.asarray(obs).shape
                action_dim = real_action.shape[0]
                obs_mm = np.memmap(
                    obs_memmap_path,
                    mode="w+",
                    dtype=np.float16,
                    shape=(target_steps,) + obs_shape,
                )
                action_mm = np.memmap(
                    action_memmap_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(target_steps, action_dim),
                )

            obs_mm[total_steps] = np.asarray(obs, dtype=np.float16)
            action_mm[total_steps] = real_action
            total_steps += 1
            obs = next_obs

            if use_render:
                env.render(
                    text={
                        "Mode": "IDM Auto Expert",
                        "Collected Steps": f"{total_steps}/{target_steps}",
                        "Episode": episode_count,
                    }
                )

            if total_steps % 500 == 0:
                print(f"进度: {total_steps}/{target_steps}")

            if terminated or truncated:
                episode_count += 1
                obs, info = env.reset()

    except KeyboardInterrupt:
        print("\n用户中断采集，正在保存已收集数据...")
    finally:
        env.close()

    if total_steps == 0 or obs_mm is None or action_mm is None:
        print("未收集到数据。")
        return

    obs_mm.flush()
    action_mm.flush()

    data = {
        "format": "memmap",
        "num_samples": int(total_steps),
        "obs_path": obs_memmap_path,
        "actions_path": action_memmap_path,
        "obs_dtype": "float16",
        "actions_dtype": "float32",
        "obs_shape": (int(total_steps),) + tuple(obs_mm.shape[1:]),
        "actions_shape": (int(total_steps),) + tuple(action_mm.shape[1:]),
    }

    with open(save_path, "wb") as f:
        pickle.dump(data, f)

    actions = np.memmap(
        action_memmap_path,
        mode="r",
        dtype=np.float32,
        shape=tuple(data["actions_shape"]),
    )
    throttle_neg_ratio = float((actions[:, 1] < 0).mean())
    print("=" * 68)
    print(f"数据已保存到 {save_path}")
    print(f"总样本数: {total_steps} | episode数: {episode_count}")
    print(
        f"动作统计: steer(mean/std)=({actions[:,0].mean():.3f}/{actions[:,0].std():.3f}), "
        f"throttle(mean/std)=({actions[:,1].mean():.3f}/{actions[:,1].std():.3f}), "
        f"throttle<0比例={throttle_neg_ratio:.3f}"
    )
    print("=" * 68)


if __name__ == "__main__":
    collect_idm_expert_data()
