import os
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from metadrive import MetaDriveEnv


class RewardLoggingCallback(BaseCallback):
    def __init__(self, save_path="checkpoints/best_model", rolling_window=50, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.episode_count = 0
        self.best_episode_reward = float("-inf")
        self.recent_rewards = deque(maxlen=rolling_window)
        self.recent_lengths = deque(maxlen=rolling_window)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep_info = info.get("episode")
            if ep_info is None:
                continue

            ep_reward = float(ep_info["r"])
            ep_len = int(ep_info["l"])
            self.episode_count += 1
            self.recent_rewards.append(ep_reward)
            self.recent_lengths.append(ep_len)
            rolling_reward = sum(self.recent_rewards) / len(self.recent_rewards)
            rolling_length = sum(self.recent_lengths) / len(self.recent_lengths)
            print(
                f"[Train] Episode {self.episode_count} | Reward: {ep_reward:.2f} | "
                f"Length: {ep_len} | Mean{len(self.recent_rewards)}R: {rolling_reward:.2f} | "
                f"Mean{len(self.recent_lengths)}L: {rolling_length:.1f} | Timesteps: {self.num_timesteps}"
            )

            if ep_reward > self.best_episode_reward:
                self.best_episode_reward = ep_reward
                self.model.save(self.save_path)
                print(f"🏆 新最佳训练回合奖励: {ep_reward:.2f}，已保存 {self.save_path}.zip")

        return True

def make_env(traffic_density, start_seed, num_scenarios):
    def _init():
        return MetaDriveEnv({
            "use_render": False,
            "traffic_density": traffic_density,
            "map": "SCO",
            "num_scenarios": num_scenarios,
            "start_seed": start_seed,
        })
    return _init


def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func

def train_rl_finetune():
    # 0. 检查 BC 模型是否存在
    bc_model_path = "checkpoints/bc_policy.zip"
    if not os.path.exists(bc_model_path):
        print(f"❌ 错误：找不到文件 {bc_model_path}。请先运行第二步 2_train_bc.py！")
        return

    print("🚀 正在初始化训练环境...")
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # MetaDrive 在 Windows/单进程下通常更稳定地运行为单环境
    num_envs = 1

    print(f"正在加载 BC 预训练权重: {bc_model_path} ...")
    try:
        bc_model = PPO.load("checkpoints/bc_policy")
    except Exception as e:
        print(f"❌ 加载 BC 模型失败: {e}")
        return

    print("✅ BC 权重加载成功！开始在同一张图上分阶段强化学习 (RL) 微调...")
    stages = [
        {"name": "stage1_same_map_easy", "traffic_density": 0.04, "timesteps": 180_000},
        {"name": "stage2_same_map_medium", "traffic_density": 0.08, "timesteps": 150_000},
        {"name": "stage3_same_map_hard", "traffic_density": 0.12, "timesteps": 120_000},
    ]

    policy_state = {k: v.detach().cpu().clone() for k, v in bc_model.policy.state_dict().items()}
    model = None
    train_env = None

    for stage in stages:
        stage_name = stage["name"]
        traffic_density = stage["traffic_density"]
        timesteps = stage["timesteps"]

        print(f"\n===== {stage_name} | traffic_density={traffic_density} | timesteps={timesteps} =====")

        if train_env is not None:
            train_env.close()
        train_env = DummyVecEnv([
            make_env(traffic_density=traffic_density, start_seed=42, num_scenarios=1)
            for i in range(num_envs)
        ])
        train_env = VecMonitor(train_env)

        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=linear_schedule(7e-5),
            n_steps=1024,
            batch_size=256,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=8e-4,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs={"log_std_init": -1.0},
            verbose=1,
            tensorboard_log="logs/tb",
        )
        model.policy.load_state_dict(policy_state)

        checkpoint_callback = CheckpointCallback(
            save_freq=20_000,
            save_path="checkpoints",
            name_prefix=f"{stage_name}_rl",
        )
        reward_logging_callback = RewardLoggingCallback(save_path="checkpoints/best_model", rolling_window=50)

        try:
            model.learn(
                total_timesteps=timesteps,
                reset_num_timesteps=True,
                callback=[checkpoint_callback, reward_logging_callback],
            )
            policy_state = {k: v.detach().cpu().clone() for k, v in model.policy.state_dict().items()}
        except KeyboardInterrupt:
            print("用户强制停止训练，正在保存当前模型...")
            break

    # 4. 保存最终模型
    save_path = "checkpoints/rl_final_model"
    if model is not None:
        model.save(save_path)
    print(f"✅ RL 微调完成，最终模型已保存至 {save_path}.zip")
    if os.path.exists("checkpoints/best_model.zip"):
        print("🏆 训练回合奖励最佳模型已保存至 checkpoints/best_model.zip")
    
    # 关闭环境
    if train_env is not None:
        train_env.close()

if __name__ == "__main__":
    train_rl_finetune()