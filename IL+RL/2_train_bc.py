import pickle
import os
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from metadrive import MetaDriveEnv
# from imitation.algorithms import bc # 这行其实用不到，因为我们是手写的BC循环，可以注释掉
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split


def print_action_stats(title, actions_np):
    steer = actions_np[:, 0]
    throttle = actions_np[:, 1]
    print(
        f"{title} | steer(mean/std/min/max)=({steer.mean():.3f}/{steer.std():.3f}/{steer.min():.3f}/{steer.max():.3f}) | "
        f"throttle(mean/std/min/max)=({throttle.mean():.3f}/{throttle.std():.3f}/{throttle.min():.3f}/{throttle.max():.3f}) | "
        f"throttle<0比例={float((throttle < 0).mean()):.3f}"
    )


class ExpertMemmapDataset(Dataset):
    def __init__(self, obs_np, actions_np):
        self.obs_np = obs_np
        self.actions_np = actions_np

    def __len__(self):
        return int(self.obs_np.shape[0])

    def __getitem__(self, idx):
        obs = torch.as_tensor(np.asarray(self.obs_np[idx], dtype=np.float32), dtype=torch.float32)
        act = torch.as_tensor(np.asarray(self.actions_np[idx], dtype=np.float32), dtype=torch.float32)
        return obs, act


def train_bc():
    # 0. 确保文件夹存在
    os.makedirs("checkpoints", exist_ok=True)

    # 1. 准备训练环境 (无需渲染)
    # 【修正1】environment_num 改为 num_scenarios
    env = DummyVecEnv([lambda: MetaDriveEnv({
        "use_render": False,
        "traffic_density": 0.12,
        "map": "SCO",
        "num_scenarios": 1, 
        "start_seed": 42
    })])
    
    # 2. 加载数据
    data_path = "data/human_expert.pkl"
    if not os.path.exists(data_path):
        print(f"❌ 错误：找不到文件 {data_path}。请先运行第一步录制数据！")
        return

    print(f"正在加载专家数据: {data_path} ...")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and data.get("format") == "memmap":
        expert_obs_np = np.memmap(
            data["obs_path"],
            mode="r",
            dtype=np.float16 if data["obs_dtype"] == "float16" else np.float32,
            shape=tuple(data["obs_shape"]),
        )
        expert_actions_np = np.memmap(
            data["actions_path"],
            mode="r",
            dtype=np.float32,
            shape=tuple(data["actions_shape"]),
        )
        sample_count = int(data["num_samples"])
    else:
        expert_obs_np = np.asarray(data["obs"], dtype=np.float32)
        expert_actions_np = np.asarray(data["actions"], dtype=np.float32)
        sample_count = len(expert_obs_np)

    expert_actions_np = np.clip(expert_actions_np, -1.0, 1.0)
    print(f"数据加载完成，共 {sample_count} 条样本。")

    # 3. 初始化 PPO 策略网络 (作为 Student)
    # 此时主要利用它的网络结构 (Policy Network)
    student = PPO(MlpPolicy, env, verbose=1)
    
    # 4. 配置 BC 训练器
    batch_size = 128
    if isinstance(data, dict) and data.get("format") == "memmap":
        dataset = ExpertMemmapDataset(expert_obs_np, expert_actions_np)
    else:
        expert_obs = torch.as_tensor(expert_obs_np, dtype=torch.float32)
        expert_actions = torch.as_tensor(expert_actions_np, dtype=torch.float32)
        dataset = TensorDataset(expert_obs, expert_actions)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    student.policy.log_std.data.fill_(-2.0)
    student.policy.log_std.requires_grad_(False)

    actor_parameters = (
        list(student.policy.features_extractor.parameters())
        + list(student.policy.mlp_extractor.policy_net.parameters())
        + list(student.policy.action_net.parameters())
    )
    optimizer = torch.optim.AdamW(actor_parameters, lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
        min_lr=1e-5,
    )

    print_action_stats("专家动作统计", expert_actions_np)
    
    print("🚀 开始模仿学习 (BC) 预训练...")
    student.policy.train() # 开启训练模式 (启用 Dropout/BatchNorm 等)
    
    # 训练 200 个 Epoch + 早停
    max_epochs = 200
    patience = 20
    patience_count = 0
    best_val_loss = np.inf
    best_state = None

    for epoch in range(max_epochs):
        total_train_loss = 0.0
        for batch_obs, batch_act in train_loader:
            # 确保数据在正确的设备上 (CPU 或 CUDA)
            batch_obs = batch_obs.to(student.device)
            batch_act = batch_act.to(student.device)
            
            optimizer.zero_grad()

            dist = student.policy.get_distribution(batch_obs)
            pred_act = dist.mode()

            steer_loss = torch.nn.functional.smooth_l1_loss(pred_act[:, 0], batch_act[:, 0])
            throttle_loss = torch.nn.functional.smooth_l1_loss(pred_act[:, 1], batch_act[:, 1])

            sign_target = (batch_act[:, 1] > 0).float()
            sign_logit = pred_act[:, 1] / 0.25
            throttle_sign_loss = torch.nn.functional.binary_cross_entropy_with_logits(sign_logit, sign_target)

            loss = steer_loss + 1.6 * throttle_loss + 0.3 * throttle_sign_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_parameters, max_norm=0.5)
            optimizer.step()
            
            total_train_loss += loss.item()

        student.policy.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for val_obs, val_act in val_loader:
                val_obs = val_obs.to(student.device)
                val_act = val_act.to(student.device)
                val_dist = student.policy.get_distribution(val_obs)
                val_pred = val_dist.mode()

                val_steer_loss = torch.nn.functional.smooth_l1_loss(val_pred[:, 0], val_act[:, 0])
                val_throttle_loss = torch.nn.functional.smooth_l1_loss(val_pred[:, 1], val_act[:, 1])
                val_sign_target = (val_act[:, 1] > 0).float()
                val_sign_logit = val_pred[:, 1] / 0.25
                val_throttle_sign_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    val_sign_logit,
                    val_sign_target,
                )

                val_loss = val_steer_loss + 1.6 * val_throttle_loss + 0.3 * val_throttle_sign_loss
                total_val_loss += val_loss.item()

        avg_train_loss = total_train_loss / max(1, len(train_loader))
        avg_val_loss = total_val_loss / max(1, len(val_loader))
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{max_epochs} | Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in student.policy.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= patience:
            print(f"⏹️ 验证集连续 {patience} 轮无提升，提前停止训练。")
            break

        if current_lr <= 1.01e-5 and patience_count >= 8:
            print("⏹️ 学习率已降至最小且验证集长期无提升，提前停止训练。")
            break

        student.policy.train()

    if best_state is not None:
        student.policy.load_state_dict(best_state)
        print(f"✅ 已恢复验证集最优参数，Best Val Loss: {best_val_loss:.6f}")

    student.policy.eval()
    with torch.no_grad():
        probe_count = min(2048, sample_count)
        probe_obs_np = np.asarray(expert_obs_np[:probe_count], dtype=np.float32)
        probe_obs = torch.as_tensor(probe_obs_np, dtype=torch.float32).to(student.device)
        pred_action = student.policy.get_distribution(probe_obs).mode().cpu().numpy()
    print_action_stats("BC预测动作统计", pred_action)
            
    # 保存 BC 权重
    save_path = "checkpoints/bc_policy"
    student.save(save_path)
    print(f"✅ BC 预训练完成，模型已保存至 {save_path}.zip")

if __name__ == "__main__":
    train_bc()