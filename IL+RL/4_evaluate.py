import os
import numpy as np
from metadrive import MetaDriveEnv
from stable_baselines3 import PPO

def evaluate():
    # 1. 配置环境
    # 你当前 RL 是在固定同图 (seed=42, num_scenarios=1) 训练的。
    # 因此默认先做同图评估，再按需切到泛化评估。
    eval_mode = "same_map"  # 可选: "same_map" | "generalization"

    if eval_mode == "same_map":
        config = {
            "use_render": True,
            "traffic_density": 0.12,
            "map": "SCO",
            "manual_control": False,
            "num_scenarios": 1,
            "start_seed": 42,
            "window_size": (1200, 900),
        }
    else:
        config = {
            "use_render": True,
            "traffic_density": 0.12,
            "map": "SCO",
            "manual_control": False,
            "num_scenarios": 30,
            "start_seed": 5000,
            "window_size": (1200, 900),
        }

    # 评估阶段动作平滑（EMA）
    # 默认关闭，保持与训练策略一致
    use_action_smoothing = False
    action_smoothing_alpha = 0.35
    
    print(f"正在初始化演示环境... (mode={eval_mode})")
    env = MetaDriveEnv(config)
    
    # 2. 加载模型
    # 简洁 IL+RL 流程默认评估 RL 最终模型
    # 可选: "final" | "best" | "auto" | "bc"
    model_select = "bc"

    # auto: best -> final -> bc
    # final: final -> best -> bc
    # best: best -> final -> bc
    # bc: bc -> final -> best
    best_model_path = "checkpoints/best_model"
    rl_model_path = "checkpoints/rl_final_model"
    bc_model_path = "checkpoints/bc_policy"
    
    model = None
    
    if model_select == "best":
        candidate_paths = [best_model_path, rl_model_path, bc_model_path]
    elif model_select == "final":
        candidate_paths = [rl_model_path, best_model_path, bc_model_path]
    elif model_select == "bc":
        candidate_paths = [bc_model_path, rl_model_path, best_model_path]
    else:
        candidate_paths = [best_model_path, rl_model_path, bc_model_path]

    for candidate in candidate_paths:
        if os.path.exists(candidate + ".zip"):
            tag = "RL 最佳模型" if candidate == best_model_path else (
                "RL 微调最终模型" if candidate == rl_model_path else "BC 模型"
            )
            print(f"✅ 加载 {tag}: {candidate}")
            model = PPO.load(candidate)
            break

    if model is None:
        print("❌ 错误：找不到任何模型文件！请先运行训练脚本。")
        env.close()
        return
    
    obs, info = env.reset()
    
    print("="*60)
    print("🤖 自动驾驶演示开始！")
    print("按 [ESC] 退出程序")
    print("="*60)

    episode_count = 0
    prev_action = None
    try:
        # 跑 5000 步
        for i in range(5000):
            # deterministic=True 很关键
            # 训练时我们需要随机性(std)来探索，演示时我们要最稳的策略(均值)
            action, _ = model.predict(obs, deterministic=True)

            if use_action_smoothing:
                action = np.asarray(action, dtype=np.float32)
                if prev_action is None:
                    smooth_action = action
                else:
                    smooth_action = (
                        action_smoothing_alpha * action
                        + (1.0 - action_smoothing_alpha) * prev_action
                    )
                prev_action = smooth_action
                action_to_env = smooth_action
            else:
                action_to_env = action

            obs, reward, done, truncated, info = env.step(action_to_env)
            
            # 在画面左上角显示状态
            env.render(text={
                "Mode": "AI Auto-Pilot",
                "Step": i,
                "Speed": f"{info.get('velocity', 0):.1f} km/h"
            })
            
            # 如果撞车或跑完，重置环境
            if done or truncated:
                episode_count += 1
                print(f"Episode {episode_count} 结束 (Global Step {i})，重置环境...")
                obs, info = env.reset()
                prev_action = None
                
    except KeyboardInterrupt:
        print("演示已停止。")
    finally:
        env.close()

if __name__ == "__main__":
    evaluate()