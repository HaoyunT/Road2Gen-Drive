# MetaDrive 驾驶实验记录（IL+RL / GAIL）

这个仓库是我做 MetaDrive 端到端驾驶实验时整理的代码。我这里有两条**独立**实验线：

- `IL+RL`：先做行为克隆（BC），再用 PPO 微调。
- `GAIL`：单独做对抗模仿学习和泛化评估。

## 先说明（重要）

`IL+RL` 和 `GAIL` 在这个仓库里是**分开的**，我没有把两者串起来训练。也就是说：

- `IL+RL` 的模型不会自动喂给 `GAIL`；
- `GAIL` 也不依赖 `IL+RL` 的训练结果；
- 两边各自采集/处理各自的数据，各跑各的实验。

## 目录结构

```text
GAIL/
  1_collect_multiscene_data.py
  2_process_data.py
  3_train_gail.py
  4_evaluate_generalization.py

IL+RL/
  1_record_expert.py
  2_train_bc.py
  3_train_rl.py
  4_evaluate.py
  requirements.txt
```

## 环境准备（Windows）

我本地按 Python 3.10 配的，其他版本不保证完全一致。

```powershell
# 先装 IL+RL 依赖
cd .\IL+RL
python -m venv .venv
.\.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt

# GAIL 脚本常用到这些
pip install opencv-python tqdm gymnasium
```

> `metadrive-simulator` 对系统环境要求比较敏感，如果安装或渲染报错，优先检查显卡驱动和图形依赖。

## A. IL+RL 实验（独立）

### 流程

1. `1_record_expert.py`：用 IDM 自动采集专家数据。
2. `2_train_bc.py`：做 BC 预训练。
3. `3_train_rl.py`：基于 BC 权重做 PPO 微调。
4. `4_evaluate.py`：可视化评估。

### 运行

```powershell
cd .\IL+RL
python .\1_record_expert.py
python .\2_train_bc.py
python .\3_train_rl.py
python .\4_evaluate.py
```

### 主要产物

- `IL+RL/data/human_expert.pkl`
- `IL+RL/checkpoints/bc_policy.zip`
- `IL+RL/checkpoints/rl_final_model.zip`
- `IL+RL/checkpoints/best_model.zip`

## B. GAIL 实验（独立）

### 流程

1. `1_collect_multiscene_data.py`：采集多场景 expert transitions。
2. `2_process_data.py`：转成 `imitation` 需要的 `Trajectory`。
3. `3_train_gail.py`：训练 GAIL。
4. `4_evaluate_generalization.py`：在未见种子上做泛化评估。

### 运行

> 这组脚本默认参数里有 Linux 路径（`/root/autodl-tmp/...`），在 Windows 上我建议显式传本地路径。

```powershell
cd .\GAIL

# 1) 采集数据
python .\1_collect_multiscene_data.py `
  --output .\data\expert_train.pkl `
  --seed-start 0 --seed-end 799 `
  --target-frames 300000

# 2) 数据处理
python .\2_process_data.py `
  --input .\data\expert_train.pkl `
  --output .\data\gail_expert_trajs_train.pkl

# 3) GAIL 训练
python .\3_train_gail.py `
  --demo-path .\data\gail_expert_trajs_train.pkl `
  --save-dir .\checkpoints\gail_policy `
  --log-dir .\logs\tensorboard\gail

# 4) 泛化评估
python .\4_evaluate_generalization.py `
  --model-path .\checkpoints\gail_policy.zip `
  --start-seed 5000 --num-scenarios 100 --num-runs 5
```

## 实验结果（我目前的结论）

- 这两条线的结果我觉得都不错。
- `IL+RL` 这边准确性和稳定性都可以，成功率大概在 **80%** 左右。
- `GAIL` 这边虽然更难，但已经能体现出泛化能力，在未见场景上的成功率大概在 **40%** 左右。

## 算法流程详解（我的实现）

### IL+RL：先模仿、再强化（目标是稳定）

1. **专家数据采集（IDM）**  
  我在 `1_record_expert.py` 里用 IDMPolicy 自动驾驶采样，不需要手动开车；数据里保存观测和真实执行动作。

2. **行为克隆 BC 预训练**  
  我在 `2_train_bc.py` 里用 PPO 的策略网络当学生网络做监督学习，核心是让策略先学会“像专家一样开”。

3. **损失函数设计**  
  BC 训练时我不是只做一个 MSE，而是拆成：
  - 转向回归损失（smooth L1）
  - 油门/刹车回归损失（smooth L1）
  - 油门正负号分类约束（BCE）  
  这样能更好处理加速/制动切换。

4. **训练稳定化策略**  
  我用了验证集、早停、学习率退火、梯度裁剪，并保存验证集最优参数，减少过拟合和训练抖动。

5. **PPO 分阶段微调**  
  在 `3_train_rl.py` 中我加载 BC 权重后做 PPO 微调，并采用同图渐进难度（车流密度从低到高），让策略从“会开”变成“更稳、更抗干扰”。

6. **评估方式**  
  在 `4_evaluate.py` 里我默认用确定性动作评估策略上限表现，主要看成功率、碰撞和稳定性。

### GAIL：对抗模仿 + 泛化评估（目标是泛化）

1. **多场景数据构建**  
  在 `1_collect_multiscene_data.py` 里，我按大量 seed 采集专家 transitions，并做了数据清洗：
  - 过滤低速静止帧
  - 过滤转向突变帧
  - 可保留失败回合末尾恢复窗口（用于丰富困难样本）

2. **转换为轨迹格式**  
  `2_process_data.py` 会把 transitions 组织成 `imitation` 库需要的 `Trajectory`，并过滤过短轨迹。

3. **GAIL 训练主干**  
  `3_train_gail.py` 里是 PPO 生成器 + 判别器对抗训练。判别器负责区分“专家/策略”，生成器通过 PPO 优化伪奖励。

4. **混合奖励与奖励整形**  
  我用的是混合奖励：`r = α * r_gail + β * r_env`，同时叠加车道保持、出界惩罚、动作平滑、进度奖励等 shaping，提升可控性和训练稳定性。

5. **训练模式**  
  支持单阶段或双阶段训练；也支持 BC warm-up、VecNormalize、断点恢复与 checkpoint，主要是为了在长训练中更稳。

6. **泛化评估**  
  `4_evaluate_generalization.py` 在未见 seed 上做多次 run，统计成功率、碰撞率、出界率、路线完成度和转向 jerk，并可录制成功视频。

## 最后说明

我这里的 `IL+RL` 和 `GAIL` 是两条**独立实验线**：  
前者偏“已知场景稳定控制”，后者偏“未见场景泛化能力”。目前我把二者并行对比，而不是串成一个统一训练流水线。
