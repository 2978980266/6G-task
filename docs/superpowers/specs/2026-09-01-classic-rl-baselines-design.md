# 经典强化学习对比基线设计

**日期：** 2026-09-01  
**状态：** 已确认，待实施

## 目标

在不改变现有任务驱动 `TaskRelaySchedulingEnv` 语义的前提下，为开题报告补齐一套可复现实验的经典强化学习对比基线。实现应保持数据、几何、信道、任务、packet-level CAoI 和资源约束与算法代码解耦；原有 PPO 功能、命令和模型兼容路径必须继续可用。

## 对比方法

本次保留并支持以下七条算法线：

1. PPO（已有）
2. REINFORCE / Vanilla Policy Gradient
3. A2C
4. TRPO
5. DQN
6. Double DQN
7. Dueling DQN

### 动作接口分组

| 分组 | 算法 | 动作接口 | 可比性说明 |
| --- | --- | --- | --- |
| 多链路策略优化 | REINFORCE、A2C、PPO、TRPO | 现有固定长度 `MultiDiscrete` 多槽接口 | 共用相同观测编码、动作组合器、奖励与任务环境，可严格横向比较。 |
| 单候选价值学习 | DQN、Double DQN、Dueling DQN | 新增 `Discrete(top_k + 1)` 接口 | 每步选择一个候选链路或空动作；三种 DQN 方法严格可比。与多链路组对比时必须在图表和摘要中标明接口不同。 |

不加入 SAC、TD3、DDPG 等连续动作算法，避免为了适配离散调度问题而引入不自然的连续动作投影。

## 架构边界

### 环境

- 保留 `PpoSchedulingGymEnv` 和它目前的外部行为。
- 新增独立的单候选 Gym 适配器。该适配器只将一个离散候选索引转换为既有 `SchedulingAction`，并复用既有候选排序、资源验证、packet 更新和奖励计算。
- 环境、任务管理器、信道模型和规则 baseline 不感知训练算法名称。

### 算法层

- 建立算法注册表，集中定义算法名、动作接口、训练器和推理加载器，避免在脚本中散落分支。
- REINFORCE 独立实现为多离散动作的蒙特卡洛策略梯度训练器。
- DQN、Double DQN 与 Dueling DQN 共享 replay buffer、训练循环和保存格式；仅在 Bellman target 选择规则和网络头结构处变化。
- A2C、PPO 采用 Stable-Baselines3；TRPO 采用同生态的成熟可选实现。
- 所有模型都通过通用 metadata 记录算法、动作接口、观测形状、候选参数、场景/配置指纹、训练 seed 和训练步数。现有 PPO metadata 验证与旧模型兼容入口保持可用。

### 命令行与兼容性

- 新增统一训练入口 `scripts/train_rl.py`，用 `--algorithm` 切换七种算法。
- 保留 `scripts/train_ppo.py` 原有参数、默认输出与行为；它可复用共享训练代码，但不得要求用户改用新命令。
- 扩展 `scripts/run_experiment.py` 和 `scripts/trace_episode.py`，使它们能够加载新的模型策略；规则 baseline 命令保持不变。
- 实验摘要必须包含 `algorithm`、`action_interface`、训练 seed、评估 seed、模型版本和动作参数，避免混淆不同接口的结果。

## 可视化

新增 `scripts/plot_rl_comparison.py`，从统一 JSON 摘要读取结果，生成高分辨率 PNG 和 PDF。至少绘制：

1. 平均 CAoI（低为优）
2. 完成任务数（高为优）
3. 总任务奖励（高为优）
4. 每 step 完整包交付数（高为优）
5. 每 step V2V 完整包交付数（高为优）

图中按照两种动作接口分组，采用稳定的色盲友好配色；输出不依赖手工抄录指标。

## 验证策略

在实施过程中先为下列行为增加回归检查：

1. 单候选适配器的离散动作空间、空动作和合法性。
2. DQN、Double DQN 的目标值计算差异。
3. Dueling 网络的价值/优势聚合性质。
4. 算法注册、模型 metadata 校验和模型重新加载。
5. 每种算法的小规模训练、保存、加载和完整回合评估。
6. 绘图脚本对合法汇总数据的读取和图片输出。

完成后，除新增检查外，至少运行现有环境与 PPO 回归检查、编译检查、Ruff 和 Mypy。不会更改完整包、V2V 真实持包、任务奖励归因、资源约束或现有 PPO 的回归语义。

## 非目标

- 不追求本阶段算法的最优超参数或最终论文结论。
- 不引入世界模型、部分包传输、连续动作投影或对现有物理层的改造。
- 不删除、重命名或替代现有 PPO、规则 baseline、训练产物或官方 Mod 源码。
