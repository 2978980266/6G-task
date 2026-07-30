# DualMind

这是一个面向论文 `Dual-Mind World Models: A General Framework for Learning in Dynamic Wireless Networks` 的第一阶段复现实验骨架。

当前版本的目标不是一比一复刻原论文，而是先把最小可行实验跑通：

1. 重放车辆轨迹。
2. 用建筑几何近似遮挡。
3. 用数学公式模拟链路质量和速率。
4. 用 AoI 作为第一阶段优化指标。
5. 先比较几个启发式 baseline，为后续 PPO / world model 做准备。

## 第一阶段范围

当前代码已经实现：

- 车辆轨迹 CSV 读取
- 建筑快照 CSV 读取
- 基于建筑 footprint 的 LOS / NLOS 近似判断
- 基于路径损耗和 Shannon 公式的速率估计
- 单 RSU、单链路调度环境
- 三个 baseline：
  - `random_reachable`
  - `round_robin_reachable`
  - `max_aoi_reachable`
- 一个可直接接入 RL 的 `gymnasium.Env` 封装

当前刻意没有实现：

- 完整 RSSM / Dreamer world model
- LINN 或复杂逻辑推理网络
- 多 RSU / 多跳联合优化

## 目录结构

```text
DualMind/
|- data/
|  |- config/
|  |  |- example_config.json
|  |- README.md
|- scripts/
|  |- run_baselines.py
|  |- train_ppo.py
|- src/
|  |- dualmind_env/
|     |- baselines/
|     |- data/
|     |- env/
|     |- geometry/
|     |- wireless/
|     |- config.py
|     |- types.py
|- pyproject.toml
|- README.md
```

## 输入数据

### 车辆轨迹来源

当前支持两种输入方式：

1. 单个总 CSV 文件
2. 一个目录下按 `vehicleId` 分开的多个 CSV 文件

默认按照你当前的导出格式读取，至少需要这些列：

- `recordTimeUtc`
- `simulationFrame`
- `gameTime`
- `vehicleId`
- `x`
- `y`
- `z`
- `angleX`
- `angleY`
- `speed`
- `prefabName`

### 建筑快照来源

当前也支持两种输入方式：

1. 单个建筑总 CSV 文件
2. 一个目录下按 `buildingId` 分开的多个 CSV 文件

默认按照你当前的导出格式读取，至少需要这些列：

- `recordTimeUtc`
- `simulationFrame`
- `gameTime`
- `buildingId`
- `x`
- `y`
- `z`
- `angleY`
- `widthCells`
- `lengthCells`
- `sizeX`
- `sizeY`
- `sizeZ`
- `centerOffsetX`
- `centerOffsetY`
- `centerOffsetZ`
- `minY`
- `maxY`
- `prefabName`

## 建模假设

为了尽快把第一阶段跑通，当前做了这些简化：

1. 只考虑一个 RSU。
2. 每个时隙最多调度一辆车上传。
3. 建筑遮挡只做二维 footprint 近似，不做精细射线追踪。
4. 速率估计采用路径损耗 + Shannon 公式。
5. 当前奖励为 `-mean(AoI)`。

## 快速开始

先修改配置文件：

- `data/config/example_config.json`

重点填这几项：

- `trajectory_csv`
  这里既可以填一个总 CSV 文件路径，也可以填一个车辆 CSV 目录路径。
- `building_csv`
  这里既可以填一个总 CSV 文件路径，也可以填一个建筑 CSV 目录路径。
- `rsu.x`
- `rsu.y`
- `rsu.z`

然后运行：

```powershell
python scripts/run_baselines.py --config data/config/example_config.json
```

如果你想用 Codex 自带 Python，也可以把 `python` 换成对应绝对路径。

## PPO baseline

当前已经补上了一个标准 `gymnasium` 包装环境：

- `dualmind_env.env.AoiSchedulingGymEnv`

它会把当前调度问题映射成：

1. 固定长度向量观测
2. `Discrete(N)` 动作空间
3. 每一步先按当前候选车辆状态动态排序，再由动作选择“第几个槽位”
4. 排序会优先考虑 `reachable`、`AoI`、距离等因素，避免 PPO 只记固定 `vehicleId`
5. 默认不给 `PPO` 额外的空动作，减少无效探索

安装依赖后，可以直接训练一个最小 PPO baseline：

```powershell
python scripts/train_ppo.py --config data/config/example_config.json --timesteps 300000
```

如果你希望在终端看到更详细的训练过程，可以额外传：

```powershell
python scripts/train_ppo.py --config data/config/example_config.json --timesteps 20000 --verbose 1
```

如果你希望额外生成 tensorboard 日志，可以再传：

```powershell
python scripts/train_ppo.py --config data/config/example_config.json --timesteps 20000 --tensorboard-log outputs/ppo/tensorboard
```

脚本默认会：

1. 训练 PPO 调度器
2. 保存模型到 `outputs/ppo/ppo_scheduler.zip`
3. 把评估结果写到 `outputs/ppo/ppo_summary.json`
4. 默认不生成 tensorboard 日志目录

当前环境默认还会加入轻量 reward shaping：

1. 成功调度会获得 `success_bonus`
2. 成功调度高 AoI 车辆会额外获得 `aoi_priority_bonus_scale * selected_vehicle_aoi_before`
3. 选择当前无效动作会受到 `invalid_action_penalty`

这些量都可以在 `environment` 配置中手动覆盖；默认值分别是：

1. `success_bonus = 1.0`
2. `aoi_priority_bonus_scale = 0.25`
3. `invalid_action_penalty = 1.0`

训练结束后，脚本默认还会额外写一份逐步评估日志：

- `outputs/ppo/ppo_eval_trace.json`

它可以帮助你直接检查 PPO 每一步到底选了谁，以及该选择是否接近当前 reachable 车辆中的最高 AoI。

## 输出说明

脚本默认会：

1. 在终端打印 baseline 结果表。
2. 把结果写到 `outputs/baseline_summary.json`。

输出指标包括：

- `total_reward`
- `mean_step_aoi`
- `success_rate`
- `mean_reachable_rate_mbps`
- `steps`

## 下一阶段建议

第一阶段跑通后，最自然的后续工作是：

1. 先把 PPO baseline 跑稳，并和 `max_aoi_reachable` 做稳定对比。
2. 再训练一个 `s_t, a_t -> s_{t+1}, r_t` 的 learned dynamics model。
3. 再把规则约束加上，形成简化版 dual-mind 结构。
