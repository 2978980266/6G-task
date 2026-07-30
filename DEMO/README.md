# Skylines Dual-Mind DEMO

这个工程使用 `Cities: Skylines 1` 导出的车辆轨迹和建筑快照，搭建低耦合的任务驱动 `V2I + V2V + CAoI` 环境。当前主线是：25 种独立信息包、8 个任务模板、完整包传输、RSU/V2V 独立资源池、三维数学信道和 freshness-aware 任务奖励。旧的部分包 CAoI 环境仍保留为兼容实现；当前 PPO 已切换到任务驱动环境。

## 1. 当前实现了什么

当前主线设定如下：

1. 单 `RSU`。
2. 同时支持 `V2I` 和 `V2V`。
3. 信息状态使用 packet-level `CAoI`：
   - 共 `packet_count = 25` 种信息包，编号为 `1..25`
   - 每类包大小为 `packet_size_bits = 2400000.0`，即 `0.30 MB`
   - 每辆车维护 packet 级时间戳状态，车辆 `CAoI` 是这些 packet age 的均值
   - 每次只允许交付完整包，不保存部分包进度
4. `V2V` 采用中继转发语义：
   - `RSU` 先把新 packet 发送给部分车辆
   - 已持有真实更新 packet 的车辆可向尚未持包或持有旧版本的车辆转发
   - 初始 `CAoI = 0`，但初始状态不代表车辆真实持有可中继 packet
5. 每个环境 step 对应车辆 CSV 中的一条采样记录；episode 使用固定轨迹 step，不额外插值或延长回合。
   - `delta_seconds` 使用当前记录到下一条记录的 `sessionSeconds` 前向差值
   - 最后一个 step 没有下一条记录，因此使用 `fallback_step_seconds`
   - 这样 action 后的完成时刻与下一观测的 `elapsed_seconds` 连续一致
6. 环境动作已显式拆成双动作集：
   - `SchedulingAction.v2i_links`
   - `SchedulingAction.v2v_links`
7. 环境会额外返回 `ControlBroadcastAction`：
   - 表示 `RSU` 的调度控制命令已通过理想控制面可靠广播
   - 该控制面不参与 `CAoI`、速率和链路成败统计
   - 车辆位置、速度、任务请求和所需包信息直接包含在全局观测中，也视为通过理想控制面上报
8. 默认提供 8 个任务模板，每个任务需要 7 至 8 种包：
   - 同一车辆不会重复申请以前的任务
   - 每车每回合最多完成 3 个任务
   - 当前任务未完成时不能申请下一任务
   - 若任务在 step `t` 完成，则 `t+1` 不申请，最早在 `t+2` 申请
   - 固定 seed 时任务顺序可复现；自动驾驶功能后续若重新开启，其身份划分也使用独立随机流
   - 车辆取得当前任务要求的全部真实 packet 后，任务在该 step 末完成
   - 已缓存的真实 packet 可以服务后续任务，其奖励仍由完成时 AoI 决定
9. 任务完成奖励由所需包的 AoI 决定：
   - `freshness = exp(-AoI / freshness_decay_seconds)`
   - `task_reward = reward_base * mean(freshness)`
   - `task_reward_gain` 是候选动作相对空动作带来的奖励增量
   - 当前不加入资源占用、能耗或任务超时惩罚

## 2. 当前通信规则

当前环境的核心规则：

1. `RSU` 是唯一决策中心，车辆只执行调度命令。
2. 每个 step 可同时安排多条 `V2I`，共享 RSU 的 `40 MHz`、8 个资源块和 `100 Mbps` 总速率预算。
3. 多条 `V2V` 可以并发，共享独立的 `40 MHz`、8 个 sidelink 资源块和 `60 Mbps` 总速率预算。
4. 任意车辆在同一 step 内最多参与一条数据链路，包括作为 V2I 接收端、V2V 发送端或 V2V 接收端。
5. 链路实际带宽由资源块数量决定，信道使用该带宽计算噪声、SNR 和物理速率，再施加单车及总速率限制。
6. 控制面始终可靠；只有 V2I/V2V 数据链路检查可达性，`reachable = false` 的链路不会进入候选。
7. `V2V` 候选要求发送端真实持有 packet，且接收端尚未真实持有该 packet，或发送端时间戳更新。
8. 当前观测中会分别给出：
   - `candidate_v2i_links`
   - `candidate_v2v_links`
9. 动作显式指定链路、资源块数量和完整包 ID；不足一个完整包的链路不会形成可执行候选。
10. 重复车辆、未知候选、V2I/V2V 集合归属错误、资源块超限或总速率超限会使整个动作 invalid；非法动作不会更新 packet，也不会统计实际资源占用。
11. 同一步不允许车辆先接收 V2I 再立即通过 V2V 转发。

这套设计对应论文里的中心化调度语义：`RSU` 知道全局状态并统一选链路，`V2V` 则要求发送车真实持包，并只补齐接收端缺失的真实版本或更新旧版本。

## 3. 当前 CAoI 设计

当前 `CAoI` 的实现不是把一个标量 `AoI` 强行改名，而是显式维护 packet-level freshness state。

每辆车当前维护：

1. `packet_timestamps`
   - 每个 packet 最近一次被该车持有时，对应的信息生成时间戳
2. `packet_updated_flags`
   - 标记该 packet 是否来自真实更新
3. `caoi_seconds`
   - 基于当前时刻对全部 packet age 求均值得到

动作后的更新方式：

1. `V2I`
   - `RSU` 视为拥有当前时刻的新鲜 packet
   - 根据 `floor(rate_mbps * 1_000_000 * delta_seconds / packet_size_bits)` 计算完整可交付包数
   - 动作只能选择不超过该容量的完整包编号
   - 不足一个完整包的剩余 bit 不跨 step 累积
2. `V2V`
   - 发送车只能转发自己真实持有的 packet
   - 接收车尚未真实持有该 packet 时允许接收；已经持有时只接受时间戳更新的版本
3. `StepResult`
   - 统计的是 action 之后的 `CAoI`
   - 因此更接近论文里 `A_{t+1}` 的口径

当前默认关闭初始 CAoI 随机化。所有车辆的全部 packet 在轨迹首帧从 `AoI = 0` 开始，之后随环境时间统一增长。
初始 `packet_updated_flags = False`，因此 `AoI = 0` 只表示统一的仿真起点，不表示车辆真实持有可中继内容。
初始 packet 时间戳以轨迹首帧的真实 `elapsed_seconds` 为基准，不会引入首帧时间偏移。

## 4. 当前信道模型

当前使用的是轻量、可替换的数学公式信道，不是论文原版的 `ray tracing`。

已实现：

1. 基于建筑 footprint 和建筑高度的三维 `LoS/NLoS` 判断。
2. 使用三维欧氏距离的对数距离路径损耗模型。
3. `NLoS` 额外损耗。
4. 轻量车辆遮挡近似。
5. 噪声功率随链路实际分配带宽计算。
6. 带实现余量、效率折减和频谱效率上限的 Shannon 速率近似。
7. 原始物理速率之后先施加单车链路上限；调度动作还必须满足 RSU 和 V2V 各自的总速率预算。

当前默认参数：

1. `rsu_tx_power_dbm = 26.0`
2. `vehicle_tx_power_dbm = 23.0`
3. `rsu_total_bandwidth_hz = 40000000.0`
4. `v2v_total_bandwidth_hz = 40000000.0`
5. `rsu_resource_block_hz = 5000000.0`
6. `v2v_resource_block_hz = 5000000.0`
7. `max_resource_blocks_per_link = 2`
8. `rsu_total_rate_limit_mbps = 100.0`
9. `v2v_total_rate_limit_mbps = 60.0`
10. `vehicle_link_rate_limit_mbps = 30.0`
11. `noise_density_dbm_per_hz = -174.0`
12. `receiver_noise_figure_db = 10.0`
13. `path_loss_offset_db = 61.4`
14. `path_loss_exponent = 2.5`
15. `nlos_extra_loss_db = 35.0`
16. `implementation_margin_db = 6.0`
17. `rate_efficiency_factor = 0.55`
18. `max_spectral_efficiency_bps_hz = 4.0`
19. `min_rate_mbps = 1.0`
20. `vehicle_blockage_loss_db = 22.0`

标准 `0.3 s` step 下，`100 Mbps` RSU 总速率预算理论上可承载 `12` 个完整包；默认实验的单 step V2I 峰值也是 `12` 个。

## 5. 当前 baseline

当前任务主线提供三个 baseline：

1. `random-task-aware`
   - 随机选择资源合法的链路集合，并优先处理能推进或完成任务的候选
2. `max-task-reward-gain`
   - 按任务完成奖励、任务进度和 freshness 收益贪心调度
3. `two-stage-task-relay-greedy`
   - 优先选择兼顾当前任务和后续中继需求的 V2I，再补充 V2V

三个任务 baseline 都会遵守配置中的 `max_active_links`、`allow_multi_v2i_from_rsu`、资源块上限和总速率预算。

此外，仓库里还提供了一个任务驱动的 `PPO` baseline：

1. 训练入口：
   - 快速流程测试：
     `python DEMO/scripts/train_ppo.py --timesteps 512 --n-steps 128 --batch-size 64 --eval-episodes 1 --top-k 16 --action-slots 8 --progress-interval 100 --checkpoint-interval 256 --output-dir DEMO/outputs/ppo_quick_test`
   - 较长训练：
     `python DEMO/scripts/train_ppo.py --timesteps 100000 --eval-episodes 3 --top-k 16 --action-slots 8 --progress-interval 500 --checkpoint-interval 5000 --verbose 1 --output-dir DEMO/outputs/ppo_task`
2. 评估入口：
   - `python DEMO/scripts/run_experiment.py --policy ppo --model-path DEMO/outputs/ppo_task/ppo_scheduler --top-k 16 --action-slots 8`
   - 没有 metadata sidecar 的旧模型必须显式增加 `--allow-legacy-model`，否则会被拒绝。
3. 当前定位：
   - 使用当前 `TaskRelaySchedulingEnv`
   - 观察编码包含任务进度、任务完成奖励增量、packet 选择、资源块、链路状态和中继扩散收益
   - 动作空间使用固定长度 `MultiDiscrete`；每个动作槽选择一个带完整 packet ID 和资源块分配的候选链路
   - `top_k` 是全体 V2I/V2V 候选的扁平排序截断，不是两类链路的独立配额；`top_k=16` 可能让 V2V 候选很少进入 PPO 可选集合，`top_k=32` 需要重新训练模型并重新生成 metadata
   - PPO 奖励以成功链路的 `task_reward_gain` 为主，并加入任务 packet 进度和 CAoI 收益作为 shaping；环境 `StepResult.task_reward` 仅作为整步统计
   - 与任务 baseline 使用相同任务环境和统计口径，但训练步数、模型收敛程度仍需在比较时明确报告
   - PPO 相关依赖按可选功能延迟加载；只运行任务 baseline 时不要求导入 NumPy、Gymnasium、Stable-Baselines3 或 PyTorch
   - `outputs/` 是被 Git 忽略的生成目录；新模型会同时生成 `ppo_scheduler.zip` 和 `ppo_scheduler.metadata.json`

### PPO 训练进度与中断行为

1. `--progress-interval` 默认是 `5000` timestep。
2. 每次进度输出包含：
   - 当前 timestep 与百分比
   - 已用时间
   - `steps/s`
   - 预计剩余时间
3. `--verbose 1` 会额外显示 Stable-Baselines3 的 rollout 和训练指标。
4. `--checkpoint-interval` 默认是 `50000` timestep，设为 `0` 可关闭。
   - checkpoint 文件名为 `ppo_scheduler_checkpoint_<实际步数>.zip`
   - 每个 checkpoint 同时保存同名 metadata sidecar
   - 当前支持周期保存，但还没有自动断点续训命令；中断后只能手动选择最近 checkpoint 作为后续续训起点
5. 摘要同时记录：
   - `requested_training_timesteps`：命令行请求值
   - `training_timesteps`：Stable-Baselines3 实际采样值，可能因 rollout 对齐略大于请求值
6. 当配置中的 `max_steps` 小于轨迹长度时，Gymnasium 将回合标记为 `truncated=True`，并把截断点后的真实下一帧作为 PPO bootstrap 观测；只有自然到达轨迹末尾才是 `terminated=True`。
7. 2026-07-20 在当前机器上的一次参考运行：
   - `5000/100000` timestep 用时约 `6m55s`
   - 速度约 `12.1 steps/s`
   - 预计 `100000` timestep 约需两小时以上，实际速度受 CPU、PPO 参数和候选计算量影响

### PPO 当前已知限制

1. PPO 的任务奖励项使用本动作成功链路的 `task_reward_gain` 之和；环境的 `StepResult.task_reward` 继续保留为整步真实任务完成统计。
2. 动作前已经满足的任务仍会出现在环境统计中，但不会进入 PPO reward；wrapper 的 `info` 同时提供 `task_reward` 和 `task_reward_gain` 以便审计。
3. PPO 训练 episode 的 seed 从训练 seed 开始按 episode 序号递增，同一次运行可复现但不会反复使用同一个任务顺序；自动评估使用训练 seed 加 `1000000` 的独立 seed 区间。
4. 当前普通 Stable-Baselines3 PPO 不会读取 `action_masks()`：
   - 填充候选、重复候选和资源冲突候选由动作组合器跳过
   - 已验证不会生成非法环境动作，但会降低有效动作采样效率
5. 动作槽按槽位顺序依次组合候选；当多个槽选择冲突候选时，前面的槽优先。该顺序是当前动作语义的一部分。
6. 新训练模型必须带有 metadata sidecar；它会校验场景、轨迹/建筑快照、编码版本、`top_k`、`action_slots` 和观测维度。旧模型只能通过 `--allow-legacy-model` 显式加载，且不应直接用于正式对比。
7. 当前推理入口只接受任务版 `MultiDiscrete` 模型；旧 `Discrete` PPO 模型没有完整兼容。
8. `invalid_action_count=0` 只说明组合器提交给环境的最终动作合法；重复候选、车辆冲突、RB/速率超限的 slot 可能已在组合阶段被跳过，因此该指标不能代表 PPO 充分利用了 8 个 action slot。
9. 当前 PPO 的任务覆盖和资源利用率仍低于规则 baseline。近期 `top_k=32`、`100000` timesteps、训练 seed=7 的三回合评估平均为：平均 CAoI `2.952`、完成任务 `24.67`、有效链路 `3.39/step`、完整包交付 `8.61/step`、V2V 链路 `0.96/step`。这些数字用于定位稀疏调度问题，不是最终论文结果。

兼容说明：

旧命令 `random-set`、`max-caoi-gain`、`max-aoi-gain` 和 `two-stage-relay-greedy` 仍可用，会映射到对应任务策略。旧的 `RelaySchedulingEnv` 仍保留，但 PPO 训练和评估入口默认不再使用它。

其中 `two-stage-task-relay-greedy` 的 `V2I` 评分除了考虑接收车自己的即时 `CAoI` 收益，还会考虑后续潜在的中继扩散收益，具体体现在候选链路上的：

1. `relay_peer_count`
2. `relay_peer_total_gain_seconds`
3. `relay_peer_total_priority_gain_seconds`

这些值只统计当前几何和信道下可达、且至少能完整转发一个任务相关 packet 的潜在接收车；后两个字段分别表示预计 CAoI 收益及其可选优先级加权值。当前自动驾驶场景关闭，因此默认权重均为 `1.0`。

## 6. 可选自动驾驶优先场景

当前使用单一配置文件：

- `data/config/train.yaml`

该配置文件中的顶层 `seed` 是全工程默认随机种子。`check_links.py`、
`run_experiment.py`、环境装配入口和 PPO 训练入口在未显式传入 `--seed`
时都以这个值作为基准；命令行 `--seed` 只用于临时覆盖。任务版 PPO wrapper
会在自动 reset 时使用 `base_seed + episode_index`，而公共环境入口的显式 seed
仍保持单回合可复现。

自动驾驶优先功能的代码接口仍然保留，但当前默认场景已经关闭，不划分自动驾驶和人驾车辆，也不对任何车辆增加优先权重。

后续重新开启时，这个场景不会改动车辆轨迹、信道模型或 `V2V` relay 资格，只会在环境层增加身份标签，并让调度策略优先更新自动驾驶车辆的信息。

当前规则如下：

1. 车辆身份在每个 episode 的 `reset()` 时按 `seed` 和 `penetration_rate` 随机划分。
2. 同一 `seed` 下自动驾驶/人驾划分可复现。
3. 自动驾驶优先是软优先，不是硬约束：
   - 真实 `CAoI` 更新量仍按 packet-level 状态计算
   - 只是调度排序时会对自动驾驶接收端使用更高的优先级分数
4. 当前默认中性参数：
   - `enabled = false`
   - `penetration_rate = 0.0`
   - `priority_weight = 1.0`
5. 是否开启由同一配置文件里的 `autonomy.enabled` 控制。
6. 后续需要该场景时，再同时配置开启开关、渗透率和优先权重。

策略层的含义是：

1. `max-task-reward-gain` 先看任务完成奖励增益和任务进度，再看加权后的 CAoI 与中继扩散收益。
2. `two-stage-task-relay-greedy` 同时考虑加权后的即时收益和加权后的可达中继扩散潜力。
3. `random-task-aware` 保留随机性，并优先处理能推进或完成当前任务的候选。
4. 旧命令名会映射到这些任务策略，不再代表旧策略类的原始排序逻辑。

## 7. 代码结构

主要目录如下：

1. 配置：
   - `src/skylines_demo/config.py`
2. 共享类型：
   - `src/skylines_demo/types.py`
3. 数据读取：
   - `src/skylines_demo/data/`
4. 几何与遮挡：
   - `src/skylines_demo/geometry/`
5. 信道模型：
   - `src/skylines_demo/wireless/`
6. 环境：
   - `src/skylines_demo/env/task_relay_env.py`，当前任务主线
   - `src/skylines_demo/env/relay_env.py`，旧 CAoI/PPO 兼容环境
7. 策略：
   - `src/skylines_demo/policies/`
8. 任务管理：
   - `src/skylines_demo/tasks/`
9. 装配入口：
   - `src/skylines_demo/bootstrap.py`
10. 脚本：
   - `scripts/inspect_scene.py`
   - `scripts/check_links.py`
   - `scripts/trace_episode.py`
   - `scripts/check_env_rules.py`
   - `scripts/check_task_rules.py`
   - `scripts/check_ppo_wrapper.py`
   - `scripts/train_ppo.py`
   - `scripts/run_experiment.py`

`src/*.egg-info` 属于 `pip/setuptools` 在安装或构建时生成的包元数据，不是业务源码，不需要手工维护；删除后重新执行可编辑安装时可能再次生成。

## 8. 默认配置

默认配置文件：

- `data/config/train.yaml`

当前默认场景使用：

1. 全局默认随机种子：`seed = 7`
   - 所有公共脚本与环境装配入口未显式传入 seed 时使用该值
   - 命令行 `--seed` 仅临时覆盖本次运行
2. 车辆轨迹：`D:\vscode\6G\Cities_Skylines_data\VehicleTrackLogs\20260623_151408`
3. 建筑快照：`D:\vscode\6G\Cities_Skylines_data\BuildingTrackLogs\20260623_151408`
4. `RSU` 位置：
   - `x = 1350.894`
   - `y = 180.0`
   - `z = 2386.964`
5. `CAoI` 参数：
   - `initial_caoi_seconds = 0.0`
   - `initial_caoi_min_seconds = null`
   - `initial_caoi_max_seconds = null`
   - `packet_count = 25`
   - `packet_size_bits = 2400000.0`
   - `age_tolerance_seconds = 8.0`
6. 任务参数：
   - `max_completed_tasks_per_vehicle = 3`
   - `cooldown_steps = 1`
   - `reward_base = 10.0`
   - `freshness_decay_seconds = 4.0`
7. 自动驾驶参数：
   - `enabled = false`
   - `penetration_rate = 0.0`
   - `priority_weight = 1.0`

## 9. 常用脚本

### 查看场景

```powershell
python DEMO/scripts/inspect_scene.py
```

### 检查候选链路

```powershell
python DEMO/scripts/check_links.py --steps 5 --candidates 4
```

默认输出会显示实际使用的 seed、每步的候选/选中链路数量，以及候选的有效速率、资源块、完整包、任务进度和奖励增量。需要距离、LoS、原始速率、车辆身份和完整交付包数等诊断信息时，使用：

```powershell
python DEMO/scripts/check_links.py --steps 5 --candidates 4 --verbose
```

### 逐 step 追踪完整信息传递流程

`check_links.py` 主要展示动作执行前的候选链路与计划调度。需要核对 packet 是否在动作后真实更新到接收车、V2V 是否正确中继、资源是否实际占用以及任务何时完成时，使用：

```powershell
python DEMO/scripts/trace_episode.py --policy max-task-reward-gain --show-all-vehicles --jsonl DEMO/outputs/max_task_trace.jsonl | Tee-Object DEMO/outputs/max_task_trace.log
```

该脚本默认追踪整个 episode。终端和 `.log` 会按 step 输出：

1. 动作前的车辆任务、真实持包、缺失包和 CAoI
2. 策略实际选择的 V2I/V2V 链路
3. 每条选中链路的 packet、资源块、带宽、距离、建筑 LoS、车辆遮挡、路径损耗、SNR、原始/有效速率与完整包容量
4. 实际成功/失败链路、资源块占用、完整交付包数、动作后 CAoI、任务完成和奖励
5. 下一观测中相关车辆的持包与任务变化

默认只打印被选中链路或任务完成涉及的车辆；加入 `--show-all-vehicles` 会打印全部在场车辆。常用参数：

```powershell
# 只回放前 3 步
python DEMO/scripts/trace_episode.py --policy max-task-reward-gain --steps 3 --show-all-vehicles

# 临时覆盖随机种子
python DEMO/scripts/trace_episode.py --policy two-stage-task-relay-greedy --seed 11
```

`--jsonl` 会生成每 step 一行的结构化记录，包含完整动作前车辆状态、选中/成功/失败链路、控制广播、资源占用和动作后统计；脚本会自动创建输出目录。PPO 追踪仍需要提供与训练模型一致的 `--model-path`、`--top-k` 和 `--action-slots`。

当前 `max-task-reward-gain` 会优先选择任务奖励增量更高的候选。候选按可交付完整包容量选择 packet 优先级列表的前缀，因此车辆只缺一个任务包时，也可能同时接收已持有但更旧的任务相关包以刷新其 AoI。这是有效更新而非同版本重复传输；但当前没有“只传缺失包”的子集候选，也未按边际任务收益/RB 做全局优化。

### 检查环境规则

```powershell
python DEMO/scripts/check_env_rules.py
```

当前覆盖的规则包括：

1. 默认任务模板数量、包数量、包覆盖和跨任务重复符合设计
2. 场景配置中的默认 seed 可复现地控制任务顺序；环境装配未显式传入 seed 时会读取该配置
3. 不足一个完整包时不更新任何包状态
4. 不可达的数据链路不会进入候选集合
5. 链路实际分配带宽等于资源块数量乘单块带宽
6. 整个回合内多条 V2I/V2V 都满足资源块、总速率和车辆半双工约束
7. 控制面每个 step 都可靠可达，且空数据动作不占数据资源
8. 默认场景至少存在单 step 交付 10 个 V2I 完整包的可行调度
9. V2V 只能转发发送车真实持有的完整 packet；接收端已有真实版本时必须保证时间戳更新
10. 每车最多完成 3 个任务，且任务 ID 不重复
11. 完成任务后必须经过一个完整冷却 step
12. 任务奖励公式及候选的增量奖励计算正确
13. 路径损耗使用三维距离
14. 建筑 LoS/NLoS 判断会考虑建筑高度
15. 非法动作不更新数据，也不统计实际资源占用
16. `check_links.py` 在同一 seed 下输出一致
17. V2I/V2V 动作放错集合时会被判为非法
18. 任务关闭时不会继续报告“可申请新任务”
19. baseline 会遵守配置中的最大链路数和单/多 V2I 开关
20. 初始 CAoI 与首帧时间基准一致，旧环境固定 seed 的重复 reset 可复现
21. 接收端未真实持包时，时间戳相等的真实 packet 可以覆盖初始占位状态
22. 相邻 step 满足“当前 `elapsed_seconds + delta_seconds = 下一 step elapsed_seconds`”
23. 中继扩散统计对未真实持包的接收端采用与实际 V2V 相同的等时间戳覆盖规则

### 跑实验

```powershell
python DEMO/scripts/run_experiment.py --policy max-task-reward-gain
```

可选策略：

1. `random-task-aware`
2. `max-task-reward-gain`
3. `two-stage-task-relay-greedy`
4. `random-set`、`max-caoi-gain`、`max-aoi-gain`、`two-stage-relay-greedy`，兼容旧命令名
5. `ppo`，使用当前任务环境，需要提供与模型一致的 `--model-path`、`--top-k` 和可选的 `--action-slots`

默认汇总按 `caoi`、`tasks`、`delivery_per_step` 和 `execution` 分组展示，并明确打印实际使用的 `seed`。完整的扁平统计字段仍保留，可用于排查或写入 JSON：

```powershell
python DEMO/scripts/run_experiment.py --policy max-task-reward-gain --full-output
```

`--summary-json` 始终写入完整统计。自动驾驶场景关闭时，完整统计中的 `autonomous_vehicle_count = 0`、`human_vehicle_count = 16`，加权 CAoI 与普通平均 CAoI 相同。

## 10. 当前验证状态

当前已确认以下命令可以跑通：

1. `python DEMO/scripts/check_env_rules.py`
2. `python DEMO/scripts/check_links.py --steps 3 --candidates 5`
3. `python DEMO/scripts/run_experiment.py --policy max-task-reward-gain`
4. `python DEMO/scripts/run_experiment.py --policy two-stage-task-relay-greedy`
5. `python -m compileall DEMO/src DEMO/scripts`
6. `python DEMO/scripts/check_ppo_wrapper.py`
7. `python -m ruff check DEMO/src DEMO/scripts`
8. `python -m mypy DEMO/src`
9. `python DEMO/scripts/trace_episode.py --policy max-task-reward-gain --steps 2 --show-all-vehicles --jsonl DEMO/outputs/trace_episode_smoke.jsonl`

`outputs/` 为 Git 忽略目录，可能包含本机烟雾测试产物，但仓库不保证提供可直接使用的预训练模型。新模型评估前应同时确认 `ppo_scheduler.zip` 和 `ppo_scheduler.metadata.json` 存在；旧模型必须显式使用 `--allow-legacy-model`。

当前默认配置下，`max-task-reward-gain`（配置 `seed = 7`）的已验证结果：

1. 共 `23` 个 step，控制广播 `23` 次。
2. 非法动作数为 `0`。
3. 完成任务 `36` 个，总任务奖励约 `248.84`。
4. V2I 平均交付约 `9.13` 个完整包/step。
5. V2V 平均交付约 `5.43` 个完整包/step。
6. 平均 `CAoI` 约为 `2.675`。
7. 相同 seed 的完整汇总逐项一致，不同 seed 会产生不同结果。

任务 PPO 当前额外验证状态：

1. 端到端烟雾训练、模型保存、metadata 生成、重新加载和完整回合评估已通过。
2. 周期 checkpoint 与同名 metadata sidecar 已通过 smoke 验证。
3. 20 个 seed 的随机 `MultiDiscrete` 回放均未产生非法环境动作。
4. 模型 `top_k`、`action_slots`、场景/配置指纹或编码版本不一致时会明确报错。
5. 没有 metadata 的旧模型默认拒绝，只有显式 `--allow-legacy-model` 才能加载。

## 11. 当前定位

这个工程当前更像“论文任务骨架 + 可替换实验平台”，重点在于：

1. 用真实城市导出的车辆与建筑数据建场景
2. 把 `RSU/V2I/V2V/CAoI` 这套机制跑通
3. 让三个任务感知 baseline 能在同一任务环境里比较
4. 为后续替换更复杂信道、完整学习算法或 `CAoI` 变体保留低耦合边界

当前需要明确的一点：

1. 当前主线已经从纯 CAoI 调度升级为 freshness-aware 任务完成调度。
2. 物理层仍然是工程近似，不应表述成“完整复现论文全部实验系统”。
3. `PPO` 已适配任务驱动环境、完整 packet、资源块候选和多链路动作集合。
4. PPO 与任务 baseline 现在使用同一环境口径；正式比较时仍需固定训练预算、模型版本和评估回合数，并明确训练 seed 与独立评估 seed 区间。
5. 当前严格动作掩码和自动断点续训仍待完善；周期 checkpoint 已加入，但中断后仍需手动选择 checkpoint。

如果后续继续扩展，建议优先保持模块边界稳定，而不是把环境、信道和策略逻辑重新耦合到一起。
