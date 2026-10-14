# 6G：面向车辆风险判断的感知信息通信调度实验平台

本文于 **2026-10-14** 整理，实验与验收结果按各节记录的执行日期解读。当前设计、使用说明和有效实验结论统一保留在本 README；接手约束保留在根目录 `AGENTS.md`，不再单独维护两套文档目录。主工程位于 `DEMO/`，所有命令均从 `D:\vscode\6G` 执行。参数以 YAML 为准，原始实验依据保留在 `outputs/`。

项目使用《Cities: Skylines 1》导出的车辆轨迹和建筑快照，研究：**车辆感知受限、通信资源有限时，怎样选择信息内容、接收车辆、发送时机和传递路径，使通信更有效地服务交叉风险判断。** 调度器选择上报、下发、中继哪些对象快照以及何时停止发送；车辆使用实际掌握的信息和固定运动模型判断，环境再用真实轨迹评测。

感知与通信的结合目前发生在信息决策层：共享观测补充接收车的感知盲区，任务与期限帮助选择消息。尚未建立感知改善无线波束、信道速率或实际驾驶控制的物理闭环。

目前已经完成独立风险环境、规则策略、固定通信对照、统一损失与费用评分，以及专用学习器的实现和短流程验收。正式固定预算训练与独立新场景验证尚未开展；四组学习烟雾终态的贪心评估均为 STOP、零发送，尚未取得学习性能收益。

## 1. 当前研究、完成程度与旧工程的关系

### 当前完成程度

| 工作 | 已完成的内容 | 结论与剩余事项 |
|---|---|---|
| 真实场景与风险环境 | 固定车辆轨迹、建筑遮挡、感知、快照缓存、逐跳通信和风险任务评测 | 当前场景为 16 辆车、18 个建筑、23 step、46 个风险任务 |
| 风险模型与评分 | B 因果自身运动模型、A 额外权限对照、对象级严格损失与实际发送费用 | 模型修整已结束，任务、真值和阈值冻结；已知失败继续保留 |
| 规则通信实验 | 普通策略、期限停止、版本/需求上报、中继及费用配对 | B 中需求上报将 57 次发送降至 16 次，全部任务评分不变；中继减少部分负任务弃权，真对象成功仍为 6/8，详见第 6 节 |
| 专用风险学习器 | S0 有限回合 PPO、合法组合与 STOP、模型保存/重载、四组短流程验证 | 四组烟雾终态贪心均 STOP、零发送；尚未证明学习性能收益，详见第 7 节 |
| 固定预算训练 | 已实现运行入口并明确预算、seed 与评估合同 | 200 回合/模型的固定训练尚未执行，须后续明确指令 |
| 动作域扩展与泛化 | 已明确后续条件 | S1 尚未实现，独立新场景验证尚未开展 |

### 与旧工程的关系

| 实验线 | 研究内容 | 当前状态与入口 |
|---|---|---|
| **独立感知风险通信，当前研究重点** | 用有限上报、下发和车辆中继共享对象快照，优化风险判断损失与发送费用 | `RiskCommEnv`、规则对照和专用学习器已实现；B 为主实验，A 为显式对照 |
| 25 包任务与 CAoI，旧实验兼容维护 | 完整包调度、任务完成、信息新鲜度、V2I/V2V 中继 | `TaskRelaySchedulingEnv`、三个规则 baseline 和任务版 SB3 PPO 保留原入口；维护说明见第 9 节 |
| 经典 RL 对比，独立工作区 | PPO、REINFORCE、A2C、TRPO、DQN 等对照 | 历史接手记录指向 `D:\vscode\6G-classic-rl-baselines`；不能假定当前 DEMO 包含该分支功能 |

风险线与旧 25 包任务线使用独立环境、动作、评价和学习接口。风险线复用场景、几何和通信可达性，不使用旧线的 2.4 Mbit 包、RB 分配和总速率预算。两条线的 reward、任务数和 CAoI 不能直接混成同一张性能表。

`risk.yaml` 的 `enabled`、窗口预测和车辆中继默认均关闭。独立风险脚本是显式实验入口，可以在 `enabled: false` 下运行；旧脚本不会因此切换到风险环境。当前优先研究通信调度、信息选择和中继，保持风险模型、任务真值与阈值固定。

## 2. 仓库结构、数据与运行准备

```text
6G/
├── README.md                    # 全仓库说明与实验入口
├── .gitignore                   # 全仓库唯一的忽略规则
├── AGENTS.md                    # 接手约束与已确认设计决策
├── 总体思路.md                  # 本地研究设计，不纳入 Git
├── Cities_Skylines_data/         # 游戏导出的车辆与建筑数据
├── Mods/                        # 自建车辆轨迹/建筑快照采集 Mod
└── DEMO/
    ├── pyproject.toml            # Python 包与可选 RL 依赖
    ├── data/config/              # train.yaml 与 risk.yaml
    ├── src/skylines_demo/        # 场景、几何、信道、环境、策略和学习器
    ├── scripts/                  # 检查、回放、对比、训练和评估入口
    └── outputs/                  # 本地模型、日志、JSONL 与实验结果，Git 忽略
```

当前采集 Mod 为 `VehicleTrajectoryRecorder` 和 `BuildingSnapshotRecorder`。历史文档提及的 `DualMind` 是早期参考工程，不是默认入口，也不列作当前工作树中必需存在的目录。经典 RL 隔离工作区保持独立，未经明确指令不合并、推送或删除。

默认数据固定为 `20260623_151408` 会话：

- 车辆：`Cities_Skylines_data/VehicleTrackLogs/20260623_151408/`。共 17 个 CSV，`vehicle_2810.csv` 为空，实际 16 辆有效车辆。
- 建筑：`Cities_Skylines_data/BuildingTrackLogs/20260623_151408/`。当前读取 18 个建筑快照，使用位置、朝向、尺寸、中心偏移和高度构造遮挡。
- 当前回合为 23 个真实轨迹 step；一个 step 对应一条有效采样记录，不额外聚合 16 帧、不插值或延长轨迹。
- 回放帧与旧任务环境的 step 时长使用相邻 `sessionSeconds` 的前向差；末步或缺失时使用 `fallback_step_seconds=0.3`。旧任务动作完成时刻与下一观测时间连续。
- 风险线的实际 `elapsed_seconds` 同样取真实回放时间；在线 `RiskObservation.delta_seconds` 与窗口预测则只取当前及历史正时间间隔的中位数，没有历史时回退到 0.3 s，不读取下一帧间隔。这两个时间口径不能混用。
- 单 RSU 坐标为 `(1350.894, 180.0, 2386.964)`，场景 seed 为 `train.yaml` 顶层的 `7`。

需要 Python 3.10 或以上。公共脚本会添加本地 `DEMO/src`，直接检查场景和运行规则策略不要求先安装可选 RL 依赖。需要包安装或学习功能时，在根目录执行：

```powershell
Set-Location D:\vscode\6G
python -m pip install -e ./DEMO
# 使用旧 SB3 PPO 或独立风险学习器时安装可选依赖
python -m pip install -e "./DEMO[rl]"
```

DEMO 打包使用 `pyproject.toml` 内联的简短包说明，完整文档集中于本 README，构建时无需读取包目录外的文件。setuptools 构建源码包时会提示包内没有常规 README；这是集中维护文档后的提示，源码包与 wheel 构建已验证通过。缓存和 `*.egg-info/` 是可再生成产物；`outputs/` 虽然被 Git 忽略，却可能包含不可替代的实验证据与模型，不能仅因被忽略就删除。

## 3. 独立风险环境：信息怎样流动

一次回合中的信息流为：**真实场景 → 当前感知 → 对象快照 → 通信调度 → 逐跳投递 → 接收车风险判断 → 离线计分。**

车辆感知范围默认 40 m、视野 120°；RSU 为 120 m、360°，还需检查遮挡。感知与通信分别判断，能看见不代表链路一定可达。通信复用 `train.yaml` 的信道参数、端点发射功率和 `reachable` 布尔规则，候选与实际执行使用同一规则。

| 类型 | 内容与路径 | 默认单步名额 |
|---|---|---:|
| 上行 U | 车辆将其合法掌握的对象快照上报 RSU | 2 |
| 下行 D | RSU 将对象快照下发给接收车辆 | 2 |
| 中继 R | 车辆把合法掌握的快照转发给另一车辆 | 显式开启时 1，默认关闭 |

任一车辆同 step 最多参与一条数据链路；禁止向接收车发送它自己的对象信息。非法动作整体失败，不做部分投递。控制面理想可靠、不占数据名额，也不计发送费用。

快照保留对象原始状态、观测 step、来源和速度依据。**本 step 发送，最早下一 step 可用。** 上行后再下行、接收后再中继都需要后续 step；多次 V2V 接力不会刷新原始观测时间。上行允许将 RSU 当前本地同版本存入数据面缓存，供失去本地可见性后保留；下行和中继要求接收车的有效版本严格更新。

| 信息 | 可使用范围 |
|---|---|
| 本 step 本地可见对象 | 当前风险判断与合法发送 |
| 此前实际通信接收并保留的快照 | 按合法版本参与判断和转发 |
| 纯本地历史观测 | 仅用于速度估计，失见后不单独授予当前判断、控制摘要或转发资格 |
| 未来真实轨迹、任务标签、真实相关对象 | 离线评测与明确标注的诊断；普通在线策略不读取 |

先使用 step `t` 已经可用的信息判断到期任务，再执行动作 `a_t`。因此 `a_t` 不能改变当步已经作出的任务判断，未来可用信息才可能影响后续任务。

风险线的通信 CAoI 是辅助诊断，基于最后实际数据面接收版本，字段 `mean_caoi_steps` 的单位是 step；本地感知刷新不降低它。从未收到的对象不计该 CAoI，零接收车辆单列、值为 `null`。缓存方法虽沿用 `caoi_seconds` 名称，实际计算的是 step 年龄，不能与旧任务线的秒单位 CAoI 直接混用。

## 4. 风险任务、真值与 A/B 模型

离线从轨迹相交位置形成并合并冲突区域；车辆在区域内的一次连续访问形成一次经过，包含圈内静止时间，出圈后再进入另计。代表时刻取本次访问距离区域中心最近的时刻，并列取最早；静止最近点使用本次访问内最近移动段的航向。

默认固定场景为 **15 个冲突点、60 次经过、46 项判断任务，其中 8 正、38 负**。判断期限在经过前 4 个采样步。真值正任务要求另一车辆经过同一区域、经过时间差不超过 1.5 s，且航向差在 45°～135°。事件区域为已有的 8 m 合并区域；5 m 核心圆仅作诊断。

这是交叉风险事件定义，不等于实际碰撞。轨迹固定，当前没有刹车、避让、路径控制或通信改变车辆运动的闭环。修正经过算法后必须重建任务与真值，旧 48 任务结果不能直接与当前 46 任务作策略性能比较。

在线模型使用两次合法位置观测的位移与真实时间差估速，配合快照当前航向补偿信息年龄，再估计经过区域的时间。导出的 `speed` 字段不能直接当作 m/s。两种模式共用区域、交叉航向和预计经过时间差判据，区别在自身 ETA 的信息权限：

| 模式 | 自身 ETA 的信息 | 地位与模型版本 |
|---|---|---|
| **B：`causal-self-motion`** | 自身当前状态与合法历史运动；忽略公开 schedule，既有预测窗内无法预测自身经过时弃权 | YAML 默认主实验；`crossing-area-heading-eta-v3-causal-self-motion-v1` |
| A：`public-crossing-step` | 离线真实经过生成的公开自身 `crossing_step`，结合历史采样间隔估 ETA | 显式权限增强对照；`crossing-area-heading-eta-v3` |

A 的经过信息是额外授予的自身行程权限，不能描述为现实中天然可得的导航信息。B 的因果性限定于在线 ETA/模型；两种模式仍共用离线生成的任务地图与公开判断期限，不能宣称整套任务生成均因果。B 不得由期限加 lead 反推 ETA。兼容接口中直接构造 `RiskCommConfig()` 未指定模式时仍可默认 A；项目实验脚本加载 YAML 后采用 B。

| 判断输出 | 含义 |
|---|---|
| `risk` | 至少一个已知对象满足事件判据 |
| `no_risk` | 已知对象均可判断，没有预测到事件；不证明未观测区域安全 |
| `timeout` | 没有可用的其他车辆对象信息 |
| `insufficient` | 有对象，但缺少速度等依据或自身 ETA 不可估，且没有对象明确触发风险 |

后两者都计弃权，负任务不计正确，正任务仍计风险识别失败。有明确风险的对象优先触发报警，不能被其他对象缺少速度压掉。恒速、恒航向近似对转弯、启停和静止后再运动有局限；当前模型有限修复已结束，后续通信研究保留这些失败。

失败归因按纯通信、纯模型、混合三类互斥记录，并保留对象级诊断。比较通信或模型相关原因时分别包含混合项，不能把对象失败数相加当成任务失败数。独立合法路径机会不代表可共同实现的系统容量；无限制贪心和受限诊断 oracle 都不是已经验证的最优上界。

## 5. 当前优化目标与计分口径

统一公式位于 `DEMO/src/skylines_demo/risk/scoring.py`：

```text
LΣ = 5 × F + 1 × FP + 0.25 × A⁻
JΣ = LΣ + 0.01 × M
L  = LΣ / N
J  = JΣ / N
```

- `F`：正任务严格失败数。至少报警一个真实相关对象才成功；仅错误对象报警，即使二值答案为“有风险”，仍失败。
- `FP`：负任务误报数。
- `A⁻`：负任务弃权数，包含 timeout 与 insufficient。
- `M`：实际上行、下行和中继的发送总数，每一跳计一次；共享消息不按它服务的任务数重复收费。
- `N`：完整任务目录数量，当前为 46。

正任务弃权或仅错对象报警只扣正任务失败的 5，不叠加负任务罚项；真实与错误对象同时报警可以成功，额外错误对象另列。旧二值 `correct`、`missed`、`false_positive` 字段保留原含义，严格 `positive_failure_count` 不能用旧漏报代替。这些权重是固定实验评分，不代表事故概率或现实经济损失。

每步反馈为 `r_t = -到期任务损失 - 0.01×实际发送数`；完整有限回合取 γ=1，累计 reward 等于 `-JΣ`，不添加 CAoI shaping。任务在规定期限按 task_id 仅结算一次；step 0 的损失在 `initial_task_loss_sum` 单列，仍计入回合目标。无人使用、重复或未改变缓存的实际发送也收费；非法动作发送费用为 0，但到期任务照常结算。

`--steps` 截断只记录已执行前缀，不补未到期任务，正式均值与比例为 `null`，完整目录和待结算 task_id 保留。末步发送仍收费，不增加真实轨迹帧或重新判题。

| 输出字段 | 用途 |
|---|---|
| `objective` | 严格成功/失败、三项损失、U/D/R 费用、J、名额利用率和结算完整性 |
| `task_scores` | 每个任务的唯一计分与真实/额外错误报警对象 |
| `reward_feedback` / `reward` | 每步真实目标反馈与回合 reward 核对 |
| `delivery_records` / `delivery_schedule` | 原始版本、来源、发送 step、最早可用 step 和每跳路径 |
| 旧二值、错误归因、通信 CAoI | 补充诊断，不能替换当前主目标 |

## 6. 已完成的规则通信实验与结果

已实现无通信、年龄、事件触发、期限、接收端价值等策略，及显式中继、期限停止、合法版本上报和公开需求上报。`receiver-value` 的排序分数是规则启发式，不是环境 reward、成功概率或已知真实收益。

2026-10-09 固定研究运行了 88 项完整回合组合并重复核验，保存 116 组全任务配对。B 主实验固定场景 seed=7、23 step、46 任务、2U+2D，显式中继时 1R，窗口预测关；以下主表不设整回合发送 cap：

| 策略 | 真对象成功 / 8 | F | FP | A⁻ | M | LΣ | JΣ（越小越好） |
|---|---:|---:|---:|---:|---:|---:|---:|
| `no-communication` | 5 | 3 | 0 | 24 | 0 | 21.00 | 21.00 |
| `age` | 5 | 3 | 0 | 28 | 92 | 22.00 | 22.92 |
| `event-trigger` | 5 | 3 | 0 | 22 | 92 | 20.50 | 21.42 |
| `deadline` | 5 | 3 | 0 | 16 | 92 | 19.00 | 19.92 |
| `receiver-value` | 6 | 2 | 1 | 18 | 57 | 15.50 | 16.07 |
| `receiver-value-relay` | 6 | 2 | 1 | 14 | 73 | 14.50 | 15.23 |
| `receiver-value-stop` | 6 | 2 | 1 | 18 | 47 | 15.50 | 15.97 |
| `version-uplink` | 6 | 2 | 1 | 18 | 17 | 15.50 | 15.67 |
| `demand-uplink` | 6 | 2 | 1 | 18 | 16 | 15.50 | 15.66 |
| `demand-uplink-relay` | 6 | 2 | 1 | 14 | 32 | 14.50 | 14.82 |

完整 A/B、其他方法、U/D/R 分项和全部预算结果见本地 [study_final 汇总](DEMO/outputs/risk_fixed_b_comm_20261009_171625_3029627/study_final/summary.json)，[配对索引](DEMO/outputs/risk_fixed_b_comm_20261009_171625_3029627/study_final/pair_index.json)保留全部 46 任务的对象与路径依据。2026-10-13 以当前代码复核了无通信、receiver-value 和显式 receiver-value-relay 三项，结果与上表一致；其他行来自既有固定实验记录。

A 是自身行程权限增强对照，单列如下；其额外权限不能解释为 B 调度算法的收益：

| A 策略 | 真对象成功 / 8 | F | FP | A⁻ | M | LΣ | JΣ |
|---|---:|---:|---:|---:|---:|---:|---:|
| `no-communication` | 5 | 3 | 1 | 21 | 0 | 21.25 | 21.25 |
| `receiver-value` | 5 | 3 | 3 | 10 | 62 | 20.50 | 21.12 |
| `receiver-value-relay` | 6 | 2 | 3 | 12 | 80 | 16.00 | 16.80 |
| `demand-uplink` | 5 | 3 | 3 | 10 | 21 | 20.50 | 20.71 |
| `demand-uplink-relay` | 6 | 2 | 3 | 12 | 39 | 16.00 | 16.39 |

结果需要分别解释：

- **节省发送**：期限停止将 57 次降至 47 次；版本/需求上报进一步降至 17/16 次，对应全部 46 份 `task_scores` 与原 receiver-value 一致。支持当前场景存在冗余发送，不代表信息流完全相同或跨场景无损。
- **中继减少部分弃权**：需求上报开中继时，发送 16→32，负弃权 18→14，具体为 7 项消除、3 项新增；真对象成功仍为 6/8，没有进一步增加正任务识别成功。当前不能概括为普遍降低危险事件漏报。
- **等费用比较**：预设 cap 为 20/30/40/47；费用匹配先完整运行两臂，取实际 M 的较小值作 cap，再重跑。按 D→R→U 截取，不填充无用发送。同 cap 不一定等实际 M；只有实际 M 相等才能标作等费用。等 M=16 的中继关/开 JΣ 为 15.66/14.41，其中含误报转弃权，不能称为正确安全判断。

上报版本筛选可能放弃 RSU 同版本缓存留存机会，在其他场景可能有损。窗口预测尚无独立有效收益证据，默认保持关闭。预测的 `early_send_count` 需要与关闭预测完整回放的全部下行/中继真实投递比较，确认同等或更新版本更早送出；仅预测侧送出的版本另计 `prediction_extra_send_count`，不能混称提前发送。预测只读取当前/历史轨迹和采样间隔，不读取未来真实状态或采样时间。

当前 B 价值策略及其上报改进仍保留两项严格失败：

| 任务 | 已知原因与结论边界 |
|---|---|
| `risk-10697-11-1` | step 0 的真实对象 3460 缺合法速度，输出 insufficient，损失 5 为初始常数；不能归因首个动作或靠下一步消息救回 |
| `risk-4966-5-2` | 仅错误对象 11370 报警，真实对象 8641 未知，严格损失 5；旧二值答案虽然为 risk，仍没有真实对象识别成功，尚未证明合法策略可恢复 |

`risk-1380-5-3` 也说明二值答案的局限：B 无通信时已经因错误对象报警，普通 RSU 下行又补到了真实对象 15325，但 outcome 没变。这是对象依据恢复，不能写成旧漏报下降或中继独有收益。反事实须完整重演无通信回合和速度估计，不能删除通信缓存后仍保留通信学到的速度。

## 7. 专用风险通信学习器：实现与实际进展

2026-10-10 已实施 `DEMO/src/skylines_demo/risk_learning/`，学习上报、下发、中继消息的选择与 STOP，保持风险模型和评分固定。它是专用有限回合 PPO，不复用旧 SB3 的 25 包任务编码。

首轮限定在 S0 `receiver-risk-domain-v1`：复用接收端风险域资格，所有候选保留、不截 top_k。候选为 32 维、全局为 6 维安全白名单投影，支持变长 padding；共享候选网络、自回归逐项选择、动态合法掩码和独立 STOP 门控。逐项组合不推进世界时间，组合完成才执行一次真实 step。

S0 是策略研究范围，筛选之前的物理合法候选与筛选之后的可选候选必须区分：

- U 保留全部环境合法上行，包括可留下 RSU 数据面缓存的同版本上行；版本/需求偏好交给策略，不做硬掩码。
- D 限定为接收端存在后续公开任务、显式 A/B 固定模型当前预测 risk 的合法下行；不代表其他合法快照物理不可达。
- R 开启时沿用接收端有未来任务、发送车版本更新的合法中继资格；关闭时无 R 候选。
- STOP 始终允许。同域 receiver-value、期限停止、版本/需求上报及中继适配保持原排序与原动作/计分结果；age 等全域结果单列，限制到 S0 后需另命名。

Actor/Critic 不接收标签、真实相关对象、反馈或完整控制运动表；B 不读 crossing 或重建自身 ETA，上行/中继不绕过权限读取私有运动 payload。反馈和对象级证据保留在离线训练目标与评测分支。

候选键只用于查找和端点冲突，不学习车辆/任务 ID embedding，不直接把完整 `RiskObservation` 或日志字典送入网络。D 的模型派生值来自 RSU 合法快照，目标接收车自身运动仅用于其自身预测；U/R 不公开私有对象运动，也不能借 RSU 的其他版本冒充待中继内容。默认无 RNN，不累计历史评测反馈或失见对象运动。

编码顺序、尺度与缺失位以 `risk_learning/encoding.py` 为准，不按真值拟合归一化器。共享候选网络为两层 64 单元 Tanh，均值池化排除 padding；无候选用零池化并强制 STOP。STOP 独立门控初始概率 0.2，不随候选数增多自动稀释。组合最多选 4 条真实消息，开中继时 5 条；一个联合组合只算一个世界 step。联合路径概率保留选择顺序，重算使用同一安全快照和逐选掩码，贪心解码不代表最优集合。

训练使用完整 23 step Monte Carlo 回报，γ=1、β=0、熵系数 0，无额外 shaping；CPU、Adam 学习率 3e-4、clip 0.2、value 系数 0.5、梯度裁剪 0.5。训练 seed 只影响初始化、采样与优化器，场景 seed 固定 7，不能把多训练 seed 当作多场景验证。

| 阶段 | 每组回合 / 世界 step | rollout / minibatch / epoch | PPO 更新 | checkpoint |
|---|---|---|---:|---|
| `smoke` | 2 / 46 | 2 回合 / 23 / 1 | 1 次，含 2 个优化器 step | 0000、0002 |
| `fixed`，已实现但尚未执行 | 200 / 4600 | 20 回合 / 92 / 5 | 10 次 | 0000、0100、0200；主结果只取 0200 |

已完成 B/A × 中继关/开四组，训练 seed 均为 7、每组 2 个完整回合，共 184 个训练世界 step。训练、保存、严格重载及两遍完整贪心回放均有验收记录。**四组终态贪心都选择 STOP、零发送：B 的 JΣ=21，A 的 JΣ=21.25。** 这证明实现和证据流程可运行，不证明学习收益、收敛、最优或泛化；随机训练回合曾发送消息，其分数波动也不能当作学习改善。

模型为 `episode_XXXX.pt`，必须配同名 `episode_XXXX.metadata.json`。metadata 校验模式/权限、域、网络、编码、组合、评分权重、配置、源码/数据/任务内容、依赖、模型哈希与实际预算；不存在跳过校验或 legacy 入口。checkpoint 包含优化器和 Python/NumPy/Torch/独立 generator 状态，但尚无自动续训。源码漂移时拒绝加载，不得手改旧 sidecar 掩盖差异。

训练与评估输出独占创建，已有目录/JSONL/checkpoint 拒绝覆盖；异常保留 partial，不优化截断回合，不自动续训。烟雾模型不接续 fixed；固定阶段从头初始化，预设 seed 为 7/17/27/37/47，B 在先、A 分表，中继关/开分别报告。

固定方案若包含 B/A × 中继关/开 × 5 seed，共 20 个模型、92000 世界 step；只批准 B 则为 10 个模型、46000 step，明确 A 学习对照未执行。预算用尽即停，不因无收益自动追加训练、筛 seed 或挑 checkpoint；该内部探索预算仍不证明收敛或论文性能。按模式/域/中继报告所有 seed、均值、样本标准差与范围，不把同一 46 任务打散、切时间段或按车辆拆分当独立测试集。重复贪心只核对一致性；随机推理 seed=1000007..1000016，先按模型汇总，再比较训练 seed，不能当作独立场景样本。

中继分别训练的关/开模型反映路径可用性与重新优化；开中继父模型的 R 屏蔽评估是固定权重下的分布变化诊断，两者分开报告。每次完整回放保留 46 任务、对象依据、消息原始来源/版本、每跳时序、缓存前后状态及全部 cap/配对；共享路径费用不可跨任务相加。只按前次 step 匹配的估速依据标“可能支持”，不能当唯一来源证明。

完整工程验收、训练/重载结果、逐对象与逐跳证据见[本地学习执行报告](DEMO/outputs/risk_learning_20261010_112722_812903/report.md)及[交付索引](DEMO/outputs/risk_learning_20261010_112722_812903/artifact_index.json)。固定预算、S1 动作域扩展、新场景和自动长训练尚未运行，须后续明确执行指令。

## 8. 独立风险实验的操作入口

### 场景、规则检查与普通策略对比

```powershell
python DEMO/scripts/inspect_scene.py
python DEMO/scripts/check_risk_scene.py --json
python DEMO/scripts/check_risk_rules.py --json
python DEMO/scripts/check_risk_relay.py --json
python DEMO/scripts/check_risk_mechanisms.py
python DEMO/scripts/run_risk_compare.py --policies no-communication,age,event-trigger,deadline,receiver-value --json
```

普通 compare 可选 `related`、`unrelated`、`receiver-value-relay`；`diagnostic-oracle` 与 unlimited 系列是有额外权限或资源的诊断，不作为普通策略的公平成绩。版本/需求上报和期限停止组合通过专用固定研究入口运行，不能直接把这些名字传给普通 compare。

### 显式比较中继与窗口预测

```powershell
python DEMO/scripts/run_risk_compare.py --policies receiver-value-relay --compare-relay --json
python DEMO/scripts/run_risk_compare.py --policies receiver-value --compare-link-prediction --json
```

compare/trace 的开关为三态：省略保留 YAML，`--enable-v2v-relay` / `--link-prediction` 开启，`--no-enable-v2v-relay` / `--no-link-prediction` 关闭。使用中继策略名时仍须核对实际开关，输出会回显生效配置。

### 逐步回放、固定研究与模型审计

每次输出使用全新文件或目录，不覆盖旧证据：

```powershell
$riskRunRoot = "DEMO/outputs/risk_run_" + (Get-Date -Format 'yyyyMMdd_HHmmss_fffffff')
python DEMO/scripts/trace_risk_episode.py --policy receiver-value --steps 3 --json --jsonl ($riskRunRoot + '.jsonl')
# 完整固定研究：88 项运行、重复检查与 116 组配对，会生成较多证据
python DEMO/scripts/run_risk_fixed_b_study.py --output-dir ($riskRunRoot + '_study')
# 单独审计 B/A；不修改默认 YAML
python DEMO/scripts/audit_risk_model.py --ego-eta-mode causal-self-motion --output-dir ($riskRunRoot + '_B') --repeat
python DEMO/scripts/audit_risk_model.py --ego-eta-mode public-crossing-step --output-dir ($riskRunRoot + '_A') --repeat
```

JSONL 只写逐 step 记录，包含动作、投递、判断、计分和下一状态；终止/截断标志及总摘要由 `--json` 提供。`--steps 0 --jsonl` 产生空文件，零步目录与待结算情况需查看摘要。A 的普通 compare/trace 可使用独立风险配置副本，只修改 `ego_eta_mode` 后传入 `--risk-config`，其余参数固定；学习与固定研究入口自行显式选择模式。

### 学习合同检查与小烟雾

以下训练示例仅用于获得对应执行授权后的短流程检查；阅读本 README 或通过检查不自动启动训练：

```powershell
$riskLearningRoot = "DEMO/outputs/risk_learning_" + (Get-Date -Format 'yyyyMMdd_HHmmss_fffffff')
$env:PYTHONUTF8 = '1'
# 四配置完整推理合同检查，不训练
python DEMO/scripts/check_risk_learning.py --output-dir ($riskLearningRoot + '/check')
# 单组 B、关中继、2 回合小烟雾
python DEMO/scripts/train_risk_scheduler.py --mode B --relay off --domain receiver-risk-domain-v1 --train-seed 7 --episodes 2 --phase smoke --output-dir ($riskLearningRoot + '/smoke_B_off_seed7')
# 严格重载同名 sidecar，贪心两遍完整回放
python DEMO/scripts/evaluate_risk_scheduler.py --model-path ($riskLearningRoot + '/smoke_B_off_seed7/episode_0002.pt') --metadata-path ($riskLearningRoot + '/smoke_B_off_seed7/episode_0002.metadata.json') --decoding greedy --repeat 2 --output-dir ($riskLearningRoot + '/eval_B_off_seed7')
```

`train_risk_scheduler.py` 必须明确 mode、relay、train-seed、episodes、phase 和 output；无参数退出，不训练。`smoke` 只接受 2 回合，`fixed` 只接受 200。评估 `greedy` 必须 repeat=2，`stochastic` 补充必须 repeat=10；`--disable-relay-diagnostic` 仅用于已开中继父模型的独立 R 屏蔽诊断，并保存父模型哈希。`--global-cap` 可显式取 0/20/30/40/47，不改主训练。

固定训练矩阵入口为 `run_risk_learning_suite.py`，先写不可变冻结清单，各组从头初始化。当前只查看参数，不将固定预算当作本轮已授权工作：

```powershell
python DEMO/scripts/run_risk_learning_suite.py --help
```

## 9. 旧 25 包任务与 CAoI：兼容维护

本节保留旧任务平台的接口、规则和历史回归依据；当前研究优先级见第 1 节，独立风险环境与学习器见第 3～8 节。旧任务线使用 `DEMO/data/config/train.yaml` 与 `TaskRelaySchedulingEnv`。共 25 种包，ID=1..25，每种 2,400,000 bit（0.30 MB）；每车维护各包时间戳和真实持包标志，CAoI 是全部 packet age 的均值。

初始 CAoI=0、真实持包标志为 False，只是统一仿真起点。接收车尚未真实持包时允许等时间戳真实包覆盖占位；已有真实版本时要求严格更新。只交付完整包，容量为 `floor(rate_mbps×1e6×delta_seconds/packet_size_bits)`；不足一包不更新、不保存部分进度、不跨 step 累积剩余 bit。V2V 只能转发动作前真实持有的包，同 step 禁止先收 V2I 再转发。

默认 8 个任务模板，每任务需 7～8 种包；同车任务不重复、每回合最多完成 3 个。完成后冷却 1 个完整 step：t 完成，t+1 不申请，最早 t+2 申请。缓存真实包可服务后续任务。模板为：

```text
T1: 1,2,3,5,6,8,12       T2: 1,3,9,10,15,16,20
T3: 2,4,6,11,14,20,25   T4: 2,5,7,8,12,17,21,25
T5: 1,3,6,9,13,18,22,24 T6: 1,4,8,10,11,15,19,23
T7: 2,3,5,7,14,16,20,25 T8: 4,6,8,12,13,17,21,24
```

任务完成 reward 为 `10×mean(exp(-AoI_i/4))`。`task_reward_gain` 表示本动作相对动作前状态的增量；动作前缓存已经满足的任务可能计入整步 `task_reward`，不能归因给本动作或直接作 PPO reward。任务检查在 action 数据更新后统一执行，CAoI 报告 action 后的 `A_{t+1}`。

V2I/V2V 各有独立 40 MHz 资源池、8 个 5 MHz RB；单链路最多 2 RB，总速率分别为 100/60 Mbps，单车链路上限 30 Mbps。RSU 可并发多 V2I，每车同一步仍只能参与一条数据链路。动作显式指定链路、RB 和包 ID；端点冲突、未知候选、类型归属错误、RB 或速率超限使整组动作 invalid，不更新包也不计资源占用。

信道基于三维距离、建筑 footprint/高度 LoS/NLoS、对数距离路径损耗和轻量车辆遮挡。先按实际 RB 带宽计算噪声、SNR、物理速率，再施加限制。默认 RSU/车辆功率 26/23 dBm、NLoS 损耗 35 dB、车辆遮挡 22 dB、实现余量 6 dB、速率效率 0.55；完整参数以 YAML 为准。这是工程近似，不是 Sionna/ray tracing 物理层完整复现。标准 0.3 s 下 RSU 理论容量约 12 包，规则检查要求存在至少约 10 包的可行单步 V2I 调度。

自动驾驶优先默认关闭，渗透率 0、权重 1.0；启用后是软调度优先，不改变轨迹、真实 CAoI 或中继资格。`max_steps=null`、`max_active_links=null`、`allow_multi_v2i_from_rsu=true`，不为了提升某次结果改变动作约束。

三个 baseline 为 `random-task-aware`、`max-task-reward-gain`、`two-stage-task-relay-greedy`。旧命令 `random-set`、`max-caoi-gain`、`max-aoi-gain`、`two-stage-relay-greedy` 仅作映射兼容。两阶段策略的中继扩散指标只统计实际可达、满足任务需求与完整包容量的 peer；当前候选仍按包优先级前缀选择，没有缺失包子集或全局边际收益/RB 优化。

```powershell
python DEMO/scripts/check_links.py --steps 5 --candidates 4 --verbose
python DEMO/scripts/run_experiment.py --policy max-task-reward-gain
python DEMO/scripts/run_experiment.py --policy two-stage-task-relay-greedy --full-output
python DEMO/scripts/trace_episode.py --policy max-task-reward-gain --steps 2 --show-all-vehicles
```

`check_links` 查看动作前候选和计划调度；`trace_episode` 核对实际包更新、下一步中继和任务完成。完整回放可加 `--jsonl DEMO/outputs/<新名称>.jsonl`；`run_experiment --summary-json <路径>` 保存完整摘要。固定 seed 可复现任务顺序，独立任务/身份随机流互不扰动。

任务版 SB3 PPO 使用固定 `MultiDiscrete`，默认 top_k=16、action_slots=8、训练 300000 timestep、n_steps=512、batch_size=128，每 50000 timestep 保存 checkpoint，进度间隔默认 5000。top_k 对全部候选扁平截断，可能使 V2V 很少进入；改为 32 会改变编码，需要重训与匹配 metadata。普通 SB3 PPO 不消费 action_masks，组合器按槽顺序跳过重复、车辆冲突和资源超限项，前槽优先。invalid_action_count=0 只说明最终环境动作合法，不代表槽都有效或充分使用链路。

```powershell
# 旧任务线的短流程训练，不是独立风险学习
python DEMO/scripts/train_ppo.py --timesteps 512 --n-steps 128 --batch-size 64 --eval-episodes 1 --top-k 16 --action-slots 8 --progress-interval 100 --checkpoint-interval 256 --output-dir DEMO/outputs/ppo_quick_test
python DEMO/scripts/run_experiment.py --policy ppo --model-path DEMO/outputs/ppo_quick_test/ppo_scheduler --top-k 16 --action-slots 8
```

模型保存 `ppo_scheduler.zip` 与 `ppo_scheduler.metadata.json`，校验场景、数据、配置、动作参数和观测编码；旧无 sidecar 模型须显式 `--allow-legacy-model`，新 MultiDiscrete 不兼容旧 Discrete。训练自动 reset 用 `base_seed+episode_index`，评估用训练 seed+1000000 的独立区间。`max_steps` 截短返回 truncated，并以真实下一帧 bootstrap；自然末尾才 terminated。摘要区分请求与实际采样 timesteps，自动断点续训尚未实现。

旧线 seed=7 的历史规则回归基准为 23 step、36 个完成任务、总奖励约 248.84、平均 CAoI 约 2.675，平均每步交付 V2I/V2V 约 9.13/5.43 包。近期 top_k=32 的 PPO 长训练仍有稀疏调度与 V2V 利用不足；这些记录用于流程与瓶颈诊断，不是跨 seed 最终论文结论。所有策略应报告 CAoI、任务奖励/完成数、完整包、资源、非法与失败；PPO 另报告预算、编码、训练/评估 seed 与回合数。

经典 RL 隔离分支保留七种算法（含 double-dqn、dueling-dqn）的统一训练与绘图入口。DQN 三变体为单候选 Discrete，其他算法为多槽 MultiDiscrete；图表须按接口分组，不能写成完全等价的性能比较。当前根目录不提供该分支的命令执行环境，需切换到已保留的独立工作区并阅读其说明。

## 10. 核心模块边界

以下路径以 `DEMO/src/skylines_demo/` 为基准：

| 模块 | 职责 |
|---|---|
| `config.py`、`bootstrap.py`、`types.py` | 配置、场景装配与共享类型 |
| `data/`、`geometry/`、`wireless/` | 数据读取、三维遮挡与可替换数学信道 |
| `perception/` | 感知、快照/缓存、时序、合法历史窗口预测 |
| `risk/conflicts.py`、`risk/model.py` | 离线任务目录、在线固定风险模型 |
| `risk/executor.py`、`risk/evaluator.py`、`risk/scoring.py` | 投递、离线结果核对、唯一计分与反馈 |
| `risk/opportunities.py` | 独立任务对象的感知/合法多跳机会诊断 |
| `env/risk_comm_env.py` | 独立风险环境与状态推进 |
| `policies/risk_*.py` | 原风险 baseline、接收端价值、中继、公开上报及同域训练对照 |
| `risk_learning/domain.py`、`encoding.py`、`actions.py` | S0 白名单投影、32/6 编码与合法自回归组合 |
| `risk_learning/policy.py`、`rollout.py`、`training.py` | 网络、完整回合回报与有限 PPO |
| `risk_learning/metadata.py`、`evidence.py` | 严格模型合同、内容冻结、对象/路径/费用证据 |
| `env/task_relay_env.py`、`tasks/`、`policies/task_policy.py`、`rl/` | 旧 25 包任务线及其 SB3 PPO |
| `env/relay_env.py`、`rl/common.py` | 更早的 CAoI/离散动作兼容实现，非默认主入口 |

信道层不读取任务奖励；策略只选择合法动作，环境负责校验和更新。替换信道、模型、freshness 或学习算法时保持这些边界，不把某个策略的收益写入真值或环境规则。

## 11. 验证入口与已记录的验收

按实际修改范围运行检查。场景、环境、任务或编码改动通常需要相关规则回归；新增逻辑另补与改动相称的验证。当前工作树已没有 `DEMO/tests`，不再把整套 pytest 命令列作现有入口，也不因此恢复旧测试目录。可用脚本从根目录执行：

```powershell
python DEMO/scripts/check_env_rules.py
python DEMO/scripts/check_task_rules.py
python DEMO/scripts/check_ppo_wrapper.py
python DEMO/scripts/check_risk_rules.py --json
python DEMO/scripts/check_risk_scene.py --json
python DEMO/scripts/check_risk_relay.py --json
python DEMO/scripts/check_risk_mechanisms.py
python -m compileall DEMO/src DEMO/scripts
D:\Anaconda\Scripts\ruff.exe check DEMO/src DEMO/scripts
D:\Anaconda\python.exe -m mypy DEMO/src
```

2026-10-10 学习实现的既有验收记录为：340 项 pytest 通过（114 新增、226 既有）、风险及旧环境/任务/PPO 检查通过、编译与 Anaconda Ruff/Mypy 通过，Mypy 覆盖 59 个源文件；原固定研究 88 项/116 配对重现，除执行 manifest 外与历史结果逐字节一致。这些是有日期的验收记录，不能代替后续修改后的重新检查。

当前机器曾出现 Anaconda NumPy 与用户 Torch 的 OpenMP 导入顺序冲突。新学习入口延迟加载并先 NumPy 后 Torch，不使用 `KMP_DUPLICATE_LIB_OK`。Python 启动器可能打印既有路径诊断，须结合退出码和完整日志判断。Anaconda 的 mypy 模块入口可用于此前 mypy.exe DLL 问题；不要因默认 Python 缺静态检查依赖而跳过检查。

## 12. 当前边界与后续工作

已有证据支持可运行平台、固定场景内冗余上报筛选和部分中继减少弃权。任务相关调度是否稳定优于其他方法、学习是否改善目标、结果是否泛化，仍需研究。

后续应在冻结 B 主/A 对照、模型/任务/传感器/名额/数据不变的条件下，获得明确预算指令后运行从头初始化的固定训练矩阵，保留所有 seed/checkpoint/cap 与失败对象，再开展独立新场景验证。同场景重复、训练 seed 变化或较低发送量不能替代泛化、准确率或最优性证据。

S1 `all-legal-snapshot-domain-v1` 尚未实现，只有后续明确允许扩集合时才开展。其 D 不按当前预测 risk/no_risk/未知筛掉，同时必须新增 `all-legal-public-demand`、同域 deadline/age/随机合法 STOP 对照，保留 S0。规则仅使用公开期限、合法版本与模型派生类别，组合次序 R→D→U 是启发式，不是最优；同版本上行仍保留留存机会。S1 从头训练并使用同预算/seed，S1 学习与 S1 规则比较才用于讨论学习收益，S0→S1 单列动作集合变化，不混成公平排名。

信息泄漏、冻结漂移、非法动作或即时转发、原始版本改写、漏/重复结算、回报与 J 不一致、非有限概率/梯度、metadata/拒覆盖失效或误标等费用时，停止当前阶段并保留失败证据。无性能改善本身不等于实现错误，也不授权调阈值、加 shaping、扩场景或继续长训练。

当前没有实现真实避撞控制、干扰、波束管理、多径、Fresnel zone、完整毫米波协议栈或 Sionna/ray tracing。旧任务线不支持部分包跨步累积或即时两跳；风险线也保持逐跳下一步可用。工程应描述为真实轨迹驱动、可替换数学信道的研究平台，不声称完整复现原论文系统。

## 13. 原始证据入口与维护约定

设计依据、当前结论、限制和后续安排已整合进本 README 与根目录 `AGENTS.md`；早期计划、提示词与重复说明不再单独保留。原始输出继续按生成时的模型与源码版本解读，其中默认 A 的历史记录不能覆盖当前默认 B。

| 需要核对的内容 | 原始证据 |
|---|---|
| 模型与通信审计、修复前后回放 | [最终审计 JSON](DEMO/outputs/risk_model_audit_20261008/final/audit.json)、[重复核验](DEMO/outputs/risk_model_audit_20261008/final/reproducibility.json)；同目录保留修复前源码和中间结果 |
| A/B 冻结、对象依据和残留失败 | [汇总 CSV](DEMO/outputs/risk_baseline_freeze_20261009_110412_928121/summary.csv)、[失败清单](DEMO/outputs/risk_baseline_freeze_20261009_110412_928121/remaining_errors.json)、[关键对象数值](DEMO/outputs/risk_baseline_freeze_20261009_110412_928121/final_case_objects.json) |
| 严格计分、回报与重复一致性 | [计分回归记录](DEMO/outputs/risk_objective_20261009_164717_604029/regression_verified.json)；同目录有 compare/trace 完整回放 |
| 固定 B/A 规则、预算与对象路径配对 | [最终汇总](DEMO/outputs/risk_fixed_b_comm_20261009_171625_3029627/study_final/summary.json)、[配对索引](DEMO/outputs/risk_fixed_b_comm_20261009_171625_3029627/study_final/pair_index.json) |
| 专用学习器验收、模型与全部烟雾结果 | [执行报告](DEMO/outputs/risk_learning_20261010_112722_812903/report.md)、[交付索引](DEMO/outputs/risk_learning_20261010_112722_812903/artifact_index.json)、[消息证据索引](DEMO/outputs/risk_learning_20261010_112722_812903/delivery_index.json) |

源码/数据的集合指纹是相对路径到 SHA256 字典的排序紧凑 JSON 哈希，不是 Git commit ID；严格冻结与模型合同以 `risk_learning/metadata.py` 和 sidecar 为准。本次文档整理不修改实验源码、数据、模型或既有输出，也不改旧模型 sidecar。

README 负责当前设计、参数用法与结论，AGENTS 负责接手约束和稳定决策，YAML 负责可调参数，outputs 保存原始证据。未来只补充当前有效信息，无需恢复独立的早期计划和提示词目录；已有输出不因文档合并而覆盖或清理。删除其他文件仍须用户同意。
