# BSense 实时脑电分类与 P300 机器狗控制

这是一个独立运行的 FP1/FP2 双通道 EEG 实时项目，包含两套入口：

- `bsense-classifier`：原有的通用实验模型实时分析界面；
- `bsense-dog-controller`：面向脑控机器狗赛道的 M7 P300 六指令控制台。

通用分类器可以：

- 加载项目内置的 6 个 BSense 模型；
- 自动查找或按名称连接其他软件发布的 LSL EEG 流；
- 校验 FP1、FP2 通道，检测到 FP2/FP1 倒序时自动换序；
- 对连续任务定时推理，对事件任务接收 Marker 或手动触发；
- 在桌面界面显示分类、置信度、各类别概率和信号质量；
- 将每次结果保存为 JSONL，便于后续统计和接入其他软件。

> 原有六个模型属于实验模型，训练对象数量仍少。只有专用
> `bsense-dog-controller` 会经过安全联锁向机器狗桥接层发送指令；通用分类器仍然
> 只输出分析结果。该项目不能用于医疗、安全或驾驶判断。

## P300 脑控机器狗方案

本方案与 `bsense-lsl` 的 `m7_p300` 数据格式配套。采集端的每个
`p300_flash` Marker 已包含 `flash_command`、`flash_position`、`sequence` 和
`is_target`，用于训练“单次闪烁是否为目标”的二分类模型。在线控制时，专用控制台
自己显示同样的六宫格，并按 LSL 时钟把每次闪烁与 FP1/FP2 EEG 对齐；一个 Trial
默认执行 10 个 Sequence，再将同一指令的目标概率求均值，选出：

| 指令 | 桥接命令 | 默认速度 |
|---|---|---|
| 前进 | `forward` | `linear_x=0.35` |
| 后退 | `backward` | `linear_x=-0.25` |
| 左转 | `left` | `angular_z=0.65` |
| 右转 | `right` | `angular_z=-0.65` |
| 急停 | `stop` | 速度归零并锁定 |
| 待机 | `idle` | 速度归零但保持已解锁状态 |

只有同时满足以下条件时，移动指令才会下发：

1. M7 Trial 六个指令都有足够的有效闪烁；
2. 胜出指令的平均目标概率达到阈值；
3. 第一名相对第二名的概率差达到阈值；
4. EEG 质量合格窗口比例达到阈值；
5. 操作员已经显式完成“安全检查并解锁”；
6. 桥接端在线、未急停，并明确返回当前无障碍物。

每个移动指令默认只持续 `0.8` 秒，控制端定时器会再次发送 `stop`。空格、Esc、
界面急停按钮或解码出的“急停”都会立即归零并锁定，恢复前必须重新解锁。

### 当前模型尚未训练时

程序不会用随机模型或硬编码答案冒充脑控结果。没有
`models/m7_p300/model.joblib` 时仍可启动界面：

- 点击六宫格可联调桥接命令和运动方向；
- 默认 `stdout` 传输只打印 JSON，不会移动真实机器狗；
- M7 模型加载前不能连接 P300 解码器，也不能开始连续脑控；
- 人工联调指令会在界面明确标为“人工点击”，不会记录成脑电解码。

控制台会把人工联调与真实 P300 决策分开写入
`logs/m7_control_<时间>.jsonl`，其中 `brain_decoded` 可直接用于视频佐证和赛后审计。

Windows 双击：

`run_dog_controller.bat`

或在 PowerShell 中运行：

```powershell
cd D:\codebase\BCI\bsense-realtime-classifier
.\run_dog_controller.ps1
```

安装项目后也可以运行：

```powershell
bsense-dog-controller `
  --model D:\codebase\BCI\bsense-realtime-classifier\models\m7_p300\model.joblib `
  --stream-name "你的 EEG 流名称" `
  --bridge-config D:\codebase\BCI\bsense-realtime-classifier\config\unitree_bridge.json
```

### M7 模型 artifact 契约

训练完成的模型继续使用现有 `artifact_schema_version=2`，并至少满足：

```text
task                 = m7_p300
target_kind          = classification
label_mapping        = {0: "non_target", 1: "target"}
feature_mode         = erp
deployment_mode      = event_locked_marker_required
channel_count        = 2
window_offset_seconds= -0.2
window_seconds       = 1.2
```

实际 `sfreq`、带通范围、特征名称和模型对象必须来自训练过程，不能在部署端猜测。
现有 `ModelRuntime` 会检查实时特征名称与训练 artifact 是否一致。由于 FP1/FP2
不是典型 P300 最优位置，应以留一受试者验证、固定参赛者校准和现场混淆矩阵来设定
置信度及领先差阈值。

### Unitree 桥接配置

默认联调配置位于 `config/unitree_bridge.json`。为防止首次启动误动作，默认使用：

```json
{"transport": "stdout"}
```

真机推荐使用 `config/unitree_bridge_ros2.json`，其链路为：

```text
BioMulti Lite --LSL EEG--> P300 实时控制台
  --双向 Unix Socket--> bsense_unitree_bridge
  --/api/sport/request + CycloneDDS--> Unitree 机器狗
```

Python 与 ROS 2 Bridge 使用 `bsense.unitree.command.v1`。控制端发送：

```json
{
  "schema": "bsense.unitree.command.v1",
  "kind": "command",
  "command": "forward",
  "linear_x": 0.35,
  "angular_z": 0.0,
  "duration_s": 0.8,
  "source": "bsense_p300",
  "confidence": 0.81,
  "emergency": false,
  "timestamp_utc": "..."
}
```

ROS 2 Bridge 会响应：

```json
{
  "connected": true,
  "obstacle_detected": false,
  "emergency_stopped": false,
  "detail": "ok"
}
```

桥接节点默认订阅 Go2-W 的 `/lf/sportmodestate`，并读取其中的
`range_obstacle`；任一有效距离小于默认
`0.45 m` 时立即停止；状态超过 1 秒未更新、距离数据未知、Socket 断开、动作时长
到期或急停锁存时也会停止。Python 和 C++ 两侧都会限制动作时长，C++ 还会把线速度
限制到 `0.5 m/s`、角速度限制到 `1.0 rad/s`。急停后只有操作员再次点击
“安全检查并解锁”，且机器人状态与避障状态均正常时，Bridge 才会解除锁存。

HTTP 与 UDP 仍保留为其他桥接实现的可选适配器。HTTP `status_url` 也支持
`obstacle_clear` 作为 `obstacle_detected` 的反向字段。UDP 是单向模式，无法确认
避障状态，不建议用于比赛。

### Ubuntu / ROS 2 真机部署

需要 ROS 2 Humble、Unitree ROS 2 功能包以及可用的 `unitree_api`、`unitree_go`
消息：

```bash
cd /home/dd/bsense-realtime-classifier
source /opt/ros/humble/setup.bash
source /home/dd/unitree_ros2/setup.sh
colcon build --packages-select bsense_realtime_classifier
```

连接机器狗网卡后启动完整 P300 控制台：

```bash
export BCI_CYCLONEDDS_INTERFACE=enp3s0
./scripts/run_p300_unitree.sh
```

实际网卡名应通过 `ip -br link` 确认。可用环境变量覆盖默认路径：

- `BCI_P300_MODEL`：训练完成的 `m7_p300/model.joblib`；
- `BCI_UNITREE_SETUP`：Unitree ROS 2 的 `setup.sh`；
- `BCI_UNITREE_SOCKET`：双向 Unix Socket 路径；
- `BCI_SPORT_STATE_TOPIC`：默认使用 Go2-W 的 `/lf/sportmodestate`；
- `BCI_OBSTACLE_DISTANCE_M`：障碍停止距离，默认 `0.45`。

真机和比赛模式默认启用避障联锁，使用
`config/unitree_bridge_ros2.json`。如果联调设备确实没有避障传感器，可显式执行：

```bash
BCI_REQUIRE_OBSTACLE_CLEAR=false ./scripts/run_p300_unitree.sh
```

该命令会自动使用
`config/unitree_bridge_ros2_no_obstacle_sensor.json` 并显示安全警告。此模式会忽略
障碍物状态，只允许在清空场地、有人值守且急停可用的受控联调环境中使用，禁止用于
比赛。若通过 `BCI_BRIDGE_CONFIG` 指定其他配置，其中的
`require_obstacle_clear` 必须与 `BCI_REQUIRE_OBSTACLE_CLEAR` 一致，否则启动脚本会
拒绝运行。

Unix Socket 只能用于同一台 Ubuntu 主机，因此正式方案要求 P300 控制台和 ROS 2
Bridge 在同一台比赛电脑运行；脑电设备仍通过 LSL 发布数据。若解码器必须运行在
Windows 电脑上，应改用保留的 HTTP 传输，并在 Ubuntu 侧增加对应网关。

## 一键启动

确保发布 EEG 数据的软件已经启动，然后双击：

`run_classifier.bat`

本机脚本会优先使用现有的 Conda 环境：

`D:\ProgramData\miniforge3\envs\bci-gpu\python.exe`

也可以在 PowerShell 中运行：

```powershell
cd D:\codebase\BCI\bsense-realtime-classifier
.\run_classifier.ps1
```

界面中的使用顺序：

1. 在“内置任务”中选择模型。
2. EEG 流名称可留空；有多个 EEG 流时填写要连接的准确名称。
3. 启动负责发布 EEG 的软件。
4. 点击“开始接收并分析”。
5. 按界面左下角显示的任务步骤操作。
6. 观察分类、置信度和信号质量；结果会同时写入 `logs` 文件夹。

M1 和 M4A 默认开启“自动滚动分析”，每 4 秒产生一次候选结果，不需要反复
点击手动触发。Marker 或“立即对齐触发一次”用于让窗口与外部任务提示严格对齐。
M4B 是快速 ERP 事件任务，必须使用 Marker 或手动触发，不能改成普通连续分析。

## LSL 输入要求

EEG 流必须满足以下要求：

| 项目 | 要求 |
|---|---|
| LSL 类型 | `EEG` |
| 通道数 | 恰好 2 路 |
| 通道 | FP1、FP2 |
| 通道顺序 | 推荐 FP1、FP2；FP2、FP1 会自动换序 |
| 采样率 | 固定采样率；程序会插值到模型的 250 Hz |
| 数值单位 | 必须与训练数据保持一致，当前训练数据按设备原始幅值使用 |

如果 LSL 元数据没有通道名称，程序只能假设“第 1 路是 FP1、第 2 路是 FP2”，界面会显示警告。如果通道名称是其他位置，程序会停止，防止错误通道产生看似正常的结果。

事件 Marker 流默认名称为 `BSense Experiment Markers`。每个 Marker 样本可以是纯文本，也可以是以下 JSON：

```json
{"event": "target_highlight"}
```

JSON 中也支持 `marker` 或 `label` 字段。

## 六个任务怎么执行

| 内置任务 | 输出 | 运行方式 | 你需要做什么 |
|---|---|---|---|
| `m1_mi` | `idle` / `left_mi` / `right_mi` | 事件触发，4 秒 | 根据提示保持静息，或只想象左手/右手反复握拳，不要真的动手 |
| `m2_nback` | `0-back` / `1-back` / `2-back` | 连续，8 秒窗口、4 秒更新 | 在另一个软件中正常完成对应 N-back 任务 |
| `m3a_artifact` | `clean_baseline` / `motion_artifact` | 连续，4 秒更新 | 正常佩戴；它用于发现眨眼、点头、摇头或移动造成的污染 |
| `m3b_fatigue` | KSS 1–9 数值 | 连续，8 秒更新 | 持续执行任务，同时定期记录自己的真实 KSS 分数作校准 |
| `m4a_intent` | `intent_absent` / `intent_present` | 事件触发，4 秒 | 根据提示保持无操作意图，或明确形成准备操作目标的意图 |
| `m4b_target` | `non_target` / `target` | 事件触发，1.2 秒 | 注视目标；每次对象高亮时立即发送 `target_highlight` |

事件触发任务推荐发送这些 Marker：

- `m1_mi`：`mi_idle`、`mi_left`、`mi_right`
- `m4a_intent`：`intent_absent`、`intent_present`
- `m4b_target`：`target_highlight`

没有 Marker 流时，可以在刺激或状态开始的同一瞬间点击“手动触发一次”。`m4b_target` 对时序最敏感，正式采集时应使用 Marker，不建议依赖手动点击。

## 为什么会显示低置信度

`unknown` 不代表没有收到 EEG。它表示模型已经算出候选类别，但没有通过信号质量、
连续稳定性或置信度门槛。界面现在会大字显示候选类别，并在下方说明它仍属于
“未接纳结果”；JSON 继续使用 `state=-1`、`label=unknown`，同时保留
`candidate_label`。

可以在界面的“接纳阈值”中调整门槛。降低门槛只会减少 `unknown`，不会提高真实
准确率。M1 当前三人跨受试者 balanced accuracy 约为 0.401，接近三分类随机基线，
因此它经常输出接近 33%–45% 的概率是符合现有模型能力的。

## 当前是否使用 fNIRS

当前六个内置模型都只使用 FP1/FP2 EEG，没有把 fNIRS 或其他传感器直接拼入模型。
BioMulti Lite 和历史 XDF 还包含可用于后续模型的数据：

- 16 路、约 25 Hz 的原始 fNIRS 双波长数据；
- 6 路、约 12.5 Hz 的加速度和陀螺仪；
- Heart Rate；
- 2 路 Metric 与 13 路 General Metric。

Motion 最适合先用于运动伪迹门控；fNIRS 可用于 N-back 工作负荷和区块级疲劳/注意
状态；心率可作为疲劳辅助特征。fNIRS 当前是原始光强，必须在记录光路几何、DPF
和转换方法后计算 HbO/HbR，不能把 16 路原始数值直接当成血氧特征。厂商尚未公开
Metric/General Metric 各索引的完整语义，在确认映射前只适合记录和探索。

界面“输入模态”会区分“模型实际使用的数据”和“当前 LSL 能发现的流”。即使
BioMulti Lite 页面勾选了多项，如果 LSL 实际只发布/只发现 EEG，当前程序也不会
声称已经使用 fNIRS。

## 输出说明

默认结果保存在：

`logs/<任务>_<时间>.jsonl`

每一行是一条独立 JSON。分类模型的重要字段包括：

- `state`：数值类别；未通过检查时为 `-1`
- `label`：类别名称；未通过检查时为 `unknown`
- `candidate_label`：模型原始候选类别
- `confidence`：平滑后的置信度
- `probabilities`：各类别概率
- `signal_quality` / `quality_ok`：当前窗口的信号质量
- `accepted`：结果是否通过质量、置信度和稳定性检查

疲劳回归模型则输出 `value` 和 `raw_value`。

## 命令行运行

适合后台运行或给其他程序调用：

```powershell
$env:PYTHONPATH = "D:\codebase\BCI\bsense-realtime-classifier\src"
D:\ProgramData\miniforge3\envs\bci-gpu\python.exe -m bsense_classifier.cli `
  --model D:\codebase\BCI\bsense-realtime-classifier\models\m3a_artifact\model.joblib `
  --stream-name "你的 EEG 流名称" `
  --output D:\codebase\BCI\bsense-realtime-classifier\logs\artifact.jsonl
```

如果要在其他环境安装：

```powershell
python -m pip install -e .
```

随后可以使用 `bsense-classifier`（桌面界面）或 `bsense-classifier-cli`（命令行）。

## 模型适用范围

项目内的模型已经训练完成，并不是只改了分类头。它们使用本设备 FP1/FP2 数据训练，包括训练时一致的滤波、频谱/ERP 特征和分类器或回归器。

但目前跨人泛化证据有限。建议继续采集不同受试者，优先保证：

- 每个人都完成全部类别，且类别次数尽量平衡；
- 任务顺序随机或交叉，不要所有人都按同一个固定顺序完成；
- 记录受试者编号、任务事件、真实标签和 KSS；
- 设备位置、参考电极、增益、单位和采样率与当前训练数据一致；
- 每个任务至少做一次留一受试者验证，不能把同一人的相邻窗口同时分到训练集和测试集。

积累到更多不同受试者后，应重新训练群体模型；如果最终使用者固定，还可以再采集该使用者少量校准数据做个体微调或校准。
