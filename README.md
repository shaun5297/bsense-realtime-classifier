# BSense 实时脑电分类器

这是一个独立运行的 FP1/FP2 双通道 EEG 实时分类项目。它可以：

- 加载项目内置的 6 个 BSense 模型；
- 自动查找或按名称连接其他软件发布的 LSL EEG 流；
- 校验 FP1、FP2 通道，检测到 FP2/FP1 倒序时自动换序；
- 对连续任务定时推理，对事件任务接收 Marker 或手动触发；
- 在桌面界面显示分类、置信度、各类别概率和信号质量；
- 将每次结果保存为 JSONL，便于后续统计和接入其他软件。

> 当前模型属于实验模型，训练对象数量仍少。程序只输出分析结果，不会直接控制设备，也不能用于医疗、安全或驾驶判断。

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
