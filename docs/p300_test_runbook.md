# P300 脑控机器狗 测试运行手册

目标：**加载训练好的模型 → 跑 P300 选择 → 下发指令给机械狗。**

先读第 0 节，再按场景 A（台架，无设备）或场景 B（真机）执行。

---

## 0. 必须先知道的事

已安装的模型 `models/m7_p300/model.joblib` 判别力**接近随机**：

| 指标 | 实测 | 随机水平 |
|---|---:|---:|
| 单闪烁 AUC | 0.524 | 0.500 |
| 试次 6 选 1 正确率 | 0.229 | 0.167 |
| **默认门槛下的接受率** | **约 0.2 %** | — |

后果很直接：

- **用默认门槛跑，500 次选择里大约只有 1 次会通过门控**，机械狗基本不会动。
  这是门控在正确工作，不是 bug。
- **把门槛调低能让指令流动起来，但那些指令来自噪声**，正确率约等于掷骰子。

因此：**台架联排可以跑、可以看指令流；真机上不要让它自主移动。**
真出成绩的前提是回采集端换顶区电极（Pz/Cz）——采集软件自己在 marker 里写着
`device_channel_limit=fp1_fp2_not_p300_optimal`。

---

## 1. 前提检查

```bash
cd /Users/shaun/BCI/bsense-realtime-classifier
PY=/Users/shaun/BCI/bsense-p300-pilot/.venv/bin/python

# 模型在不在、契约对不对
$PY -c "
import sys; sys.path.insert(0,'src')
from bsense_classifier.p300_control import P300TargetRuntime
r = P300TargetRuntime.load('models/m7_p300/model.joblib')
print('task=%s classes=%s target_class=%s 窗口 %s..%s s' % (
    r.runtime.task, r.runtime.classes, r.target_class,
    r.window_offset_seconds, r.window_offset_seconds + r.window_seconds))
"
```

期望输出：`task=m7_p300 classes=(0, 1) target_class=1 窗口 -0.2..1.0 s`

```bash
# 测试套件（唯一失败项是 macOS socket 路径长度，与模型无关）
$PY -m pytest tests/ -q
```

---

## 2. 场景 A：台架联排（无设备，验证整条链路）

用真实录制的 XDF 回放成实时 LSL 流代替设备。**两个终端。**

**终端 1 —— 回放 EEG**

```bash
cd /Users/shaun/BCI/bsense-realtime-classifier
PY=/Users/shaun/BCI/bsense-p300-pilot/.venv/bin/python
$PY tools/replay_xdf_eeg_lsl.py \
  --xdf "/Users/shaun/BCI/data/p300/sub-P001/ses-20260915_155436/sub-P001_ses-20260915_155436_task-m7_p300.xdf" \
  --stream-name "BSense Bench EEG" --speed 1.0
```

看到 `LSL EEG ready: name=BSense Bench EEG` 即就绪。`--speed 1.0` 是实时；
调大只影响回放快慢，会改变闪烁与 EEG 的对齐，联排用 1.0。

**终端 2 —— 控制台**

```bash
cd /Users/shaun/BCI/bsense-realtime-classifier
PY=/Users/shaun/BCI/bsense-p300-pilot/.venv/bin/python
$PY -m bsense_classifier.robot_app \
  --model "models/m7_p300/model.joblib" \
  --stream-name "BSense Bench EEG" \
  --sequences-per-trial 6 \
  --confidence-threshold 0.50 \
  --margin-threshold 0.0
```

界面里依次：**加载模型 → 连接 EEG → 安全检查并解锁 → 开始选择**。

- 默认 `config/unitree_bridge.json` 的 `transport` 是 `stdout`，
  **只往终端打印 JSON，不会移动任何真实设备** —— 台架联排就该用这个。
- 连接成功后日志会打印生效的门控门槛，调低过会额外打一条警告。
- 前 20 次选择可参考的实测：**720/720 次闪烁完成打分、20/20 次试次产出决策、
  20/20 被接受**（`--confidence-threshold 0.50 --margin-threshold 0.0` 下）。

**只看流程、不想开界面**（更快的自检）：

```bash
cd /Users/shaun/BCI/p300-retrain
PYTHONPATH=src /Users/shaun/BCI/bsense-p300-pilot/.venv/bin/python -m p300_retrain.e2e \
  --xdf "/Users/shaun/BCI/data/p300/sub-P001/ses-20260915_155436/sub-P001_ses-20260915_155436_task-m7_p300.xdf" \
  --model "/Users/shaun/BCI/bsense-realtime-classifier/models/m7_p300/model.joblib" \
  --trials 20 --sequences 6 --speed 6.0 \
  --confidence-threshold 0.50 --margin-threshold 0.0
```

---

## 3. 场景 B：真机

**先只跑到"连接 EEG + 解码"，确认解码正常，再接机器狗。**

### 3.1 接真实设备

1. 启动设备厂商软件，确认它发布 FP1/FP2 双通道 EEG 流。
2. 查流名和通道标签：

```bash
PY=/Users/shaun/BCI/bsense-p300-pilot/.venv/bin/python
$PY -c "
from pylsl import resolve_byprop, StreamInlet
found = resolve_byprop('type', 'EEG', timeout=5.0)
print('找到 %d 路 EEG 流' % len(found))
for short in found:
    # 必须开 inlet 再取 info()：resolve_byprop 返回的是 shortinfo，
    # 通道描述（labels）要等 inlet.info() 才完整。控制台内部就是这么做的。
    info = StreamInlet(short).info(timeout=2.0)
    labels = []
    node = info.desc().child('channels').child('channel')
    for _ in range(int(info.channel_count())):
        labels.append(node.child_value('label'))
        node = node.next_sibling('channel')
    print('  名称=%r  通道数=%d  采样率=%g  标签=%s'
          % (info.name(), info.channel_count(), info.nominal_srate(), labels))
"
```

期望：通道数为 2、标签为 `['FP1','FP2']` 或 `['FP2','FP1']`（后者程序会自动换序）。
若标签为 `['','']`，程序会显示"LSL 没有通道标签"，按第 1 通道=FP1、第 2 通道=FP2
处理 —— **此时必须自己确认设备通道顺序**。

**把 `名称=` 后面的字符串原样传给 `--stream-name`。**

3. 控制台加 `--stream-name "<上面打印的名字>"`。**务必显式指定**——场地上
   `resolve_byprop('type','EEG')` 可能找到多路流。

### 3.2 先不接机器狗跑一轮

```bash
$PY -m bsense_classifier.robot_app \
  --model "models/m7_p300/model.joblib" \
  --stream-name "<设备流名>" \
  --sequences-per-trial 6
```

**先用默认门槛**（不加 threshold 参数）。观察日志里的
`门控门槛：置信度 ≥ 0.55、领先差 ≥ 0.08、质量比例 ≥ 0.8`。
如果接受率极低（预期如此），说明门控工作正常。

### 3.3 接机器狗

只在你已经接受"指令来自噪声"这一前提下：

```bash
./scripts/run_p300_unitree.sh          # 内部用 unitree_bridge_ros2.json
```

或手动：

```bash
$PY -m bsense_classifier.robot_app \
  --model "models/m7_p300/model.joblib" \
  --stream-name "<设备流名>" \
  --sequences-per-trial 6 \
  --bridge-config "config/unitree_bridge_ros2.json"
```

安全联锁（不要绕过）：`require_obstacle_clear: true`、必须显式"安全检查并解锁"、
每个移动指令只持续 `motion_duration_seconds=0.8` 秒随后自动 `stop`、
空格/Esc/界面急停按钮立即归零并锁定。

---

## 4. 参数速查

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model` | `models/m7_p300/model.joblib` | 留空则自动找默认路径 |
| `--stream-name` | 空 | 空则按 type=EEG 解析，**多流环境下务必指定** |
| `--sequences-per-trial` | 10 | 2–10；每次选择的闪烁轮数 |
| `--confidence-threshold` | 0.55 | 调低会放宽接受率，**只用于台架** |
| `--margin-threshold` | 0.08 | 第一名与第二名概率差，同上 |
| `--min-quality-ratio` | 0.8 | 合格 EEG 窗口比例 |
| `--bridge-config` | `config/unitree_bridge.json` | `stdout` 是安全的台架传输 |
| `--socket-path` | 空 | 覆盖 ROS2 桥接的 Unix Socket 路径 |

每指令有效闪烁要求 = `min(6, sequences_per_trial)`：2 轮→2 次、6 轮→6 次、10 轮→6 次。

---

## 5. 故障排查

| 现象 | 原因 / 处理 |
|---|---|
| 界面提示"请先加载训练完成的 task=m7_p300 二分类模型" | 模型路径不对，或模型 `task` 字段不是 `m7_p300` |
| 日志报"检测到 FP2、FP1 顺序，已自动交换" | 正常，程序已自动换序 |
| 日志报"LSL 没有通道标签" | 程序按第 1 通道=FP1、第 2 通道=FP2 处理；确认设备通道顺序 |
| 15 秒内找不到 EEG 流 | 检查设备软件是否在发布；用 3.1 的脚本确认流名 |
| 大量 `flash_skipped / incomplete_eeg_window` | EEG 断流或采样率不匹配；查信号质量显示 |
| 所有试次都被拒绝 | **预期行为**（见第 0 节）。看日志里的 `rejection_reasons` |
| `AF_UNIX path too long`（测试套件） | macOS 临时路径过长，与模型无关；Linux 上通过 |

---

## 6. 改用其他模型

```bash
# P001 个体模型（判别力略好一点：AUC 0.514 / 试次 0.237）
--model "models/m7_p300/model_P001_personal.joblib"

# 新旧混合对照模型（更差：AUC 0.512 / 试次 0.119）
--model "models/m7_p300/model_mixed_cohorts.joblib"
```

三个模型都符合同一份 artifact 契约，可随时互换。
