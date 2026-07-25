"""Human-readable task instructions selected from the loaded model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskGuidance:
    title: str
    short_goal: str
    steps: tuple[str, ...]
    trigger_note: str
    limitation: str
    result_labels: tuple[str, ...]


GUIDANCE = {
    "m1_mi": TaskGuidance(
        title="运动想象：静息 / 左手 / 右手",
        short_goal="按照任务软件的提示，在 4 秒内保持静息或想象对应手部动作。",
        steps=(
            "佩戴设备并保持面部、眼睛和身体放松，不要实际抬手。",
            "“静息”：保持不动；“左手”：想象左手反复握拳；“右手”：想象右手反复握拳。",
            "每次状态持续 4 秒，等待界面给出结果后再进入下一次。",
        ),
        trigger_note="界面默认每 4 秒自动分析；使用 mi_idle / mi_left / mi_right Marker 或手动触发，可让分析窗口与任务开始时刻严格对齐。",
        limitation="FP1/FP2 不在运动皮层附近，左右手结果目前仅用于实验。",
        result_labels=("idle", "left_mi", "right_mi"),
    ),
    "m2_nback": TaskGuidance(
        title="工作负荷：0-back / 1-back / 2-back",
        short_goal="在另一个任务软件中执行 N-back，分类器每 4 秒更新一次负荷结果。",
        steps=(
            "启动你的 N-back 任务软件，并确认其规则是 0-back、1-back、2-back。",
            "正常完成按键任务，尽量减少说话、眨眼和头部动作。",
            "至少等待 8 秒形成第一个分析窗口，之后结果会连续更新。",
        ),
        trigger_note="这是连续模型，不需要手动触发；任务软件只负责呈现刺激。",
        limitation="现有训练数据采用固定递增顺序，结果可能混入时间和疲劳效应。",
        result_labels=("0-back", "1-back", "2-back"),
    ),
    "m3a_artifact": TaskGuidance(
        title="信号质量：干净 / 动作伪迹",
        short_goal="正常佩戴并尽量保持稳定，程序持续判断当前 EEG 是否被动作污染。",
        steps=(
            "静坐、放松额头和下颌，确认界面能稳定显示 clean_baseline。",
            "正常使用时无需刻意执行动作；出现眨眼、点头、摇头或走动时可能显示 motion_artifact。",
            "当结果为动作伪迹或信号质量异常时，不使用该窗口的认知分类结果。",
        ),
        trigger_note="这是连续质量门控，不需要 Marker 或手动触发。",
        limitation="动作标签是预期伪迹条件，不等同于人工确认的污染真值。",
        result_labels=("clean_baseline", "motion_artifact"),
    ),
    "m3b_fatigue": TaskGuidance(
        title="疲劳估计：KSS 数值",
        short_goal="保持正常任务状态，程序每 8 秒估计一次 1–9 的疲劳分数。",
        steps=(
            "连续佩戴设备并执行日常或认知任务，避免频繁调整设备。",
            "至少等待 8 秒出现首个结果；观察趋势，不关注单个瞬时值。",
            "定期记录真实 KSS 主观评分，作为后续个体校准标签。",
        ),
        trigger_note="这是连续回归模型，不需要 Marker 或手动触发。",
        limitation="当前独立疲劳标签很少，数值不能用于医疗、驾驶或安全判断。",
        result_labels=("KSS 1–9",),
    ),
    "m4a_intent": TaskGuidance(
        title="意图实验：无意图 / 有意图",
        short_goal="根据外部任务提示，在 4 秒窗口内保持无意图或明确形成操作意图。",
        steps=(
            "“无意图”：观看对象但不准备操作；“有意图”：明确想象准备操作该对象。",
            "保持身体不动，不要通过实际动作暴露答案。",
            "每次状态持续 4 秒，等待结果后再开始下一次。",
        ),
        trigger_note="界面默认每 4 秒自动分析；使用 intent_absent / intent_present Marker 或手动触发，可让分析窗口与任务开始时刻严格对齐。",
        limitation="模型只在外部提示条件下训练，不能证明自然场景的异步意图检测。",
        result_labels=("intent_absent", "intent_present"),
    ),
    "m4b_target": TaskGuidance(
        title="目标事件：非目标 / 目标",
        short_goal="由外部软件逐个高亮对象，程序分析高亮前 0.2 秒到高亮后 1 秒。",
        steps=(
            "注视任务软件指定的目标对象，尽量减少眨眼。",
            "外部软件每次高亮对象时发送 target_highlight Marker。",
            "没有 Marker 时，在高亮出现的同一瞬间点击“手动触发”。",
        ),
        trigger_note="必须事件触发；普通连续推理对这个模型没有意义。",
        limitation="FP1/FP2 不是典型 P300 顶区通道，当前结果仅用于探索。",
        result_labels=("non_target", "target"),
    ),
}


def guidance_for(task: str) -> TaskGuidance:
    return GUIDANCE.get(
        task,
        TaskGuidance(
            title=f"未知任务：{task}",
            short_goal="请查看模型训练说明后再运行。",
            steps=("确认模型任务定义、窗口长度和标签。",),
            trigger_note="程序无法为未知任务确定触发方式。",
            limitation="未知任务不会被视为已验证用途。",
            result_labels=(),
        ),
    )
