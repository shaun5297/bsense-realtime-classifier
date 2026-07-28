"""P300 flash scoring and FP1/FP2 LSL decoding for robot commands."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pylsl import StreamInlet, local_clock, resolve_byprop

from .lsl_engine import interpolate_buffer, stream_channel_labels
from .model_runtime import ArtifactError, ModelRuntime, plan_channels
from .robot_bridge import RobotCommand


P300_COMMANDS: tuple[RobotCommand, ...] = (
    "forward",
    "backward",
    "left",
    "right",
    "stop",
    "idle",
)
P300_LABELS: dict[RobotCommand, str] = {
    "forward": "前进",
    "backward": "后退",
    "left": "左转",
    "right": "右转",
    "stop": "急停",
    "idle": "待机",
}


@dataclass(frozen=True)
class P300Decision:
    trial_id: int
    command: RobotCommand
    accepted: bool
    confidence: float
    margin: float
    scores: dict[RobotCommand, float]
    flash_counts: dict[RobotCommand, int]
    quality_ratio: float
    rejection_reasons: tuple[str, ...]


@dataclass
class P300TrialAggregator:
    trial_id: int
    minimum_flashes_per_command: int = 6
    confidence_threshold: float = 0.55
    margin_threshold: float = 0.08
    minimum_quality_ratio: float = 0.8
    _scores: dict[RobotCommand, list[float]] = field(
        default_factory=lambda: {command: [] for command in P300_COMMANDS}
    )
    _quality: list[bool] = field(default_factory=list)

    def add(
        self,
        command: RobotCommand,
        target_probability: float,
        *,
        quality_ok: bool,
    ) -> None:
        if command not in self._scores:
            raise ValueError(f"未知 P300 指令：{command}")
        probability = float(target_probability)
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("目标概率必须是 0–1 之间的有限数。")
        self._quality.append(bool(quality_ok))
        if quality_ok:
            self._scores[command].append(probability)

    def finish(self) -> P300Decision:
        counts = {command: len(values) for command, values in self._scores.items()}
        scores = {
            command: float(np.mean(values)) if values else 0.0
            for command, values in self._scores.items()
        }
        ranked = sorted(P300_COMMANDS, key=scores.__getitem__, reverse=True)
        command = ranked[0]
        confidence = scores[command]
        margin = confidence - scores[ranked[1]]
        quality_ratio = (
            sum(self._quality) / len(self._quality) if self._quality else 0.0
        )
        reasons: list[str] = []
        if any(count < self.minimum_flashes_per_command for count in counts.values()):
            reasons.append("insufficient_flashes")
        if quality_ratio < self.minimum_quality_ratio:
            reasons.append("low_signal_quality_ratio")
        if confidence < self.confidence_threshold:
            reasons.append("below_confidence_threshold")
        if margin < self.margin_threshold:
            reasons.append("below_margin_threshold")
        return P300Decision(
            trial_id=self.trial_id,
            command=command,
            accepted=not reasons,
            confidence=confidence,
            margin=margin,
            scores=scores,
            flash_counts=counts,
            quality_ratio=quality_ratio,
            rejection_reasons=tuple(reasons),
        )


class P300TargetRuntime:
    """Validate a binary M7 artifact and expose its target probability."""

    TARGET_LABELS = frozenset({"target", "p300_target", "is_target"})

    def __init__(self, runtime: ModelRuntime) -> None:
        if runtime.task != "m7_p300":
            raise ArtifactError(
                f"P300 控制器要求 task=m7_p300，当前模型为 {runtime.task}。"
            )
        if runtime.target_kind != "classification":
            raise ArtifactError("m7_p300 必须是 target/non_target 二分类模型。")
        if len(runtime.classes) != 2:
            raise ArtifactError("m7_p300 模型必须恰好包含两个类别。")
        if not runtime.event_locked:
            raise ArtifactError("m7_p300 必须使用事件锁定的部署模式。")
        target_classes = [
            state
            for state, label in runtime.label_mapping.items()
            if label.strip().lower() in self.TARGET_LABELS
        ]
        if len(target_classes) == 1:
            target_class = target_classes[0]
        elif 1 in runtime.classes:
            target_class = 1
        else:
            raise ArtifactError(
                "无法识别 target 类；请将 label_mapping 中的目标类命名为 target。"
            )
        self.runtime = runtime
        self.target_class = target_class
        self.target_index = runtime.classes.index(target_class)

    @classmethod
    def load(cls, path: str) -> "P300TargetRuntime":
        return cls(ModelRuntime.load(path))

    @property
    def window_seconds(self) -> float:
        return self.runtime.window_seconds

    @property
    def window_offset_seconds(self) -> float:
        return self.runtime.window_offset_seconds

    @property
    def sfreq(self) -> float:
        return self.runtime.sfreq

    def infer_target_probability(
        self, raw_window: np.ndarray
    ) -> tuple[float, bool, str]:
        result = self.runtime.infer(raw_window)
        return (
            float(result.probabilities[self.target_index]),
            result.quality_ok,
            result.signal_quality,
        )


@dataclass(frozen=True)
class P300EngineConfig:
    stream_name: str = ""
    stream_type: str = "EEG"
    resolve_timeout: float = 15.0
    minimum_flashes_per_command: int = 6
    confidence_threshold: float = 0.55
    margin_threshold: float = 0.08
    minimum_quality_ratio: float = 0.8

    def validate(self) -> None:
        if self.resolve_timeout <= 0:
            raise ValueError("resolve_timeout 必须大于 0。")
        if self.minimum_flashes_per_command < 1:
            raise ValueError("minimum_flashes_per_command 必须至少为 1。")
        for name, value in (
            ("confidence_threshold", self.confidence_threshold),
            ("margin_threshold", self.margin_threshold),
            ("minimum_quality_ratio", self.minimum_quality_ratio),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} 必须在 0–1 范围内。")


@dataclass(frozen=True)
class FlashTrigger:
    trial_id: int
    sequence: int
    position: int
    command: RobotCommand
    timestamp: float


class P300ControlEngine:
    """Decode UI-generated P300 flashes against an incoming LSL EEG stream."""

    def __init__(
        self,
        runtime: P300TargetRuntime,
        config: P300EngineConfig,
    ) -> None:
        config.validate()
        self.runtime = runtime
        self.config = config
        self.events: queue.Queue[dict[str, Any]] = queue.Queue()
        self._requests: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="bsense-p300-control",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def submit_flash(
        self,
        *,
        trial_id: int,
        sequence: int,
        position: int,
        command: RobotCommand,
        timestamp: float | None = None,
    ) -> None:
        if command not in P300_COMMANDS:
            raise ValueError(f"未知 P300 指令：{command}")
        self._requests.put(
            (
                "flash",
                FlashTrigger(
                    trial_id,
                    sequence,
                    position,
                    command,
                    float(local_clock() if timestamp is None else timestamp),
                ),
            )
        )

    def finish_trial(self, trial_id: int) -> None:
        self._requests.put(("finish", int(trial_id)))

    def _emit(self, kind: str, **payload: Any) -> None:
        self.events.put({"kind": kind, **payload})

    def _new_aggregator(self, trial_id: int) -> P300TrialAggregator:
        return P300TrialAggregator(
            trial_id=trial_id,
            minimum_flashes_per_command=self.config.minimum_flashes_per_command,
            confidence_threshold=self.config.confidence_threshold,
            margin_threshold=self.config.margin_threshold,
            minimum_quality_ratio=self.config.minimum_quality_ratio,
        )

    def _resolve_stream(self) -> Any:
        property_name = "name" if self.config.stream_name else "type"
        property_value = self.config.stream_name or self.config.stream_type
        deadline = time.monotonic() + self.config.resolve_timeout
        self._emit(
            "status",
            message=f"正在查找 LSL EEG 流：{property_name}={property_value}",
        )
        while not self._stop_event.is_set() and time.monotonic() < deadline:
            streams = resolve_byprop(property_name, property_value, timeout=1.0)
            if streams:
                return streams[0]
        raise TimeoutError(f"在 {self.config.resolve_timeout:g} 秒内没有找到 EEG 流。")

    def _drain_requests(
        self,
        pending: deque[FlashTrigger],
        finish_requested: set[int],
    ) -> None:
        while True:
            try:
                kind, value = self._requests.get_nowait()
            except queue.Empty:
                return
            if kind == "flash":
                pending.append(value)
            elif kind == "finish":
                finish_requested.add(value)

    @staticmethod
    def _pending_trial(pending: deque[FlashTrigger], trial_id: int) -> bool:
        return any(trigger.trial_id == trial_id for trigger in pending)

    def _finalize_ready(
        self,
        pending: deque[FlashTrigger],
        finish_requested: set[int],
        aggregators: dict[int, P300TrialAggregator],
    ) -> None:
        for trial_id in list(finish_requested):
            if self._pending_trial(pending, trial_id):
                continue
            aggregator = aggregators.pop(trial_id, self._new_aggregator(trial_id))
            finish_requested.remove(trial_id)
            self._emit("decision", decision=aggregator.finish())

    def _run(self) -> None:
        eeg_inlet: StreamInlet | None = None
        try:
            info = self._resolve_stream()
            eeg_inlet = StreamInlet(
                info,
                max_buflen=max(30, int(self.runtime.window_seconds * 8)),
            )
            full_info = eeg_inlet.info(timeout=2.0)
            input_sfreq = float(full_info.nominal_srate())
            if input_sfreq <= 0:
                raise ArtifactError("EEG 流必须提供固定采样率。")
            channel_plan = plan_channels(
                stream_channel_labels(full_info),
                int(full_info.channel_count()),
            )
            self._emit(
                "connected",
                stream_name=full_info.name(),
                input_sfreq=input_sfreq,
                model_sfreq=self.runtime.sfreq,
                channel_status=channel_plan.status,
                channel_warning=channel_plan.warning,
            )
            timestamps: deque[float] = deque()
            samples: deque[list[float]] = deque()
            pending: deque[FlashTrigger] = deque()
            finish_requested: set[int] = set()
            aggregators: dict[int, P300TrialAggregator] = {}
            while not self._stop_event.is_set():
                chunk, chunk_timestamps = eeg_inlet.pull_chunk(
                    timeout=0.05,
                    max_samples=max(1, int(round(input_sfreq / 4))),
                )
                if chunk_timestamps:
                    timestamps.extend(float(value) for value in chunk_timestamps)
                    samples.extend([float(value) for value in row] for row in chunk)
                    keep_after = float(chunk_timestamps[-1]) - 30.0
                    while timestamps and timestamps[0] < keep_after:
                        timestamps.popleft()
                        samples.popleft()
                self._drain_requests(pending, finish_requested)
                if timestamps:
                    latest = timestamps[-1]
                    while pending:
                        trigger = pending[0]
                        window_start = (
                            trigger.timestamp + self.runtime.window_offset_seconds
                        )
                        window_end = window_start + self.runtime.window_seconds
                        if latest < window_end:
                            break
                        pending.popleft()
                        raw = interpolate_buffer(
                            list(timestamps),
                            list(samples),
                            window_start,
                            self.runtime.window_seconds,
                            self.runtime.sfreq,
                            channel_plan.indices,
                        )
                        if raw is None:
                            self._emit(
                                "flash_skipped",
                                trial_id=trigger.trial_id,
                                command=trigger.command,
                                reason="incomplete_eeg_window",
                            )
                            continue
                        probability, quality_ok, quality_reason = (
                            self.runtime.infer_target_probability(raw)
                        )
                        aggregator = aggregators.setdefault(
                            trigger.trial_id,
                            self._new_aggregator(trigger.trial_id),
                        )
                        aggregator.add(
                            trigger.command,
                            probability,
                            quality_ok=quality_ok,
                        )
                        self._emit(
                            "flash_result",
                            trial_id=trigger.trial_id,
                            sequence=trigger.sequence,
                            command=trigger.command,
                            target_probability=probability,
                            quality_ok=quality_ok,
                            signal_quality=quality_reason,
                        )
                self._finalize_ready(
                    pending,
                    finish_requested,
                    aggregators,
                )
        except Exception as exc:
            if not self._stop_event.is_set():
                self._emit("error", message=f"{type(exc).__name__}: {exc}")
        finally:
            if eeg_inlet is not None:
                try:
                    eeg_inlet.close_stream()
                except Exception:
                    pass
            self._emit("stopped")
