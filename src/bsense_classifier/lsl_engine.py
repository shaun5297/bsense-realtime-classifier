"""Threaded LSL acquisition and real-time inference engine."""

from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pylsl import StreamInlet, resolve_byprop, resolve_streams

from .model_runtime import ArtifactError, InferenceResult, ModelRuntime, plan_channels


TASK_MARKERS = {
    "m1_mi": {"mi_idle", "mi_left", "mi_right"},
    "m4a_intent": {"intent_absent", "intent_present"},
    "m4b_target": {"target_highlight"},
}


@dataclass(frozen=True)
class EngineConfig:
    stream_name: str = ""
    stream_type: str = "EEG"
    marker_stream_name: str = "BSense Experiment Markers"
    resolve_timeout: float = 15.0
    confidence_threshold: float | None = None
    smoothing_windows: int = 3
    confirmation_windows: int = 2
    use_markers: bool = True
    auto_analyze_event_tasks: bool = True
    output_path: Path | None = None


@dataclass(frozen=True)
class PendingTrigger:
    event: str
    event_timestamp: float
    window_start: float
    source: str


def stream_channel_labels(info: Any) -> list[str]:
    """Read channel labels from a pylsl StreamInfo description."""

    labels: list[str] = []
    try:
        channel = info.desc().child("channels").child("channel")
        for _ in range(int(info.channel_count())):
            labels.append(str(channel.child_value("label") or "").strip())
            channel = channel.next_sibling("channel")
    except Exception:
        return []
    return labels if any(labels) else []


def interpolate_buffer(
    timestamps: list[float] | np.ndarray,
    samples: list[list[float]] | np.ndarray,
    start: float,
    duration: float,
    sfreq: float,
    channel_indices: tuple[int, int],
) -> np.ndarray | None:
    """Interpolate irregular LSL samples onto the model's fixed sample grid."""

    ts = np.asarray(timestamps, dtype=np.float64)
    values = np.asarray(samples, dtype=np.float64)
    if ts.ndim != 1 or values.ndim != 2 or len(ts) != len(values):
        return None
    count = int(round(duration * sfreq))
    if count < 2 or len(ts) < 2:
        return None
    end = start + duration
    tolerance = max(2.0 / sfreq, 0.02)
    if ts.min() > start + tolerance or ts.max() < end - tolerance:
        return None
    order = np.argsort(ts, kind="stable")
    ts = ts[order]
    values = values[order]
    unique_mask = np.concatenate(([True], np.diff(ts) > 1e-9))
    ts = ts[unique_mask]
    values = values[unique_mask]
    if len(ts) < 2:
        return None
    grid = start + np.arange(count, dtype=np.float64) / sfreq
    selected = values[:, list(channel_indices)]
    return np.vstack(
        [np.interp(grid, ts, selected[:, channel]) for channel in range(2)]
    )


def result_record(
    result: InferenceResult,
    *,
    timestamp: float,
    window_start: float,
    window_end: float,
    accepted: bool,
    state: int | None,
    label: str | None,
    confidence: float | None,
    probabilities: tuple[float, ...] = (),
    value: float | None = None,
    trigger: PendingTrigger | None = None,
    confidence_threshold: float | None = None,
    rejection_reasons: tuple[str, ...] = (),
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "protocol": "bci_result_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "lsl_timestamp": timestamp,
        "task": result.task,
        "target_kind": result.target_kind,
        "window_start": window_start,
        "window_end": window_end,
        "signal_quality": result.signal_quality,
        "quality_ok": result.quality_ok,
        "accepted": accepted,
    }
    if result.target_kind == "classification":
        record.update(
            {
                "state": state if accepted else -1,
                "label": label if accepted else "unknown",
                "candidate": result.candidate,
                "candidate_label": result.label,
                "confidence": confidence,
                "confidence_threshold": confidence_threshold,
                "classes": list(result.classes),
                "probabilities": list(probabilities),
                "rejection_reasons": list(rejection_reasons),
            }
        )
    else:
        record.update(
            {
                "target_name": result.target_name,
                "value": value,
                "raw_value": result.raw_value,
            }
        )
    if trigger is not None:
        record["trigger"] = asdict(trigger)
    return record


class RealtimeEngine:
    """Receive FP1/FP2 LSL data and emit inference events on a worker thread."""

    def __init__(self, runtime: ModelRuntime, config: EngineConfig) -> None:
        self.runtime = runtime
        self.config = config
        self.events: queue.Queue[dict[str, Any]] = queue.Queue()
        self._manual_requests: queue.Queue[str] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_lock = threading.Lock()
        self._latest_lsl_timestamp: float | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="bsense-lsl-inference",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def manual_trigger(self, event: str = "manual_trigger") -> bool:
        if not self.runtime.event_locked:
            self._emit(
                "status",
                message="该模型为连续自动分析，无需手动触发。",
            )
            return False
        with self._latest_lock:
            ready = self._latest_lsl_timestamp is not None
        if not ready:
            self._emit("status", message="尚未收到 EEG 数据，暂时不能触发。")
            return False
        self._manual_requests.put(event)
        return True

    def _emit(self, kind: str, **payload: Any) -> None:
        self.events.put({"kind": kind, **payload})

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

    def _try_marker_inlet(self, *, announce_missing: bool = True) -> StreamInlet | None:
        if not self.config.use_markers or not self.runtime.event_locked:
            return None
        streams = resolve_byprop(
            "name",
            self.config.marker_stream_name,
            timeout=0.5,
        )
        if not streams:
            if announce_missing:
                self._emit(
                    "status",
                    message="未找到实验 Marker 流；M1/M4A 可自动分析，也可手动触发。",
                )
            return None
        self._emit(
            "status",
            message=f"已连接 Marker 流：{self.config.marker_stream_name}",
        )
        return StreamInlet(streams[0], max_buflen=60)

    @staticmethod
    def _marker_name(sample: list[Any]) -> str:
        if not sample:
            return ""
        raw = str(sample[0]).strip()
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw
        if isinstance(parsed, dict):
            return str(
                parsed.get("event")
                or parsed.get("marker")
                or parsed.get("label")
                or raw
            )
        return raw

    def _drain_triggers(
        self,
        marker_inlet: StreamInlet | None,
        pending: deque[PendingTrigger],
    ) -> None:
        while True:
            try:
                event = self._manual_requests.get_nowait()
            except queue.Empty:
                break
            with self._latest_lock:
                timestamp = self._latest_lsl_timestamp
            if timestamp is None:
                continue
            pending.append(
                PendingTrigger(
                    event=event,
                    event_timestamp=timestamp,
                    window_start=timestamp + self.runtime.window_offset_seconds,
                    source="manual",
                )
            )
            self._emit("status", message="已手动触发，正在收集分析窗口。")
        if marker_inlet is None:
            return
        allowed = TASK_MARKERS.get(self.runtime.task, set())
        while True:
            sample, timestamp = marker_inlet.pull_sample(timeout=0.0)
            if timestamp is None:
                break
            marker = self._marker_name(sample)
            if allowed and marker not in allowed:
                continue
            pending.append(
                PendingTrigger(
                    event=marker,
                    event_timestamp=float(timestamp),
                    window_start=float(timestamp)
                    + self.runtime.window_offset_seconds,
                    source="lsl_marker",
                )
            )
            self._emit("status", message=f"收到事件：{marker}，正在分析。")

    def _classification_decision(
        self,
        result: InferenceResult,
        probability_history: deque[np.ndarray],
        candidate_history: deque[int],
        *,
        triggered: bool,
    ) -> tuple[
        bool,
        int,
        str,
        float,
        tuple[float, ...],
        float,
        tuple[str, ...],
    ]:
        probability_history.append(np.asarray(result.probabilities, dtype=float))
        averaged = np.mean(np.vstack(probability_history), axis=0)
        index = int(np.argmax(averaged))
        state = result.classes[index]
        confidence = float(averaged[index])
        candidate_history.append(state)
        confirmation = 1 if triggered else max(1, self.config.confirmation_windows)
        stable = (
            len(candidate_history) >= confirmation
            and len(set(list(candidate_history)[-confirmation:])) == 1
        )
        threshold = (
            self.runtime.default_threshold
            if self.config.confidence_threshold is None
            else self.config.confidence_threshold
        )
        accepted = result.quality_ok and stable and confidence >= threshold
        reasons: list[str] = []
        if not result.quality_ok:
            reasons.append(f"signal_quality:{result.signal_quality}")
        if not stable:
            reasons.append("waiting_for_stability")
        if confidence < threshold:
            reasons.append("below_confidence_threshold")
        return (
            accepted,
            state,
            self.runtime.label_mapping.get(state, str(state)),
            confidence,
            tuple(float(item) for item in averaged),
            threshold,
            tuple(reasons),
        )

    def _write_record(self, record: dict[str, Any]) -> None:
        if self.config.output_path is None:
            return
        output = Path(self.config.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _infer_window(
        self,
        raw_window: np.ndarray,
        window_start: float,
        window_end: float,
        probability_history: deque[np.ndarray],
        candidate_history: deque[int],
        value_history: deque[float],
        trigger: PendingTrigger | None = None,
    ) -> None:
        result = self.runtime.infer(raw_window)
        if result.target_kind == "classification":
            (
                accepted,
                state,
                label,
                confidence,
                probabilities,
                threshold,
                rejection_reasons,
            ) = (
                self._classification_decision(
                    result,
                    probability_history,
                    candidate_history,
                    triggered=trigger is not None,
                )
            )
            result = InferenceResult(
                **{
                    **asdict(result),
                    "candidate": state,
                    "label": label,
                    "confidence": confidence,
                    "probabilities": probabilities,
                }
            )
            record = result_record(
                result,
                timestamp=window_end,
                window_start=window_start,
                window_end=window_end,
                accepted=accepted,
                state=state,
                label=label,
                confidence=confidence,
                probabilities=probabilities,
                trigger=trigger,
                confidence_threshold=threshold,
                rejection_reasons=rejection_reasons,
            )
        else:
            assert result.value is not None
            value_history.append(result.value)
            value = float(np.mean(value_history))
            record = result_record(
                result,
                timestamp=window_end,
                window_start=window_start,
                window_end=window_end,
                accepted=result.quality_ok,
                state=None,
                label=None,
                confidence=None,
                value=value,
                trigger=trigger,
            )
        self._write_record(record)
        self._emit("result", record=record)

    def _run(self) -> None:
        eeg_inlet: StreamInlet | None = None
        try:
            info = self._resolve_stream()
            eeg_inlet = StreamInlet(
                info,
                max_buflen=max(30, int(self.runtime.window_seconds * 5)),
            )
            full_info = eeg_inlet.info(timeout=2.0)
            sfreq = float(full_info.nominal_srate())
            channel_count = int(full_info.channel_count())
            if sfreq <= 0:
                raise ArtifactError("EEG 流必须提供固定采样率。")
            labels = stream_channel_labels(full_info)
            channel_plan = plan_channels(labels, channel_count)
            marker_inlet = self._try_marker_inlet()
            auto_event_analysis = (
                self.config.auto_analyze_event_tasks
                and self.runtime.task in {"m1_mi", "m4a_intent"}
            )
            available_stream_types = sorted(
                {
                    str(stream.type()).strip()
                    for stream in resolve_streams(wait_time=0.25)
                    if str(stream.type()).strip()
                }
            )
            self._emit(
                "connected",
                stream_name=full_info.name(),
                source_id=full_info.source_id(),
                input_sfreq=sfreq,
                model_sfreq=self.runtime.sfreq,
                channel_status=channel_plan.status,
                channel_warning=channel_plan.warning,
                channel_labels=list(channel_plan.labels),
                model_modalities=["EEG"],
                available_stream_types=available_stream_types,
                auto_event_analysis=auto_event_analysis,
            )
            timestamps: deque[float] = deque()
            samples: deque[list[float]] = deque()
            pending: deque[PendingTrigger] = deque()
            probability_history: deque[np.ndarray] = deque(
                maxlen=max(1, self.config.smoothing_windows)
                if auto_event_analysis or not self.runtime.event_locked
                else 1
            )
            candidate_history: deque[int] = deque(
                maxlen=max(1, self.config.confirmation_windows)
            )
            value_history: deque[float] = deque(
                maxlen=max(1, self.config.smoothing_windows)
            )
            last_continuous_end: float | None = None
            retry_marker_at = time.monotonic() + 5.0
            while not self._stop_event.is_set():
                chunk, chunk_timestamps = eeg_inlet.pull_chunk(
                    timeout=0.2,
                    max_samples=max(1, int(round(sfreq))),
                )
                if chunk_timestamps:
                    timestamps.extend(float(value) for value in chunk_timestamps)
                    samples.extend([float(value) for value in row] for row in chunk)
                    latest = float(chunk_timestamps[-1])
                    with self._latest_lock:
                        self._latest_lsl_timestamp = latest
                    keep_after = latest - max(
                        30.0,
                        self.runtime.window_seconds * 4
                        + abs(self.runtime.window_offset_seconds),
                    )
                    while timestamps and timestamps[0] < keep_after:
                        timestamps.popleft()
                        samples.popleft()
                self._drain_triggers(marker_inlet, pending)
                if (
                    marker_inlet is None
                    and self.config.use_markers
                    and self.runtime.event_locked
                    and time.monotonic() >= retry_marker_at
                ):
                    marker_inlet = self._try_marker_inlet(announce_missing=False)
                    retry_marker_at = time.monotonic() + 5.0
                if not timestamps:
                    continue
                latest = timestamps[-1]
                if self.runtime.event_locked:
                    while pending:
                        trigger = pending[0]
                        window_end = (
                            trigger.window_start + self.runtime.window_seconds
                        )
                        if latest < window_end:
                            break
                        pending.popleft()
                        raw = interpolate_buffer(
                            list(timestamps),
                            list(samples),
                            trigger.window_start,
                            self.runtime.window_seconds,
                            self.runtime.sfreq,
                            channel_plan.indices,
                        )
                        if raw is None:
                            self._emit(
                                "status",
                                message=f"事件 {trigger.event} 的 EEG 窗口不完整，已跳过。",
                            )
                            continue
                        probability_history.clear()
                        candidate_history.clear()
                        self._infer_window(
                            raw,
                            trigger.window_start,
                            window_end,
                            probability_history,
                            candidate_history,
                            value_history,
                            trigger,
                        )
                    if not auto_event_analysis:
                        continue
                if last_continuous_end is None:
                    last_continuous_end = latest
                if (
                    latest - last_continuous_end
                    < self.runtime.stride_seconds
                ):
                    continue
                window_end = latest
                window_start = window_end - self.runtime.window_seconds
                raw = interpolate_buffer(
                    list(timestamps),
                    list(samples),
                    window_start,
                    self.runtime.window_seconds,
                    self.runtime.sfreq,
                    channel_plan.indices,
                )
                last_continuous_end = window_end
                if raw is None:
                    continue
                self._infer_window(
                    raw,
                    window_start,
                    window_end,
                    probability_history,
                    candidate_history,
                    value_history,
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
