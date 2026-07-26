from __future__ import annotations

from collections import deque

import numpy as np

from bsense_classifier.lsl_engine import (
    EngineConfig,
    PendingTrigger,
    RealtimeEngine,
    interpolate_buffer,
    result_record,
)
from bsense_classifier.model_runtime import InferenceResult


class _StubRuntime:
    def __init__(self, event_locked: bool) -> None:
        self.task = "m2_nback"
        self.event_locked = event_locked
        self.window_seconds = 4.0
        self.window_offset_seconds = 0.0
        self.stride_seconds = 1.0
        self.sfreq = 250.0
        self.default_threshold = 0.5
        self.label_mapping = {}


def _engine_with_data(event_locked: bool) -> RealtimeEngine:
    engine = RealtimeEngine(_StubRuntime(event_locked), EngineConfig())
    engine._latest_lsl_timestamp = 100.0
    return engine


def test_manual_trigger_rejected_for_continuous_models() -> None:
    engine = _engine_with_data(event_locked=False)
    assert engine.manual_trigger() is False
    assert engine._manual_requests.empty()
    pending: deque[PendingTrigger] = deque()
    engine._drain_triggers(None, pending)
    assert not pending
    status = engine.events.get_nowait()
    assert status["kind"] == "status"
    assert "连续自动分析" in status["message"]


def test_manual_trigger_accepted_for_event_locked_models() -> None:
    engine = _engine_with_data(event_locked=True)
    assert engine.manual_trigger() is True
    pending: deque[PendingTrigger] = deque()
    engine._drain_triggers(None, pending)
    assert len(pending) == 1
    assert pending[0].source == "manual"
    assert pending[0].window_start == 100.0


def test_interpolate_buffer_reorders_channels() -> None:
    sfreq = 250.0
    timestamps = np.arange(0.0, 1.01, 1.0 / sfreq)
    samples = np.column_stack((np.ones(len(timestamps)), 2 * np.ones(len(timestamps))))
    window = interpolate_buffer(
        timestamps,
        samples,
        start=0.0,
        duration=1.0,
        sfreq=sfreq,
        channel_indices=(1, 0),
    )
    assert window is not None
    assert window.shape == (2, 250)
    assert np.allclose(window[0], 2.0)
    assert np.allclose(window[1], 1.0)


def test_interpolate_buffer_rejects_incomplete_window() -> None:
    timestamps = np.arange(0.2, 0.8, 0.01)
    samples = np.zeros((len(timestamps), 2))
    assert (
        interpolate_buffer(timestamps, samples, 0.0, 1.0, 100.0, (0, 1))
        is None
    )


def test_rejected_classification_is_explicit_unknown() -> None:
    result = InferenceResult(
        task="m3a_artifact",
        target_kind="classification",
        signal_quality="ok",
        quality_ok=True,
        candidate=1,
        label="motion_artifact",
        confidence=0.51,
        probabilities=(0.49, 0.51),
        classes=(0, 1),
    )
    trigger = PendingTrigger("manual_trigger", 10.0, 10.0, "manual")
    record = result_record(
        result,
        timestamp=14.0,
        window_start=10.0,
        window_end=14.0,
        accepted=False,
        state=1,
        label="motion_artifact",
        confidence=0.51,
        probabilities=(0.49, 0.51),
        trigger=trigger,
        confidence_threshold=0.60,
        rejection_reasons=("below_confidence_threshold",),
    )
    assert record["state"] == -1
    assert record["label"] == "unknown"
    assert record["candidate_label"] == "motion_artifact"
    assert record["confidence_threshold"] == 0.60
    assert record["rejection_reasons"] == ["below_confidence_threshold"]
    assert record["trigger"]["source"] == "manual"


def test_m1_m4a_auto_event_analysis_is_enabled_by_default() -> None:
    assert EngineConfig().auto_analyze_event_tasks is True
