from __future__ import annotations

import numpy as np

from bsense_classifier.lsl_engine import (
    EngineConfig,
    PendingTrigger,
    interpolate_buffer,
    result_record,
)
from bsense_classifier.model_runtime import InferenceResult


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
