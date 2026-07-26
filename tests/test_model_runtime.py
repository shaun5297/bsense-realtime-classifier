from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bsense_classifier.model_runtime import (
    ArtifactError,
    ModelRuntime,
    erp_features,
    plan_channels,
    preprocess_window,
    spectral_features,
)
from bsense_classifier.task_guidance import GUIDANCE, guidance_for


def test_channel_plan_verifies_and_reorders_fp1_fp2() -> None:
    verified = plan_channels(["Fp1", "FP2"], 2)
    assert verified.indices == (0, 1)
    assert verified.status == "verified"

    reordered = plan_channels(["FP2", "FP1"], 2)
    assert reordered.indices == (1, 0)
    assert reordered.status == "reordered"


def test_channel_plan_assumes_only_when_labels_are_absent() -> None:
    assumed = plan_channels([], 2)
    assert assumed.indices == (0, 1)
    assert assumed.status == "assumed"
    with pytest.raises(ArtifactError):
        plan_channels(["AF7", "AF8"], 2)
    with pytest.raises(ArtifactError):
        plan_channels(["FP1", "FP2", "Cz"], 3)


def test_feature_shapes_match_training_contract() -> None:
    sfreq = 250.0
    seconds = 4.0
    time = np.arange(int(sfreq * seconds)) / sfreq
    raw = np.vstack(
        (
            np.sin(2 * np.pi * 10 * time),
            np.sin(2 * np.pi * 12 * time + 0.2),
        )
    )
    processed = preprocess_window(raw, sfreq, (1.0, 45.0))
    spectral, spectral_names = spectral_features(processed, sfreq)
    assert spectral.shape == (33,)
    assert len(spectral_names) == 33

    erp, erp_names = erp_features(processed[:, :300], sfreq)
    assert erp.shape == (120,)
    assert len(erp_names) == 120


def test_every_bundled_model_loads_and_runs() -> None:
    model_root = Path(__file__).resolve().parents[1] / "models"
    expected = set(GUIDANCE)
    discovered = {path.parent.name for path in model_root.glob("*/model.joblib")}
    assert discovered == expected
    generator = np.random.default_rng(2026)
    for task in sorted(expected):
        runtime = ModelRuntime.load(model_root / task / "model.joblib")
        sample_count = int(round(runtime.sfreq * runtime.window_seconds))
        raw = generator.normal(0.0, 10.0, size=(2, sample_count))
        result = runtime.infer(raw)
        assert result.task == task
        assert result.target_kind in {"classification", "regression"}


def test_all_tasks_have_actionable_guidance() -> None:
    for task in GUIDANCE:
        guidance = guidance_for(task)
        assert guidance.title
        assert len(guidance.steps) >= 3
        assert guidance.trigger_note
        assert guidance.limitation
    assert "自动分析" in guidance_for("m1_mi").trigger_note
    assert "自动分析" in guidance_for("m4a_intent").trigger_note


class _StubClassifier:
    classes_ = np.array([0, 1])

    def predict_proba(self, _features: np.ndarray) -> np.ndarray:
        return np.array([[0.01, 0.99]])


def test_incomplete_label_mapping_falls_back_to_class_id() -> None:
    artifact = {
        "artifact_schema_version": 2,
        "task": "m3a_artifact",
        "target_kind": "classification",
        "model": _StubClassifier(),
        "feature_mode": "spectral",
        "sfreq": 250.0,
        "window_seconds": 4.0,
        "stride_seconds": 1.0,
        "bandpass_hz": (1.0, 45.0),
        "channel_count": 2,
        "deployment_mode": "continuous",
        "label_mapping": {0: "clean_baseline"},
    }
    runtime = ModelRuntime(artifact)
    raw = np.random.default_rng(0).normal(0.0, 10.0, size=(2, 1000))
    result = runtime.infer(raw)
    assert result.candidate == 1
    assert result.label == "1"
