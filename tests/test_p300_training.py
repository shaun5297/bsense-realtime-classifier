from __future__ import annotations

from pathlib import Path

import numpy as np

from bsense_classifier.p300_training import (
    BANDPASS_HZ,
    COMMANDS,
    aggregate_trials,
    batch_erp_features,
    discover_recordings,
)
from bsense_classifier.model_runtime import extract_features


def test_discover_recordings_ignores_macos_resource_forks(
    tmp_path: Path,
) -> None:
    real = tmp_path / "person-a" / "sub-x_task-m7_p300_run-001.xdf"
    real.parent.mkdir()
    real.touch()
    ignored = (
        tmp_path
        / "__MACOSX"
        / "person-a"
        / "._sub-x_task-m7_p300_run-001.xdf"
    )
    ignored.parent.mkdir(parents=True)
    ignored.touch()
    assert discover_recordings(tmp_path) == [real.resolve()]


def test_trial_aggregation_selects_highest_mean_command() -> None:
    metadata = []
    probabilities = []
    for sequence in range(1, 6):
        for command in COMMANDS:
            metadata.append(
                {
                    "subject_id": "S01",
                    "recording": "recording.xdf",
                    "global_trial": 7,
                    "flash_command": command,
                    "target_command": "right",
                }
            )
            probabilities.append(0.8 if command == "right" else 0.2)
    report, rows = aggregate_trials(metadata, np.asarray(probabilities))
    assert report["trial_count"] == 1
    assert report["command_accuracy"] == 1.0
    assert rows[0]["predicted_command"] == "right"
    assert rows[0]["margin"] > 0.5


def test_batch_erp_features_match_runtime_extraction() -> None:
    generator = np.random.default_rng(20260728)
    raw = generator.normal(0.0, 10.0, size=(4, 2, 300))
    batch, names, _processed, quality = batch_erp_features(raw)
    assert quality.all()
    for index in range(len(raw)):
        expected, expected_names, _ = extract_features(
            raw[index], 250.0, "erp", BANDPASS_HZ
        )
        assert tuple(expected_names) == names
        np.testing.assert_allclose(batch[index], expected, rtol=1e-9, atol=1e-9)


def test_full_erp_features_match_runtime_extraction() -> None:
    generator = np.random.default_rng(17)
    raw = generator.normal(0.0, 10.0, size=(3, 2, 300))
    _batch, _names, processed, quality = batch_erp_features(raw)
    assert quality.all()
    baseline_samples = 50
    corrected = processed - processed[:, :, :baseline_samples].mean(
        axis=2, keepdims=True
    )
    for index in range(len(raw)):
        expected, names, _ = extract_features(
            raw[index], 250.0, "erp_raw", BANDPASS_HZ
        )
        assert len(names) == 600
        np.testing.assert_allclose(
            corrected[index].reshape(-1),
            expected,
            rtol=1e-9,
            atol=1e-9,
        )
