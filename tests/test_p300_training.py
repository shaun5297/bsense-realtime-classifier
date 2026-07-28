from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from bsense_classifier.p300_training import (
    BANDPASS_HZ,
    COMMANDS,
    P300Dataset,
    _public_metadata_row,
    _public_training_metadata,
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


def test_public_training_metadata_removes_paths_and_subject_labels() -> None:
    dataset = P300Dataset(
        features=np.empty((2, 1)),
        raw_erp_features=np.empty((2, 1)),
        targets=np.array([0, 1]),
        groups=np.array(["person-b", "person-a"], dtype=object),
        metadata=(),
        feature_names=("feature",),
        raw_erp_feature_names=("raw",),
        recordings=(
            r"D:\private\person-b\session.xdf",
            r"D:\private\person-a\session.xdf",
        ),
        skipped_windows=0,
    )

    group_aliases, recording_aliases = _public_training_metadata(dataset)
    row = _public_metadata_row(
        {
            "subject_id": "person-b",
            "recording": dataset.recordings[0],
            "global_trial": 1,
        },
        group_aliases=group_aliases,
        recording_aliases=recording_aliases,
    )

    assert group_aliases == {
        "person-a": "acquisition_001",
        "person-b": "acquisition_002",
    }
    assert list(recording_aliases.values()) == [
        "recording_001.xdf",
        "recording_002.xdf",
    ]
    assert row == {
        "subject_id": "acquisition_002",
        "recording": "recording_001.xdf",
        "global_trial": 1,
    }


def test_packaged_p300_models_use_public_training_metadata() -> None:
    project_root = Path(__file__).resolve().parents[1]
    model_paths = (
        project_root / "models" / "m7_p300" / "model.joblib",
        project_root / "models" / "m7_p300" / "eegnet" / "model.joblib",
    )

    for model_path in model_paths:
        artifact = joblib.load(model_path)
        assert artifact["training_subjects"] == [
            "acquisition_001",
            "acquisition_002",
            "acquisition_003",
        ]
        assert artifact["training_recordings"] == [
            "recording_001.xdf",
            "recording_002.xdf",
            "recording_003.xdf",
        ]
