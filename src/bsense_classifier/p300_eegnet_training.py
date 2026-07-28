"""Train and export a two-channel EEGNet for BSense M7 P300 flashes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

from .eegnet_model import TorchEEGNetClassifier
from .p300_training import (
    BANDPASS_HZ,
    TARGET_SFREQ,
    WINDOW_OFFSET_SECONDS,
    WINDOW_SECONDS,
    P300Dataset,
    _json_ready,
    _public_metadata_row,
    _public_training_metadata,
    _target_probabilities,
    _write_csv,
    aggregate_trials,
    build_dataset,
    flash_metrics,
)


MODEL_NAME = "eegnet_2ch_erp"


def _new_classifier(
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    device: str,
) -> TorchEEGNetClassifier:
    return TorchEEGNetClassifier(
        n_channels=2,
        n_times=int(round(TARGET_SFREQ * WINDOW_SECONDS)),
        temporal_filters=8,
        depth_multiplier=2,
        pointwise_filters=16,
        kernel_length=64,
        dropout=0.5,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=1e-3,
        weight_decay=1e-4,
        validation_fraction=0.15,
        patience=5,
        random_state=seed,
        device=device,
    )


def _artifact(
    model: TorchEEGNetClassifier,
    dataset: P300Dataset,
    *,
    groups: list[str],
) -> dict[str, Any]:
    group_aliases, recording_aliases = _public_training_metadata(dataset)
    return {
        "artifact_schema_version": 2,
        "task": "m7_p300",
        "target_kind": "classification",
        "model_family": "pytorch",
        "model_name": MODEL_NAME,
        "model": model,
        "feature_names": list(dataset.raw_erp_feature_names),
        "feature_mode": "erp_raw",
        "sfreq": TARGET_SFREQ,
        "window_seconds": WINDOW_SECONDS,
        "window_offset_seconds": WINDOW_OFFSET_SECONDS,
        "stride_seconds": 0.175,
        "bandpass_hz": list(BANDPASS_HZ),
        "channel_count": 2,
        "channel_order": ["FP1", "FP2"],
        "label_mapping": {0: "non_target", 1: "target"},
        "unknown_state": -1,
        "default_confidence_threshold": 0.55,
        "deployment_mode": "event_locked_marker_required",
        "experimental_model": True,
        "control_output_enabled": True,
        "training_subjects": [group_aliases[group] for group in groups],
        "training_recordings": [
            recording_aliases[recording] for recording in dataset.recordings
        ],
        "evaluation": "leave_one_acquisition_folder_out",
        "architecture_reference": (
            "Lawhern EEGNet; adapted from local Braindecode/ARL references"
        ),
        "p300_control_defaults": {
            "sequences_per_trial": 10,
            "minimum_flashes_per_command": 6,
            "confidence_threshold": 0.55,
            "margin_threshold": 0.08,
            "minimum_quality_ratio": 0.8,
        },
    }


def train_and_export_eegnet(
    dataset: P300Dataset,
    output_dir: Path,
    *,
    seed: int = 20260728,
    epochs: int = 24,
    batch_size: int = 256,
    device: str = "auto",
) -> dict[str, Any]:
    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    groups = sorted(set(str(value) for value in dataset.groups))
    if len(groups) < 2:
        raise ValueError("EEGNet 留组评估至少需要两份独立采集目录。")
    group_aliases, recording_aliases = _public_training_metadata(dataset)
    public_groups = [group_aliases[group] for group in groups]
    public_recordings = [
        recording_aliases[recording] for recording in dataset.recordings
    ]

    oof = np.full(len(dataset.targets), np.nan, dtype=np.float64)
    fold_reports: list[dict[str, Any]] = []
    splitter = LeaveOneGroupOut()
    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(
            dataset.raw_erp_features,
            dataset.targets,
            dataset.groups,
        ),
        start=1,
    ):
        model = _new_classifier(
            seed=seed + fold,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
        )
        model.fit(
            dataset.raw_erp_features[train_indices],
            dataset.targets[train_indices],
        )
        probabilities = _target_probabilities(
            model,
            dataset.raw_erp_features[test_indices],
        )
        oof[test_indices] = probabilities
        fold_metadata = [dataset.metadata[index] for index in test_indices]
        trial_report, _trial_rows = aggregate_trials(
            fold_metadata,
            probabilities,
        )
        fold_reports.append(
            {
                "fold": fold,
                "test_subject": group_aliases[
                    str(dataset.groups[test_indices][0])
                ],
                "train_subjects": sorted(
                    {
                        group_aliases[str(value)]
                        for value in dataset.groups[train_indices]
                    }
                ),
                "epochs_run": len(model.training_history_),
                "best_validation_loss": model.best_validation_loss_,
                "device": model.device_used_,
                "flash_metrics": flash_metrics(
                    dataset.targets[test_indices],
                    probabilities,
                ),
                "trial_metrics": trial_report,
            }
        )
    if not np.isfinite(oof).all():
        raise RuntimeError("EEGNet 的交叉验证预测不完整。")

    trial_report, trial_rows = aggregate_trials(dataset.metadata, oof)
    flash_report = flash_metrics(dataset.targets, oof)
    _write_csv(
        output / "trial_predictions_eegnet.csv",
        [
            _public_metadata_row(
                row,
                group_aliases=group_aliases,
                recording_aliases=recording_aliases,
            )
            for row in trial_rows
        ],
    )
    flash_rows = [
        {
            **_public_metadata_row(
                row,
                group_aliases=group_aliases,
                recording_aliases=recording_aliases,
            ),
            "target_probability": float(probability),
            "predicted_target": int(probability >= 0.5),
        }
        for row, probability in zip(dataset.metadata, oof, strict=True)
    ]
    _write_csv(output / "cross_subject_flash_predictions.csv", flash_rows)

    final_model = _new_classifier(
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
    )
    final_model.fit(dataset.raw_erp_features, dataset.targets)
    model_path = output / "model.joblib"
    joblib.dump(
        _artifact(final_model, dataset, groups=groups),
        model_path,
    )

    report = {
        "task": "m7_p300",
        "model_name": MODEL_NAME,
        "training_kind": "from_scratch",
        "pretrained_weights_used": False,
        "pretrained_reason": (
            "本地权重的框架、60通道、151点和4分类契约与当前2通道、300点、"
            "二分类不匹配。"
        ),
        "recording_count": len(dataset.recordings),
        "recordings": public_recordings,
        "subject_groups": public_groups,
        "valid_flash_epochs": len(dataset.targets),
        "class_counts": {
            "non_target": int(np.sum(dataset.targets == 0)),
            "target": int(np.sum(dataset.targets == 1)),
        },
        "input_shape": [2, int(round(TARGET_SFREQ * WINDOW_SECONDS))],
        "feature_mode": "erp_raw",
        "evaluation": "leave_one_acquisition_folder_out",
        "flash_metrics": flash_report,
        "trial_metrics": trial_report,
        "folds": fold_reports,
        "final_training": {
            "epochs_run": len(final_model.training_history_),
            "best_validation_loss": final_model.best_validation_loss_,
            "device": final_model.device_used_,
        },
        "model_path": model_path.name,
        "limitations": [
            "FP1/FP2 不是典型 P300 顶区通道。",
            "仅有三个独立采集组，深度模型指标不稳定。",
            "当前结果只用于工程比较，真机仍须保留全部安全联锁。",
        ],
    }
    (output / "training_report.json").write_text(
        json.dumps(_json_ready(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = build_dataset(args.data_root)
    report = train_and_export_eegnet(
        dataset,
        args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(_json_ready(report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
