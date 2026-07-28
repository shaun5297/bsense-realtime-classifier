"""Train a deployable M7 P300 model from BSense XDF recordings."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, resample, sosfiltfilt

from .model_runtime import erp_raw_features, extract_features


TARGET_SFREQ = 250.0
WINDOW_OFFSET_SECONDS = -0.2
WINDOW_SECONDS = 1.2
BANDPASS_HZ = (0.5, 20.0)
COMMANDS = ("forward", "backward", "left", "right", "stop", "idle")


@dataclass(frozen=True)
class P300Dataset:
    features: np.ndarray
    raw_erp_features: np.ndarray
    targets: np.ndarray
    groups: np.ndarray
    metadata: tuple[dict[str, Any], ...]
    feature_names: tuple[str, ...]
    raw_erp_feature_names: tuple[str, ...]
    recordings: tuple[str, ...]
    skipped_windows: int


def _import_pyxdf() -> Any:
    try:
        import pyxdf
    except ImportError as exc:
        raise RuntimeError(
            "训练 M7 模型需要 pyxdf；请安装项目的 xdf 可选依赖。"
        ) from exc
    return pyxdf


def discover_recordings(data_root: Path) -> list[Path]:
    """Discover real M7 XDF files while ignoring macOS resource forks."""

    root = data_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"数据目录不存在：{root}")
    return sorted(
        path
        for path in root.rglob("*task-m7_p300*.xdf")
        if path.is_file()
        and "__MACOSX" not in path.parts
        and not path.name.startswith("._")
    )


def _stream_type(stream: dict[str, Any]) -> str:
    return str(stream.get("info", {}).get("type", [""])[0]).strip().lower()


def _find_stream(
    streams: Iterable[dict[str, Any]], stream_type: str
) -> dict[str, Any]:
    matches = [
        stream for stream in streams if _stream_type(stream) == stream_type.lower()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"期望恰好一个 {stream_type} 流，实际找到 {len(matches)} 个。"
        )
    return matches[0]


def _load_xdf(path: Path) -> list[dict[str, Any]]:
    pyxdf = _import_pyxdf()
    logger = logging.getLogger("pyxdf.pyxdf")
    previous_level = logger.level
    logger.setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="'where' used without 'out'.*",
            category=UserWarning,
        )
        try:
            streams, _header = pyxdf.load_xdf(str(path), verbose=False)
        finally:
            logger.setLevel(previous_level)
    return streams


def _marker_rows(marker_stream: dict[str, Any]) -> list[dict[str, Any]]:
    values = np.asarray(marker_stream["time_series"], dtype=object).reshape(-1)
    timestamps = np.asarray(marker_stream["time_stamps"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for value, timestamp in zip(values, timestamps, strict=True):
        if isinstance(value, bytes):
            text = value.decode("utf-8")
        else:
            text = str(value)
        try:
            row = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            row["_timestamp_xdf"] = float(timestamp)
            rows.append(row)
    return rows


def _interpolate_epoch(
    source_times: np.ndarray,
    source_values: np.ndarray,
    start_time: float,
    duration_seconds: float = WINDOW_SECONDS,
    sfreq: float = TARGET_SFREQ,
) -> np.ndarray | None:
    if source_values.ndim != 2 or source_values.shape[1] != 2:
        raise ValueError(
            f"M7 模型要求双通道 EEG，实际数据形状为 {source_values.shape}。"
        )
    sample_count = int(round(duration_seconds * sfreq))
    target_times = start_time + np.arange(sample_count, dtype=float) / sfreq
    if (
        len(source_times) < 2
        or target_times[0] < source_times[0]
        or target_times[-1] > source_times[-1]
    ):
        return None
    unique_times, unique_indices = np.unique(source_times, return_index=True)
    unique_values = source_values[unique_indices]
    if len(unique_times) < 2:
        return None
    return np.vstack(
        [
            np.interp(target_times, unique_times, unique_values[:, channel])
            for channel in range(2)
        ]
    )


def _recording_group(path: Path) -> str:
    """Use the acquisition folder, not the repeated pilot01 marker, as identity."""

    return path.parent.name


def batch_erp_features(
    raw_windows: np.ndarray,
    *,
    sfreq: float = TARGET_SFREQ,
    bandpass_hz: tuple[float, float] = BANDPASS_HZ,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, np.ndarray]:
    """Vectorized equivalent of the runtime ERP feature path."""

    values = np.asarray(raw_windows, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != 2:
        raise ValueError(
            f"期望 ERP 批次形状为 (epochs, 2, samples)，实际为 {values.shape}。"
        )
    centered = values - np.median(values, axis=2, keepdims=True)
    sos = butter(
        4,
        [float(bandpass_hz[0]), float(bandpass_hz[1])],
        btype="bandpass",
        fs=sfreq,
        output="sos",
    )
    processed = sosfiltfilt(sos, centered, axis=2)
    baseline_samples = max(1, int(round(0.2 * sfreq)))
    corrected = processed - processed[:, :, :baseline_samples].mean(
        axis=2, keepdims=True
    )
    output_samples = int(round(corrected.shape[2] * 50.0 / sfreq))
    downsampled = resample(corrected, output_samples, axis=2)
    features = downsampled.reshape(len(values), -1).astype(np.float64)

    _single_features, names, _single_processed = extract_features(
        values[0],
        sfreq,
        "erp",
        bandpass_hz,
    )
    if not np.allclose(features[0], _single_features, rtol=1e-9, atol=1e-9):
        raise RuntimeError("批量 ERP 特征与实时端特征不一致。")

    channel_std = processed.std(axis=2)
    finite = np.isfinite(processed).all(axis=(1, 2))
    non_flat = np.all(channel_std >= 1e-8, axis=1)
    standardized = processed / np.maximum(channel_std[:, :, None], 1e-8)
    non_extreme = np.max(np.abs(standardized), axis=(1, 2)) <= 20.0
    quality_ok = finite & non_flat & non_extreme
    return features, tuple(names), processed, quality_ok


def build_dataset(
    data_root: Path,
    *,
    selected_groups: set[str] | None = None,
    max_trials_per_recording: int | None = None,
) -> P300Dataset:
    recordings = discover_recordings(data_root)
    if selected_groups is not None:
        recordings = [
            path for path in recordings if _recording_group(path) in selected_groups
        ]
    if not recordings:
        raise ValueError(f"在 {data_root.resolve()} 下没有找到 M7 P300 XDF。")

    feature_rows: list[np.ndarray] = []
    raw_erp_feature_rows: list[np.ndarray] = []
    targets: list[int] = []
    groups: list[str] = []
    metadata: list[dict[str, Any]] = []
    feature_names: list[str] | None = None
    raw_erp_feature_names: list[str] | None = None
    skipped_windows = 0

    for path in recordings:
        streams = _load_xdf(path)
        eeg = _find_stream(streams, "eeg")
        markers = _find_stream(streams, "markers")
        source_times = np.asarray(eeg["time_stamps"], dtype=np.float64)
        source_values = np.asarray(eeg["time_series"], dtype=np.float64)
        group = _recording_group(path)
        flash_rows = [
            row for row in _marker_rows(markers) if row.get("event") == "p300_flash"
        ]
        if max_trials_per_recording is not None:
            if max_trials_per_recording < 1:
                raise ValueError("max_trials_per_recording 必须至少为 1。")
            trial_ids = sorted(
                {
                    int(row.get("global_trial", row.get("trial", 0)))
                    for row in flash_rows
                }
            )[:max_trials_per_recording]
            flash_rows = [
                row
                for row in flash_rows
                if int(row.get("global_trial", row.get("trial", 0))) in trial_ids
            ]
        if not flash_rows:
            raise ValueError(f"{path.name} 没有 p300_flash Marker。")

        recording_raw: list[np.ndarray] = []
        recording_targets: list[int] = []
        recording_metadata: list[dict[str, Any]] = []
        for row in flash_rows:
            command = str(row.get("flash_command", "")).strip().lower()
            target_command = str(row.get("target_command", "")).strip().lower()
            if command not in COMMANDS or target_command not in COMMANDS:
                skipped_windows += 1
                continue
            marker_time = float(row["_timestamp_xdf"])
            raw = _interpolate_epoch(
                source_times,
                source_values,
                marker_time + WINDOW_OFFSET_SECONDS,
            )
            if raw is None:
                skipped_windows += 1
                continue
            label = int(bool(row.get("is_target")))
            recording_raw.append(raw)
            recording_targets.append(label)
            recording_metadata.append(
                {
                    "subject_id": group,
                    "source_xdf": str(path.resolve()),
                    "recording": path.name,
                    "global_trial": int(row.get("global_trial", row.get("trial", 0))),
                    "block_number": int(row.get("block_number", 0)),
                    "sequence": int(row.get("sequence", 0)),
                    "flash_command": command,
                    "target_command": target_command,
                    "is_target": label,
                    "marker_timestamp_xdf": marker_time,
                    "quality_status": "ok",
                }
            )
        if not recording_raw:
            continue
        batch_features, names, processed, quality_mask = batch_erp_features(
            np.asarray(recording_raw, dtype=np.float64)
        )
        baseline_samples = int(round(0.2 * TARGET_SFREQ))
        corrected = processed - processed[:, :, :baseline_samples].mean(
            axis=2, keepdims=True
        )
        raw_batch = corrected.reshape(len(corrected), -1).astype(np.float32)
        _first_raw, raw_names = erp_raw_features(
            processed[0],
            TARGET_SFREQ,
        )
        if feature_names is None:
            feature_names = list(names)
        elif feature_names != list(names):
            raise RuntimeError("不同记录生成了不一致的 ERP 特征定义。")
        if raw_erp_feature_names is None:
            raw_erp_feature_names = list(raw_names)
        elif raw_erp_feature_names != list(raw_names):
            raise RuntimeError("不同记录生成了不一致的完整 ERP 特征定义。")
        for index, quality_ok in enumerate(quality_mask):
            if not bool(quality_ok):
                skipped_windows += 1
                continue
            feature_rows.append(batch_features[index])
            raw_erp_feature_rows.append(raw_batch[index])
            targets.append(recording_targets[index])
            groups.append(group)
            metadata.append(recording_metadata[index])

    if (
        not feature_rows
        or feature_names is None
        or raw_erp_feature_names is None
    ):
        raise ValueError("M7 数据没有生成任何有效 ERP 训练窗口。")
    target_array = np.asarray(targets, dtype=np.int64)
    if set(np.unique(target_array)) != {0, 1}:
        raise ValueError("M7 数据必须同时包含 target 和 non_target。")
    return P300Dataset(
        features=np.asarray(feature_rows, dtype=np.float64),
        raw_erp_features=np.asarray(raw_erp_feature_rows, dtype=np.float32),
        targets=target_array,
        groups=np.asarray(groups, dtype=object),
        metadata=tuple(metadata),
        feature_names=tuple(feature_names),
        raw_erp_feature_names=tuple(raw_erp_feature_names),
        recordings=tuple(str(path.resolve()) for path in recordings),
        skipped_windows=skipped_windows,
    )


def model_candidates(seed: int) -> dict[str, Any]:
    return {
        "balanced_logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1500,
                        solver="liblinear",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "shrinkage_lda": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LinearDiscriminantAnalysis(
                        solver="lsqr",
                        shrinkage="auto",
                        priors=[0.5, 0.5],
                    ),
                ),
            ]
        ),
    }


def _target_probabilities(model: Any, features: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
    classes = [int(value) for value in np.asarray(model.classes_)]
    if 1 not in classes:
        raise RuntimeError("候选模型没有 target=1 类。")
    return probabilities[:, classes.index(1)]


def flash_metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(targets, probabilities)),
        "average_precision": float(average_precision_score(targets, probabilities)),
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=[0, 1]
        ).tolist(),
    }


def aggregate_trials(
    metadata: Iterable[dict[str, Any]],
    probabilities: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row, probability in zip(metadata, probabilities, strict=True):
        key = (
            str(row["subject_id"]),
            str(row["recording"]),
            int(row["global_trial"]),
        )
        trial = grouped.setdefault(
            key,
            {
                "subject_id": key[0],
                "recording": key[1],
                "global_trial": key[2],
                "target_command": str(row["target_command"]),
                "scores": {command: [] for command in COMMANDS},
            },
        )
        trial["scores"][str(row["flash_command"])].append(float(probability))

    results: list[dict[str, Any]] = []
    for trial in grouped.values():
        if any(not trial["scores"][command] for command in COMMANDS):
            continue
        scores = {
            command: float(np.mean(trial["scores"][command]))
            for command in COMMANDS
        }
        ranked = sorted(COMMANDS, key=scores.__getitem__, reverse=True)
        results.append(
            {
                "subject_id": trial["subject_id"],
                "recording": trial["recording"],
                "global_trial": trial["global_trial"],
                "target_command": trial["target_command"],
                "predicted_command": ranked[0],
                "correct": ranked[0] == trial["target_command"],
                "confidence": scores[ranked[0]],
                "margin": scores[ranked[0]] - scores[ranked[1]],
                **{f"score_{command}": scores[command] for command in COMMANDS},
            }
        )
    accuracy = (
        float(np.mean([bool(row["correct"]) for row in results]))
        if results
        else 0.0
    )
    return {
        "trial_count": len(results),
        "command_accuracy": accuracy,
        "chance_accuracy": 1.0 / len(COMMANDS),
    }, results


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _public_training_metadata(
    dataset: P300Dataset,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return stable aliases that do not expose local paths or subject labels."""

    groups = sorted(set(str(value) for value in dataset.groups))
    group_aliases = {
        group: f"acquisition_{index:03d}"
        for index, group in enumerate(groups, start=1)
    }
    recording_aliases = {
        recording: f"recording_{index:03d}.xdf"
        for index, recording in enumerate(dataset.recordings, start=1)
    }
    return group_aliases, recording_aliases


def _public_metadata_row(
    row: dict[str, Any],
    *,
    group_aliases: dict[str, str],
    recording_aliases: dict[str, str],
) -> dict[str, Any]:
    public = dict(row)
    if "subject_id" in public:
        public["subject_id"] = group_aliases[str(public["subject_id"])]
    if "recording" in public:
        public["recording"] = recording_aliases[str(public["recording"])]
    return public


def train_and_export(
    dataset: P300Dataset,
    output_dir: Path,
    *,
    seed: int = 20260728,
) -> dict[str, Any]:
    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    unique_groups = sorted(set(str(value) for value in dataset.groups))
    if len(unique_groups) < 2:
        raise ValueError("至少需要两份独立采集目录才能进行留一受试者评估。")
    group_aliases, recording_aliases = _public_training_metadata(dataset)
    public_groups = [group_aliases[group] for group in unique_groups]
    public_recordings = [
        recording_aliases[recording] for recording in dataset.recordings
    ]

    reports: dict[str, Any] = {}
    oof_by_model: dict[str, np.ndarray] = {}
    splitter = LeaveOneGroupOut()
    for name, candidate in model_candidates(seed).items():
        oof = np.full(len(dataset.targets), np.nan, dtype=np.float64)
        folds: list[dict[str, Any]] = []
        for fold, (train_indices, test_indices) in enumerate(
            splitter.split(dataset.features, dataset.targets, dataset.groups),
            start=1,
        ):
            model = clone(candidate)
            model.fit(dataset.features[train_indices], dataset.targets[train_indices])
            fold_probabilities = _target_probabilities(
                model, dataset.features[test_indices]
            )
            oof[test_indices] = fold_probabilities
            fold_metadata = [dataset.metadata[index] for index in test_indices]
            trial_report, _trial_rows = aggregate_trials(
                fold_metadata, fold_probabilities
            )
            folds.append(
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
                    "flash_metrics": flash_metrics(
                        dataset.targets[test_indices], fold_probabilities
                    ),
                    "trial_metrics": trial_report,
                }
            )
        if not np.isfinite(oof).all():
            raise RuntimeError(f"{name} 的交叉验证预测不完整。")
        trial_report, trial_rows = aggregate_trials(dataset.metadata, oof)
        reports[name] = {
            "flash_metrics": flash_metrics(dataset.targets, oof),
            "trial_metrics": trial_report,
            "folds": folds,
        }
        oof_by_model[name] = oof
        _write_csv(
            output / f"trial_predictions_{name}.csv",
            [
                _public_metadata_row(
                    row,
                    group_aliases=group_aliases,
                    recording_aliases=recording_aliases,
                )
                for row in trial_rows
            ],
        )

    selected_name = max(
        reports,
        key=lambda name: (
            reports[name]["trial_metrics"]["command_accuracy"],
            reports[name]["flash_metrics"]["average_precision"],
            reports[name]["flash_metrics"]["balanced_accuracy"],
        ),
    )
    selected_model = clone(model_candidates(seed)[selected_name])
    selected_model.fit(dataset.features, dataset.targets)
    selected_oof = oof_by_model[selected_name]

    prediction_rows: list[dict[str, Any]] = []
    for row, probability in zip(
        dataset.metadata, selected_oof, strict=True
    ):
        prediction_rows.append(
            {
                **_public_metadata_row(
                    row,
                    group_aliases=group_aliases,
                    recording_aliases=recording_aliases,
                ),
                "target_probability": float(probability),
                "predicted_target": int(probability >= 0.5),
            }
        )
    _write_csv(output / "cross_subject_flash_predictions.csv", prediction_rows)

    artifact = {
        "artifact_schema_version": 2,
        "task": "m7_p300",
        "target_kind": "classification",
        "model_family": "sklearn",
        "model_name": selected_name,
        "model": selected_model,
        "feature_names": list(dataset.feature_names),
        "feature_mode": "erp",
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
        "training_subjects": public_groups,
        "training_recordings": public_recordings,
        "evaluation": "leave_one_acquisition_folder_out",
        "p300_control_defaults": {
            "sequences_per_trial": 10,
            "minimum_flashes_per_command": 6,
            "confidence_threshold": 0.55,
            "margin_threshold": 0.08,
            "minimum_quality_ratio": 0.8,
        },
    }
    model_path = output / "model.joblib"
    joblib.dump(artifact, model_path)

    class_counts = {
        "non_target": int(np.sum(dataset.targets == 0)),
        "target": int(np.sum(dataset.targets == 1)),
    }
    report = {
        "task": "m7_p300",
        "data_root_recordings": public_recordings,
        "recording_count": len(dataset.recordings),
        "subject_groups": public_groups,
        "valid_flash_epochs": len(dataset.targets),
        "skipped_windows": dataset.skipped_windows,
        "class_counts": class_counts,
        "selected_model": selected_name,
        "selected_flash_metrics": reports[selected_name]["flash_metrics"],
        "selected_trial_metrics": reports[selected_name]["trial_metrics"],
        "models": reports,
        "model_path": model_path.name,
        "limitations": [
            "FP1/FP2 不是典型 P300 顶区通道。",
            "当前独立采集人数很少，指标仅用于工程联调。",
            "真实机器狗必须继续使用解锁、避障、急停和动作超时联锁。",
        ],
    }
    report_path = output / "training_report.json"
    report_path.write_text(
        json.dumps(_json_ready(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="包含 BSense 采集目录的根路径。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="模型和评估报告输出目录。",
    )
    parser.add_argument("--seed", type=int, default=20260728)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = build_dataset(args.data_root)
    report = train_and_export(dataset, args.output_dir, seed=args.seed)
    print(json.dumps(_json_ready(report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
