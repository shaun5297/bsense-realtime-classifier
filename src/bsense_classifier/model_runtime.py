"""Model artifact loading and training-compatible FP1/FP2 feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from scipy.signal import butter, resample, sosfiltfilt, welch


EXPECTED_CHANNELS = ("FP1", "FP2")
BANDS = {
    "delta": (1.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 25.0),
    "gamma": (25.0, 45.0),
}


class ArtifactError(ValueError):
    """Raised when a model artifact is not compatible with this application."""


@dataclass(frozen=True)
class ChannelPlan:
    indices: tuple[int, int]
    labels: tuple[str, str]
    status: str
    warning: str | None


@dataclass(frozen=True)
class InferenceResult:
    task: str
    target_kind: str
    signal_quality: str
    quality_ok: bool
    state: int | None = None
    candidate: int | None = None
    label: str | None = None
    confidence: float | None = None
    probabilities: tuple[float, ...] = ()
    classes: tuple[int, ...] = ()
    value: float | None = None
    raw_value: float | None = None
    target_name: str | None = None


def normalize_channel_label(label: str) -> str:
    return (
        str(label)
        .strip()
        .upper()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
    )


def plan_channels(labels: Sequence[str], channel_count: int) -> ChannelPlan:
    """Validate or infer the FP1/FP2 order of an incoming EEG stream."""

    if channel_count != 2:
        raise ArtifactError(
            f"模型要求 2 个 EEG 通道（FP1、FP2），当前流提供 {channel_count} 个。"
        )
    clean = tuple(normalize_channel_label(label) for label in labels if label)
    if not clean:
        return ChannelPlan(
            indices=(0, 1),
            labels=EXPECTED_CHANNELS,
            status="assumed",
            warning="LSL 没有通道标签；程序按第 1 通道=FP1、第 2 通道=FP2 处理。",
        )
    if len(clean) != 2:
        raise ArtifactError("LSL 通道标签不完整，无法确认 FP1/FP2 顺序。")
    if clean == EXPECTED_CHANNELS:
        return ChannelPlan(
            indices=(0, 1),
            labels=EXPECTED_CHANNELS,
            status="verified",
            warning=None,
        )
    if clean == tuple(reversed(EXPECTED_CHANNELS)):
        return ChannelPlan(
            indices=(1, 0),
            labels=EXPECTED_CHANNELS,
            status="reordered",
            warning="检测到 FP2、FP1 顺序，程序已自动交换为 FP1、FP2。",
        )
    raise ArtifactError(
        f"模型只接受 FP1、FP2，当前 LSL 标签为：{', '.join(labels)}。"
    )


def preprocess_window(
    window: np.ndarray,
    sfreq: float,
    bandpass_hz: tuple[float, float],
) -> np.ndarray:
    values = np.asarray(window, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 2:
        raise ArtifactError(f"期望窗口形状为 (2, samples)，实际为 {values.shape}。")
    centered = values - np.median(values, axis=1, keepdims=True)
    low, high = bandpass_hz
    sos = butter(4, [low, high], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, centered, axis=1)


def signal_quality(processed: np.ndarray) -> tuple[bool, str]:
    if not np.isfinite(processed).all():
        return False, "non_finite"
    channel_std = processed.std(axis=1)
    if np.any(channel_std < 1e-8):
        return False, "flat_channel"
    standardized = processed / np.maximum(channel_std[:, None], 1e-8)
    if float(np.max(np.abs(standardized))) > 20.0:
        return False, "extreme_transient"
    return True, "ok"


def _spectral_channel(
    signal: np.ndarray, sfreq: float
) -> tuple[list[float], list[str]]:
    frequencies, power = welch(
        signal, fs=sfreq, nperseg=min(len(signal), int(4 * sfreq))
    )
    total_mask = (frequencies >= 1.5) & (frequencies <= 45.0)
    total_power = max(
        float(np.trapezoid(power[total_mask], frequencies[total_mask])), 1e-12
    )
    values: list[float] = []
    names: list[str] = []
    relative: dict[str, float] = {}
    for name, (low, high) in BANDS.items():
        mask = (frequencies >= low) & (frequencies < high)
        band_power = float(np.trapezoid(power[mask], frequencies[mask]))
        relative[name] = band_power / total_power
        values.extend([np.log10(max(band_power, 1e-12)), relative[name]])
        names.extend([f"log_{name}", f"rel_{name}"])
    normalized = power[total_mask] / max(float(power[total_mask].sum()), 1e-12)
    entropy = -float(
        np.sum(normalized * np.log2(np.maximum(normalized, 1e-12)))
    )
    cumulative = np.cumsum(power[total_mask])
    median_frequency = float(
        frequencies[total_mask][np.searchsorted(cumulative, cumulative[-1] / 2)]
    )
    values.extend(
        [
            entropy,
            median_frequency,
            np.log10(max(float(np.mean(signal**2)), 1e-12)),
            float(np.mean(np.abs(np.diff(signal)))),
            relative["theta"] / max(relative["alpha"], 1e-12),
            relative["beta"]
            / max(relative["theta"] + relative["alpha"], 1e-12),
        ]
    )
    names.extend(
        [
            "spectral_entropy",
            "median_frequency",
            "log_rms2",
            "line_length",
            "theta_alpha",
            "beta_theta_alpha",
        ]
    )
    return values, names


def spectral_features(
    processed: np.ndarray, sfreq: float
) -> tuple[np.ndarray, list[str]]:
    values: list[float] = []
    names: list[str] = []
    for channel in range(2):
        channel_values, channel_names = _spectral_channel(
            processed[channel], sfreq
        )
        values.extend(channel_values)
        names.extend(
            [f"ch{channel + 1}_{name}" for name in channel_names]
        )
    if np.all(processed.std(axis=1) > 1e-12):
        correlation = float(np.corrcoef(processed)[0, 1])
    else:
        correlation = 0.0
    if not np.isfinite(correlation):
        correlation = 0.0
    values.append(correlation)
    names.append("channel_correlation")
    return np.asarray(values, dtype=np.float64), names


def erp_features(
    processed: np.ndarray,
    sfreq: float,
    baseline_seconds: float = 0.2,
    output_sfreq: float = 50.0,
) -> tuple[np.ndarray, list[str]]:
    baseline_samples = max(1, int(round(baseline_seconds * sfreq)))
    corrected = processed - processed[:, :baseline_samples].mean(
        axis=1, keepdims=True
    )
    output_samples = int(round(corrected.shape[1] * output_sfreq / sfreq))
    downsampled = resample(corrected, output_samples, axis=1)
    names = [
        f"ch{channel + 1}_erp_t{sample / output_sfreq - baseline_seconds:.3f}"
        for channel in range(2)
        for sample in range(output_samples)
    ]
    return downsampled.reshape(-1).astype(np.float64), names


def erp_raw_features(
    processed: np.ndarray,
    sfreq: float,
    baseline_seconds: float = 0.2,
) -> tuple[np.ndarray, list[str]]:
    """Keep the full baseline-corrected ERP for compact neural networks."""

    baseline_samples = max(1, int(round(baseline_seconds * sfreq)))
    corrected = processed - processed[:, :baseline_samples].mean(
        axis=1, keepdims=True
    )
    names = [
        f"ch{channel + 1}_erp_raw_t{sample / sfreq - baseline_seconds:.3f}"
        for channel in range(2)
        for sample in range(corrected.shape[1])
    ]
    return corrected.reshape(-1).astype(np.float64), names


def extract_features(
    raw_window: np.ndarray,
    sfreq: float,
    feature_mode: str,
    bandpass_hz: tuple[float, float],
) -> tuple[np.ndarray, list[str], np.ndarray]:
    processed = preprocess_window(raw_window, sfreq, bandpass_hz)
    if feature_mode == "spectral":
        features, names = spectral_features(processed, sfreq)
    elif feature_mode == "erp":
        features, names = erp_features(processed, sfreq)
    elif feature_mode == "erp_raw":
        features, names = erp_raw_features(processed, sfreq)
    else:
        raise ArtifactError(f"不支持的特征类型：{feature_mode}")
    return features, names, processed


class ModelRuntime:
    """Validated sklearn artifact with a stable inference interface."""

    REQUIRED_FIELDS = {
        "artifact_schema_version",
        "task",
        "target_kind",
        "model",
        "feature_mode",
        "sfreq",
        "window_seconds",
        "stride_seconds",
        "bandpass_hz",
        "channel_count",
        "deployment_mode",
    }

    def __init__(self, artifact: dict[str, Any], source: Path | None = None) -> None:
        missing = sorted(self.REQUIRED_FIELDS - set(artifact))
        if missing:
            raise ArtifactError(f"模型文件缺少字段：{', '.join(missing)}")
        if int(artifact["artifact_schema_version"]) != 2:
            raise ArtifactError("只支持 artifact_schema_version=2 的模型。")
        if int(artifact["channel_count"]) != 2:
            raise ArtifactError("当前程序只支持 FP1、FP2 两通道模型。")
        if artifact["target_kind"] not in {"classification", "regression"}:
            raise ArtifactError("模型 target_kind 必须是 classification 或 regression。")
        if artifact["target_kind"] == "classification" and "label_mapping" not in artifact:
            raise ArtifactError("分类模型缺少 label_mapping。")
        self.artifact = artifact
        self.source = source
        self.model = artifact["model"]
        self.task = str(artifact["task"])
        self.target_kind = str(artifact["target_kind"])
        self.sfreq = float(artifact["sfreq"])
        self.window_seconds = float(artifact["window_seconds"])
        self.window_offset_seconds = float(
            artifact.get("window_offset_seconds", 0.0)
        )
        self.stride_seconds = float(artifact["stride_seconds"])
        self.feature_mode = str(artifact["feature_mode"])
        self.bandpass_hz = tuple(float(x) for x in artifact["bandpass_hz"])
        self.deployment_mode = str(artifact["deployment_mode"])
        self.model_name = str(artifact.get("model_name", type(self.model).__name__))
        self.default_threshold = float(
            artifact.get("default_confidence_threshold", 0.0)
        )
        self.label_mapping = {
            int(key): str(value)
            for key, value in artifact.get("label_mapping", {}).items()
        }
        self.feature_names = [str(value) for value in artifact.get("feature_names", [])]

    @classmethod
    def load(cls, path: Path | str) -> "ModelRuntime":
        source = Path(path).expanduser().resolve()
        if not source.exists():
            raise ArtifactError(f"模型文件不存在：{source}")
        artifact = joblib.load(source)
        if not isinstance(artifact, dict):
            raise ArtifactError("模型文件必须是字典形式的 BSense artifact。")
        return cls(artifact, source)

    @property
    def event_locked(self) -> bool:
        return self.deployment_mode in {
            "cue_locked_window",
            "event_locked_marker_required",
        }

    @property
    def classes(self) -> tuple[int, ...]:
        if self.target_kind != "classification":
            return ()
        return tuple(int(value) for value in np.asarray(self.model.classes_))

    def infer(self, raw_window: np.ndarray) -> InferenceResult:
        features, names, processed = extract_features(
            raw_window,
            self.sfreq,
            self.feature_mode,
            self.bandpass_hz,
        )
        if self.feature_names and names != self.feature_names:
            raise ArtifactError("实时特征定义与模型训练时不一致，已停止推理。")
        quality_ok, quality_reason = signal_quality(processed)
        if self.target_kind == "regression":
            raw_value = float(self.model.predict(features.reshape(1, -1))[0])
            valid_range = self.artifact.get("target_valid_range")
            value = (
                float(np.clip(raw_value, valid_range[0], valid_range[1]))
                if valid_range
                else raw_value
            )
            return InferenceResult(
                task=self.task,
                target_kind=self.target_kind,
                signal_quality=quality_reason,
                quality_ok=quality_ok,
                value=value,
                raw_value=raw_value,
                target_name=str(self.artifact.get("target_name", "value")),
            )
        probabilities = np.asarray(
            self.model.predict_proba(features.reshape(1, -1))[0], dtype=float
        )
        classes = self.classes
        index = int(np.argmax(probabilities))
        candidate = classes[index]
        return InferenceResult(
            task=self.task,
            target_kind=self.target_kind,
            signal_quality=quality_reason,
            quality_ok=quality_ok,
            candidate=candidate,
            label=self.label_mapping.get(candidate, str(candidate)),
            confidence=float(probabilities[index]),
            probabilities=tuple(float(value) for value in probabilities),
            classes=classes,
        )
