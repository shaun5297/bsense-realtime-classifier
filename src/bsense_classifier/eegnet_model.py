"""Compact PyTorch EEGNet adapter for two-channel event-related potentials."""

from __future__ import annotations

import copy
import random
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split


def _import_torch() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "EEGNet 训练或推理需要 PyTorch；请安装项目的 deep 可选依赖。"
        ) from exc
    return torch, nn


def _build_eegnet(
    *,
    n_channels: int,
    n_times: int,
    n_classes: int,
    temporal_filters: int,
    depth_multiplier: int,
    pointwise_filters: int,
    kernel_length: int,
    dropout: float,
) -> Any:
    """Build an EEGNet-style network adapted from the local reference models."""

    torch, nn = _import_torch()
    padding = kernel_length // 2
    depthwise_length = max(8, kernel_length // 4)

    class CompactEEGNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            spatial_filters = temporal_filters * depth_multiplier
            self.features = nn.Sequential(
                nn.Conv2d(
                    1,
                    temporal_filters,
                    kernel_size=(1, kernel_length),
                    padding=(0, padding),
                    bias=False,
                ),
                nn.BatchNorm2d(temporal_filters),
                nn.Conv2d(
                    temporal_filters,
                    spatial_filters,
                    kernel_size=(n_channels, 1),
                    groups=temporal_filters,
                    bias=False,
                ),
                nn.BatchNorm2d(spatial_filters),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 4)),
                nn.Dropout(dropout),
                nn.Conv2d(
                    spatial_filters,
                    spatial_filters,
                    kernel_size=(1, depthwise_length),
                    padding=(0, depthwise_length // 2),
                    groups=spatial_filters,
                    bias=False,
                ),
                nn.Conv2d(
                    spatial_filters,
                    pointwise_filters,
                    kernel_size=(1, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(pointwise_filters),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 8)),
                nn.Dropout(dropout),
                nn.AdaptiveAvgPool2d((1, 8)),
            )
            self.classifier = nn.Linear(pointwise_filters * 8, n_classes)

        def forward(self, values: Any) -> Any:
            features = self.features(values.unsqueeze(1))
            return self.classifier(torch.flatten(features, start_dim=1))

    return CompactEEGNet()


class TorchEEGNetClassifier(ClassifierMixin, BaseEstimator):
    """A joblib-serializable sklearn-style binary EEGNet classifier."""

    def __init__(
        self,
        *,
        n_channels: int = 2,
        n_times: int = 300,
        temporal_filters: int = 8,
        depth_multiplier: int = 2,
        pointwise_filters: int = 16,
        kernel_length: int = 64,
        dropout: float = 0.5,
        epochs: int = 24,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        validation_fraction: float = 0.15,
        patience: int = 5,
        random_state: int = 20260728,
        device: str = "auto",
    ) -> None:
        self.n_channels = n_channels
        self.n_times = n_times
        self.temporal_filters = temporal_filters
        self.depth_multiplier = depth_multiplier
        self.pointwise_filters = pointwise_filters
        self.kernel_length = kernel_length
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.random_state = random_state
        self.device = device

    def _validate_inputs(
        self,
        features: np.ndarray,
        targets: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        values = np.asarray(features, dtype=np.float32)
        expected = int(self.n_channels * self.n_times)
        if values.ndim != 2 or values.shape[1] != expected:
            raise ValueError(
                f"EEGNet 期望二维输入 (epochs, {expected})，实际为 {values.shape}。"
            )
        values = values.reshape(-1, self.n_channels, self.n_times)
        if not np.isfinite(values).all():
            raise ValueError("EEGNet 输入包含非有限值。")
        if targets is None:
            return values, None
        labels = np.asarray(targets, dtype=np.int64)
        if labels.ndim != 1 or len(labels) != len(values):
            raise ValueError("EEGNet 标签形状与输入不一致。")
        return values, labels

    def _new_model(self, n_classes: int) -> Any:
        return _build_eegnet(
            n_channels=self.n_channels,
            n_times=self.n_times,
            n_classes=n_classes,
            temporal_filters=self.temporal_filters,
            depth_multiplier=self.depth_multiplier,
            pointwise_filters=self.pointwise_filters,
            kernel_length=self.kernel_length,
            dropout=self.dropout,
        )

    def _training_device(self, torch: Any) -> Any:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
    ) -> "TorchEEGNetClassifier":
        torch, nn = _import_torch()
        values, labels = self._validate_inputs(features, targets)
        assert labels is not None
        classes = np.unique(labels)
        if not np.array_equal(classes, np.array([0, 1])):
            raise ValueError(
                f"当前 P300 EEGNet 要求标签恰好为 [0, 1]，实际为 {classes.tolist()}。"
            )
        self.classes_ = classes
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        train_indices, validation_indices = train_test_split(
            np.arange(len(values)),
            test_size=self.validation_fraction,
            stratify=labels,
            random_state=self.random_state,
        )
        training_values = values[train_indices]
        self.channel_mean_ = training_values.mean(
            axis=(0, 2), keepdims=True
        ).astype(np.float32)
        self.channel_scale_ = training_values.std(
            axis=(0, 2), keepdims=True
        ).astype(np.float32)
        self.channel_scale_ = np.maximum(self.channel_scale_, 1e-6)
        normalized = (values - self.channel_mean_) / self.channel_scale_

        tensor_values = torch.from_numpy(normalized)
        tensor_labels = torch.from_numpy(labels)
        train_dataset = torch.utils.data.TensorDataset(
            tensor_values[train_indices],
            tensor_labels[train_indices],
        )
        validation_dataset = torch.utils.data.TensorDataset(
            tensor_values[validation_indices],
            tensor_labels[validation_indices],
        )
        generator = torch.Generator()
        generator.manual_seed(self.random_state)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        validation_loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        device = self._training_device(torch)
        model = self._new_model(len(classes)).to(device)
        counts = np.bincount(labels[train_indices], minlength=len(classes))
        class_weights = len(train_indices) / (
            len(classes) * np.maximum(counts, 1)
        )
        criterion = nn.CrossEntropyLoss(
            weight=torch.as_tensor(
                class_weights,
                dtype=torch.float32,
                device=device,
            )
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        stale_epochs = 0
        self.training_history_ = []
        for epoch in range(1, self.epochs + 1):
            model.train()
            train_loss = 0.0
            train_count = 0
            for batch_values, batch_labels in train_loader:
                batch_values = batch_values.to(device)
                batch_labels = batch_labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_values)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.detach().cpu()) * len(batch_values)
                train_count += len(batch_values)

            model.eval()
            validation_loss = 0.0
            validation_count = 0
            with torch.no_grad():
                for batch_values, batch_labels in validation_loader:
                    batch_values = batch_values.to(device)
                    batch_labels = batch_labels.to(device)
                    loss = criterion(model(batch_values), batch_labels)
                    validation_loss += (
                        float(loss.detach().cpu()) * len(batch_values)
                    )
                    validation_count += len(batch_values)
            train_mean = train_loss / max(train_count, 1)
            validation_mean = validation_loss / max(validation_count, 1)
            self.training_history_.append(
                {
                    "epoch": epoch,
                    "train_loss": train_mean,
                    "validation_loss": validation_mean,
                }
            )
            if validation_mean < best_loss - 1e-5:
                best_loss = validation_mean
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break

        if best_state is None:
            raise RuntimeError("EEGNet 没有生成有效训练状态。")
        model.load_state_dict(best_state)
        self.model_ = model.to("cpu").eval()
        self.n_features_in_ = int(self.n_channels * self.n_times)
        self.device_used_ = str(device)
        self.best_validation_loss_ = float(best_loss)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        torch, _nn = _import_torch()
        if not hasattr(self, "model_"):
            raise RuntimeError("EEGNet 尚未训练。")
        values, _labels = self._validate_inputs(features)
        normalized = (values - self.channel_mean_) / self.channel_scale_
        batches: list[np.ndarray] = []
        self.model_.eval()
        with torch.no_grad():
            for offset in range(0, len(normalized), self.batch_size):
                batch = torch.from_numpy(
                    normalized[offset : offset + self.batch_size]
                )
                probabilities = torch.softmax(self.model_(batch), dim=1)
                batches.append(probabilities.cpu().numpy())
        return np.concatenate(batches, axis=0).astype(np.float64)

    def predict(self, features: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(features)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def __getstate__(self) -> dict[str, Any]:
        """Serialize weights as CPU NumPy arrays, never as CUDA tensors."""

        state = dict(self.__dict__)
        model = state.pop("model_", None)
        if model is not None:
            state["_serialized_state_"] = {
                name: tensor.detach().cpu().numpy()
                for name, tensor in model.state_dict().items()
            }
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        serialized = state.pop("_serialized_state_", None)
        self.__dict__.update(state)
        if serialized is not None:
            torch, _nn = _import_torch()
            model = self._new_model(len(self.classes_))
            model.load_state_dict(
                {
                    name: torch.from_numpy(np.asarray(value))
                    for name, value in serialized.items()
                }
            )
            self.model_ = model.to("cpu").eval()
