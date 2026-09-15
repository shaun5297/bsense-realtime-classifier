from __future__ import annotations

import joblib
import numpy as np
import pytest

# EEGNet 需要可选的 deep extra；未安装时跳过，而不是让整个测试套件失败。
pytest.importorskip("torch", reason="EEGNet 需要可选依赖 deep（pip install -e '.[deep]'）")

from bsense_classifier.eegnet_model import TorchEEGNetClassifier


def test_eegnet_probability_shape_and_joblib_roundtrip(tmp_path) -> None:
    generator = np.random.default_rng(20260728)
    features = generator.normal(size=(40, 2 * 64)).astype(np.float32)
    targets = np.tile(np.array([0, 1], dtype=np.int64), 20)
    features[targets == 1, 32:48] += 0.75
    model = TorchEEGNetClassifier(
        n_channels=2,
        n_times=64,
        temporal_filters=2,
        depth_multiplier=1,
        pointwise_filters=2,
        kernel_length=16,
        epochs=2,
        batch_size=16,
        patience=2,
        random_state=7,
        device="cpu",
    )
    model.fit(features, targets)
    expected = model.predict_proba(features[:5])
    assert expected.shape == (5, 2)
    np.testing.assert_allclose(expected.sum(axis=1), 1.0, atol=1e-6)

    path = tmp_path / "eegnet.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    actual = loaded.predict_proba(features[:5])
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
