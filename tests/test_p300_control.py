from __future__ import annotations

import numpy as np
import pytest

from bsense_classifier.model_runtime import ArtifactError, InferenceResult
from bsense_classifier.p300_control import (
    P300_COMMANDS,
    P300TargetRuntime,
    P300TrialAggregator,
)


def _accepted_aggregator() -> P300TrialAggregator:
    aggregator = P300TrialAggregator(
        trial_id=7,
        minimum_flashes_per_command=2,
        confidence_threshold=0.55,
        margin_threshold=0.08,
    )
    for command in P300_COMMANDS:
        probability = 0.82 if command == "left" else 0.18
        aggregator.add(command, probability, quality_ok=True)
        aggregator.add(command, probability - 0.02, quality_ok=True)
    return aggregator


def test_trial_aggregator_accepts_clear_winner() -> None:
    decision = _accepted_aggregator().finish()
    assert decision.accepted is True
    assert decision.command == "left"
    assert decision.confidence == pytest.approx(0.81)
    assert decision.margin > 0.5
    assert decision.quality_ratio == 1.0


def test_trial_aggregator_rejects_incomplete_or_ambiguous_trial() -> None:
    aggregator = P300TrialAggregator(
        trial_id=8,
        minimum_flashes_per_command=2,
        confidence_threshold=0.55,
        margin_threshold=0.08,
    )
    for command in P300_COMMANDS:
        aggregator.add(
            command,
            0.6,
            quality_ok=command not in {"forward", "backward"},
        )
    decision = aggregator.finish()
    assert decision.accepted is False
    assert "insufficient_flashes" in decision.rejection_reasons
    assert "below_margin_threshold" in decision.rejection_reasons
    assert "low_signal_quality_ratio" in decision.rejection_reasons


class _StubRuntime:
    task = "m7_p300"
    target_kind = "classification"
    classes = (0, 1)
    label_mapping = {0: "non_target", 1: "target"}
    window_seconds = 1.2
    window_offset_seconds = -0.2
    sfreq = 250.0
    event_locked = True

    def infer(self, _raw_window: np.ndarray) -> InferenceResult:
        return InferenceResult(
            task=self.task,
            target_kind=self.target_kind,
            signal_quality="ok",
            quality_ok=True,
            candidate=1,
            label="target",
            confidence=0.9,
            probabilities=(0.1, 0.9),
            classes=self.classes,
        )


def test_target_runtime_returns_target_probability() -> None:
    runtime = P300TargetRuntime(_StubRuntime())
    probability, quality_ok, reason = runtime.infer_target_probability(
        np.zeros((2, 300))
    )
    assert probability == 0.9
    assert quality_ok is True
    assert reason == "ok"


def test_target_runtime_rejects_wrong_task() -> None:
    runtime = _StubRuntime()
    runtime.task = "m4b_target"
    with pytest.raises(ArtifactError):
        P300TargetRuntime(runtime)
