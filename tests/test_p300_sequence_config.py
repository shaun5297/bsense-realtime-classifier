from collections import Counter
from unittest.mock import Mock

import pytest

from bsense_classifier.robot_app import RobotControlApp, build_parser, gate_is_relaxed
from bsense_classifier.p300_control import P300_COMMANDS


def test_default_preserves_ten_sequences():
    assert build_parser().parse_args([]).sequences_per_trial == 10


@pytest.mark.parametrize("count", [2, 4, 6, 10])
def test_configured_plan_balances_every_command(count):
    app = RobotControlApp.__new__(RobotControlApp)
    app.sequences_per_trial = count
    app._previous_flash = None
    plan = app._build_flash_plan()
    assert len(plan) == count*6
    assert Counter(command for _,_,command in plan) == Counter({c:count for c in P300_COMMANDS})
    assert all(a[2] != b[2] for a,b in zip(plan, plan[1:]))


@pytest.mark.parametrize("count", [0, 1, 11])
def test_invalid_sequence_count_rejected(count):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--sequences-per-trial", str(count)])


@pytest.mark.parametrize("count,minimum", [(2,2),(4,4),(6,6),(10,6)])
def test_short_sequences_require_all_flashes_and_preserve_thresholds(monkeypatch, count, minimum):
    app = _stub_app(count)
    factory = Mock()
    monkeypatch.setattr("bsense_classifier.robot_app.P300ControlEngine", factory)
    app._connect_eeg()
    config = factory.call_args.args[1]
    assert config.minimum_flashes_per_command == minimum
    assert config.confidence_threshold == .55
    assert config.margin_threshold == .08
    assert config.minimum_quality_ratio == .8


def _stub_app(sequences: int = 6, confidence: float = .55, margin: float = .08,
              quality: float = .8) -> RobotControlApp:
    """Bypass __init__ so the engine wiring can be checked without a display."""
    app = RobotControlApp.__new__(RobotControlApp)
    app.runtime = Mock()
    app.engine = None
    app.sequences_per_trial = sequences
    app.confidence_threshold = confidence
    app.margin_threshold = margin
    app.min_quality_ratio = quality
    app.stream_name = Mock()
    app.stream_name.get.return_value = "EEG"
    app.decoder_status = Mock()
    app.connect_button = Mock()
    return app


def test_default_gate_thresholds_are_unchanged():
    args = build_parser().parse_args([])
    assert (args.confidence_threshold, args.margin_threshold, args.min_quality_ratio) == (.55, .08, .8)


def test_custom_thresholds_reach_the_engine(monkeypatch):
    app = _stub_app(sequences=6, confidence=.50, margin=.0, quality=.5)
    factory = Mock()
    monkeypatch.setattr("bsense_classifier.robot_app.P300ControlEngine", factory)
    app._connect_eeg()
    config = factory.call_args.args[1]
    assert config.confidence_threshold == .5
    assert config.margin_threshold == .0
    assert config.minimum_quality_ratio == .5


@pytest.mark.parametrize("value", ["-0.1", "1.1", "nan"])
def test_thresholds_outside_unit_range_are_rejected(value):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--confidence-threshold", value])


def test_relaxed_thresholds_are_flagged():
    """The operator must be told when a looser-than-default gate is in force."""
    assert gate_is_relaxed(.55, .08, .8) is False
    assert gate_is_relaxed(.50, .08, .8) is True
    assert gate_is_relaxed(.55, .00, .8) is True
    assert gate_is_relaxed(.55, .08, .5) is True
    assert gate_is_relaxed(.70, .20, .9) is False
