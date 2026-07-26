from __future__ import annotations

import pytest

from bsense_classifier.app import parse_confidence_threshold


@pytest.mark.parametrize("value", [0.30, 0.60, 0.95, "0.75"])
def test_parse_confidence_threshold_accepts_supported_values(value: object) -> None:
    assert parse_confidence_threshold(value) == float(value)


@pytest.mark.parametrize(
    "value",
    [0.29, 0.96, float("-inf"), float("inf"), float("nan"), "invalid"],
)
def test_parse_confidence_threshold_rejects_unsupported_values(value: object) -> None:
    with pytest.raises(ValueError):
        parse_confidence_threshold(value)
