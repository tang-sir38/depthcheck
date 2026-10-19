"""The verdict rule, pinned at its boundaries.

These tests exist so that a later refactor cannot quietly move a threshold. If
one of them fails, the protocol changed; that is a decision for the authors, not
for a commit.
"""

from __future__ import annotations

import pytest

from depthcheck.decompose import (
    AMBIGUOUS,
    CRITERIA,
    FAIL,
    PASS,
    Criteria,
    decompose,
    prediction_band,
    verdict_of,
)


def test_thresholds_are_the_frozen_values() -> None:
    assert CRITERIA.implied_tol == 0.02
    assert CRITERIA.band_tol == 0.02
    assert CRITERIA.equivalence_margin == 0.05


def test_pass_requires_small_implied_and_residual_outside_equivalence() -> None:
    assert verdict_of(0.30, 0.0199, (-0.01, 0.03), (0.25, 0.35)) == PASS
    assert verdict_of(-0.30, -0.0199, (-0.03, 0.01), (-0.35, -0.25)) == PASS


def test_implied_at_the_tolerance_is_not_pass() -> None:
    assert verdict_of(0.30, 0.0200, (-0.01, 0.03), (0.25, 0.35)) == AMBIGUOUS
    assert verdict_of(0.30, 0.0201, (-0.01, 0.03), (0.25, 0.35)) == AMBIGUOUS


def test_residual_ci_touching_the_margin_is_not_pass() -> None:
    # CI starts exactly at 0.05: the band is closed at the margin.
    assert verdict_of(0.30, 0.0, (-0.01, 0.01), (0.05, 0.20)) == AMBIGUOUS
    # CI starts strictly above 0.05: PASS.
    assert verdict_of(0.30, 0.0, (-0.01, 0.01), (0.0501, 0.20)) == PASS


def test_observed_inside_prediction_band_is_fail() -> None:
    assert verdict_of(0.01, 0.0, (-0.01, 0.01), (-0.01, 0.03)) == FAIL
    assert verdict_of(0.019, 0.0, (-0.01, 0.01), (-0.01, 0.03)) == FAIL


def test_observed_outside_band_without_pass_conditions_is_ambiguous() -> None:
    assert verdict_of(0.40, 0.05, (-0.01, 0.11), (0.30, 0.45)) == AMBIGUOUS


def test_prediction_band_widens_with_uncertainty_and_tolerance() -> None:
    lo, hi = prediction_band(0.10, (-0.05, 0.25))
    assert lo == pytest.approx(-0.05)
    assert hi == pytest.approx(0.25)
    lo, hi = prediction_band(0.10, (0.095, 0.105))
    assert lo == pytest.approx(0.08)
    assert hi == pytest.approx(0.12)


def test_custom_criteria_are_honoured() -> None:
    strict = Criteria(implied_tol=0.001, band_tol=0.02, equivalence_margin=0.05)
    assert verdict_of(0.30, 0.01, (-0.01, 0.03), (0.25, 0.35), strict) == AMBIGUOUS


def test_decompose_rejects_inconsistent_residual() -> None:
    from depthcheck.decompose import Decomposition

    with pytest.raises(ValueError):
        # residual must equal observed - implied; 0.5 != 1.0 - 0.2
        Decomposition(observed=1.0, implied=0.2, residual=0.5,
                      implied_ci=(0.1, 0.3), residual_ci=(0.4, 0.6), verdict=PASS)


def test_non_finite_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        verdict_of(float("nan"), 0.0, (-0.1, 0.1), (0.1, 0.2))
    with pytest.raises(ValueError):
        verdict_of(0.5, 0.0, (float("nan"), 0.1), (0.1, 0.2))
