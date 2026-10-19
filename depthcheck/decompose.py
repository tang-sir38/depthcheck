"""The pre-registered verdict.

PLAN v1.3 section 5 fixes three verdicts and their criteria *before* any result
is seen. The wording leaves two quantities implicit -- what "prediction band"
means and what "CI excludes 0.05" means -- so this module fixes an operational
reading once, in code, and never branches on the data:

``PASS``
    ``|implied| < 0.02`` **and** the 95% CI of the residual lies entirely
    outside the equivalence band ``[-0.05, +0.05]``. The depth-computable term
    is negligible and the remaining contrast is confidently larger than the
    smallest effect size used in the prevalence analysis (E3). A noise contrast
    may be discussed, subject to the identifiability caveats.
``FAIL``
    The observed contrast lies inside the depth-implied prediction band,
    ``[min(implied_ci_lo, implied - 0.02), max(implied_ci_hi, implied + 0.02)]``.
    Depth alone can reproduce the observation; it cannot be read as a noise
    contrast.
``AMBIGUOUS``
    Neither condition holds. Report both components and do not interpret the
    contrast as noise.

Evaluation order is PASS, then FAIL, then AMBIGUOUS. The tolerances are module
constants and are pinned by ``tests/test_verdict_boundaries.py``; changing them
is a protocol amendment, not a refactor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

PASS = "PASS"
FAIL = "FAIL"
AMBIGUOUS = "AMBIGUOUS"

VERDICTS = (PASS, FAIL, AMBIGUOUS)

CRITERIA_VERSION = "PLAN_v1.3_S5"

PLAN_TEXT = (
    "PASS: |implied| < 0.02 and the residual 95% CI excludes 0.05 -> a noise "
    "contrast may be discussed; "
    "FAIL: the observed Delta F falls inside the prediction band -> it may not; "
    "AMBIGUOUS: otherwise."
)

OPERATIONAL_TEXT = (
    "Operational reading fixed in depthcheck 1.0.0: the prediction band is "
    "[min(implied_ci_lo, implied - 0.02), max(implied_ci_hi, implied + 0.02)]; "
    "'CI excludes 0.05' means the residual 95% CI lies entirely outside "
    "[-0.05, +0.05]. Evaluation order is PASS, FAIL, AMBIGUOUS."
)


def _finite(x: float, name: str) -> float:
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v!r}")
    return v


def _pair(pair: Tuple[float, float], name: str) -> Tuple[float, float]:
    if len(pair) != 2:
        raise ValueError(f"{name} must be a (lo, hi) pair")
    lo, hi = float(pair[0]), float(pair[1])
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError(f"{name} contains non-finite bounds: {(lo, hi)!r}")
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


@dataclass(frozen=True)
class Criteria:
    """Frozen thresholds. Do not tune these after looking at a result."""

    implied_tol: float = 0.02
    band_tol: float = 0.02
    equivalence_margin: float = 0.05

    def as_dict(self) -> Dict[str, float]:
        return {
            "implied_tol": self.implied_tol,
            "band_tol": self.band_tol,
            "equivalence_margin": self.equivalence_margin,
        }


CRITERIA = Criteria()


def prediction_band(
    implied: float,
    implied_ci: Tuple[float, float],
    criteria: Criteria = CRITERIA,
) -> Tuple[float, float]:
    """Depth-implied interval, widened by its own uncertainty and the tolerance."""
    lo, hi = _pair(implied_ci, "implied_ci")
    return (
        min(lo, implied - criteria.band_tol),
        max(hi, implied + criteria.band_tol),
    )


def verdict_of(
    observed: float,
    implied: float,
    implied_ci: Tuple[float, float],
    residual_ci: Tuple[float, float],
    criteria: Criteria = CRITERIA,
) -> str:
    """Return ``PASS`` / ``FAIL`` / ``AMBIGUOUS`` for one contrast."""
    observed = _finite(observed, "observed")
    implied = _finite(implied, "implied")
    r_lo, r_hi = _pair(residual_ci, "residual_ci")
    band_lo, band_hi = prediction_band(implied, implied_ci, criteria)

    outside_equivalence = (r_lo > criteria.equivalence_margin) or (
        r_hi < -criteria.equivalence_margin
    )
    if abs(implied) < criteria.implied_tol and outside_equivalence:
        return PASS
    if band_lo <= observed <= band_hi:
        return FAIL
    return AMBIGUOUS


READINGS = {
    PASS: (
        "The depth-computable component is below 0.02 Fano units and the residual "
        "is confidently outside the +/-0.05 equivalence band: the contrast is a "
        "candidate noise difference. Identifiability caveats still apply -- depth "
        "itself is not attributable to technical or biological causes."
    ),
    FAIL: (
        "The observed contrast lies inside the band that the depth term alone "
        "predicts. It cannot be read as a difference in transcriptional noise."
    ),
    AMBIGUOUS: (
        "Neither condition is met. Report the observed, implied and residual "
        "components; do not describe the contrast as a noise difference."
    ),
}


@dataclass
class Decomposition:
    """Result of one depth decomposition.

    ``residual`` is ``observed - implied`` by definition and is checked against
    the two inputs on construction.
    """

    observed: float
    implied: float
    residual: float
    implied_ci: Tuple[float, float]
    residual_ci: Tuple[float, float]
    verdict: str
    observed_ci: Tuple[float, float] = (float("nan"), float("nan"))
    p_exact: float = float("nan")
    permutation_method: str = "not-run"
    n_assignments: int = 0
    criteria: Criteria = CRITERIA
    context: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(f"unknown verdict {self.verdict!r}")
        _finite(self.observed, "observed")
        _finite(self.implied, "implied")
        _finite(self.residual, "residual")
        if abs(self.residual - (self.observed - self.implied)) > 1e-9:
            raise ValueError("residual must equal observed - implied")
        self.implied_ci = _pair(self.implied_ci, "implied_ci")
        self.residual_ci = _pair(self.residual_ci, "residual_ci")
        if math.isfinite(self.p_exact) and not (0.0 <= self.p_exact <= 1.0):
            raise ValueError("p_exact must lie in [0, 1]")

    @property
    def reading(self) -> str:
        return READINGS[self.verdict]

    @property
    def band(self) -> Tuple[float, float]:
        return prediction_band(self.implied, self.implied_ci, self.criteria)

    def as_dict(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "observed": self.observed,
            "implied": self.implied,
            "residual": self.residual,
            "implied_ci_lo": self.implied_ci[0],
            "implied_ci_hi": self.implied_ci[1],
            "residual_ci_lo": self.residual_ci[0],
            "residual_ci_hi": self.residual_ci[1],
            "observed_ci_lo": self.observed_ci[0],
            "observed_ci_hi": self.observed_ci[1],
            "p_exact": self.p_exact,
            "permutation_method": self.permutation_method,
            "n_assignments": self.n_assignments,
            "band_lo": self.band[0],
            "band_hi": self.band[1],
            "verdict": self.verdict,
            "criteria_version": CRITERIA_VERSION,
            "criteria": self.criteria.as_dict(),
        }
        out.update(self.context)
        return out


def decompose(
    observed: float,
    implied: float,
    implied_ci: Tuple[float, float],
    residual_ci: Tuple[float, float],
    observed_ci: Tuple[float, float] = (float("nan"), float("nan")),
    p_exact: float = float("nan"),
    permutation_method: str = "not-run",
    n_assignments: int = 0,
    criteria: Criteria = CRITERIA,
    context: Optional[Dict[str, object]] = None,
) -> Decomposition:
    """Build a :class:`Decomposition` and apply the frozen verdict rule."""
    observed = _finite(observed, "observed")
    implied = _finite(implied, "implied")
    residual = _finite(observed - implied, "residual")
    verdict = verdict_of(observed, implied, implied_ci, residual_ci, criteria)
    return Decomposition(
        observed=observed,
        implied=implied,
        residual=residual,
        implied_ci=implied_ci,
        residual_ci=residual_ci,
        verdict=verdict,
        observed_ci=observed_ci,
        p_exact=p_exact,
        permutation_method=permutation_method,
        n_assignments=n_assignments,
        criteria=criteria,
        context=dict(context or {}),
    )
