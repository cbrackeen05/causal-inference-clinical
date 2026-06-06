"""Shared dataclasses and small numerical helpers used across the library.

Every estimator in this package returns a :class:`CausalEstimate`, and several
return one of the richer result containers defined here. Keeping them in one
place avoids circular imports between the estimator modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import stats

__all__ = [
    "CausalEstimate",
    "MatchingResult",
    "ParallelTrendsTest",
    "EventStudyResult",
    "PlaceboTestResult",
    "RosenbaumResult",
    "EValueResult",
    "logit",
    "standardized_mean_difference",
    "normal_confidence_interval",
]


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass
class CausalEstimate:
    """A single causal effect estimate with inference and stated assumptions.

    Returned by every estimator so that results are directly comparable across
    methods (matching, IPW, DiD, synthetic control).
    """

    method: str
    estimate: float
    std_error: float
    confidence_interval: tuple[float, float]
    p_value: float
    n_treated: int
    n_control: int
    assumptions: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a one-paragraph plain-text summary of the estimate."""
        lo, hi = self.confidence_interval
        lines = [
            f"Method:   {self.method}",
            f"Estimate: {self.estimate:.4f} (SE {self.std_error:.4f})",
            f"95% CI:   [{lo:.4f}, {hi:.4f}]",
            f"p-value:  {self.p_value:.4g}",
            f"N:        {self.n_treated} treated, {self.n_control} control",
        ]
        if self.assumptions:
            lines.append("Assumptions:")
            lines.extend(f"  - {a}" for a in self.assumptions)
        return "\n".join(lines)


@dataclass
class MatchingResult:
    """Output of a matching procedure: the matched sample plus balance diagnostics."""

    matched_data: pd.DataFrame
    n_treated: int
    n_matched: int
    n_unmatched: int
    # covariate -> {"before": smd, "after": smd}
    standardized_mean_differences: dict[str, dict[str, float]]
    balance_improved: bool


@dataclass
class ParallelTrendsTest:
    """Result of a formal pre-trends test for difference-in-differences."""

    f_statistic: float
    p_value: float
    passes: bool  # True if p > 0.05 (fail to reject parallel trends)
    interpretation: str


@dataclass
class EventStudyResult:
    """Per-period (lead/lag) treatment effect estimates around the treatment date."""

    relative_periods: np.ndarray  # period - treatment_time (0 = first treated period)
    coefficients: np.ndarray
    conf_lower: np.ndarray
    conf_upper: np.ndarray
    figure: Optional[Figure] = None


@dataclass
class PlaceboTestResult:
    """Inference for synthetic control via placebo (permutation) tests."""

    treated_post_rmspe_ratio: float  # post/pre RMSPE for the treated unit
    placebo_ratios: list[float]  # same ratio for each placebo (donor) unit
    p_value: float  # fraction of placebos with ratio >= treated
    figure: Optional[Figure] = None


@dataclass
class RosenbaumResult:
    """Rosenbaum sensitivity bounds across a range of gamma values."""

    gamma: np.ndarray
    p_value_lower: np.ndarray
    p_value_upper: np.ndarray
    # Smallest gamma at which the upper-bound p-value crosses 0.05, or None.
    critical_gamma: Optional[float] = None


@dataclass
class EValueResult:
    """E-value for the point estimate and the CI limit closest to the null."""

    point_estimate_evalue: float
    confidence_interval_evalue: float
    observed_risk_ratio: float


# --------------------------------------------------------------------------- #
# Numerical helpers
# --------------------------------------------------------------------------- #
def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Logit transform with clipping to keep values away from 0 and 1."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def standardized_mean_difference(
    treated: np.ndarray, control: np.ndarray
) -> float:
    """Standardized mean difference between two groups for one covariate.

    Uses the pooled standard deviation (average of the two group variances),
    the convention used in love plots for covariate balance.
    """
    treated = np.asarray(treated, dtype=float)
    control = np.asarray(control, dtype=float)
    mean_diff = treated.mean() - control.mean()
    pooled_sd = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2.0)
    if pooled_sd == 0:
        return 0.0
    return float(mean_diff / pooled_sd)


def normal_confidence_interval(
    estimate: float, std_error: float, alpha: float = 0.05
) -> tuple[float, float]:
    """Two-sided normal-approximation confidence interval."""
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    return (estimate - z * std_error, estimate + z * std_error)
