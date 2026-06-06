"""Sensitivity analysis for unmeasured confounding.

Every observational method assumes no unmeasured confounders -- an assumption
that cannot be verified from the data. These tools quantify *how strong* an
unmeasured confounder would have to be to overturn a result:

* :func:`rosenbaum_bounds` -- for matched designs, how large the odds of
  differential treatment assignment (gamma) would need to be before the
  conclusion is no longer significant.
* :func:`e_value` -- the minimum association an unmeasured confounder would need
  with both treatment and outcome to explain away the effect (VanderWeele &
  Ding, 2017).
* :func:`plot_sensitivity_curve` -- visualises the Rosenbaum p-value bounds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from scipy import stats

from causal.utils import EValueResult, RosenbaumResult

__all__ = ["rosenbaum_bounds", "e_value", "plot_sensitivity_curve"]


# --------------------------------------------------------------------------- #
# Rosenbaum bounds
# --------------------------------------------------------------------------- #
def rosenbaum_bounds(
    matched_data: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    gamma_range: list[float] | None = None,
    pair_col: str = "pair_id",
) -> RosenbaumResult:
    """Rosenbaum sensitivity bounds for a matched study (sign-test based).

    For each matched pair the difference ``outcome[treated] - outcome[control]``
    is computed. Under an unmeasured confounder of strength ``gamma`` (the odds
    ratio of treatment between two matched units), the probability a treated
    unit shows the larger outcome lies in ``[1/(1+gamma), gamma/(1+gamma)]``,
    yielding lower/upper bounds on the one-sided p-value.

    * ``gamma = 1`` reproduces the standard matched sign-test p-value (lower and
      upper bounds coincide).
    * As ``gamma`` grows the upper bound is non-decreasing; the gamma at which it
      crosses 0.05 is how much hidden bias would overturn significance.

    ``matched_data`` must contain a pair identifier (``pair_col``) with one
    treated and one control row per pair -- exactly the frame produced by
    :meth:`PropensityScoreMatching.match`.
    """
    if pair_col not in matched_data.columns:
        raise ValueError(
            f"matched_data must contain a '{pair_col}' column identifying "
            "matched pairs (e.g. from PropensityScoreMatching.match())."
        )
    if gamma_range is None:
        gamma_range = [float(round(g, 2)) for g in np.arange(1.0, 3.01, 0.1)]

    # Difference in outcome within each matched pair (treated minus control).
    diffs = []
    for _, pair in matched_data.groupby(pair_col):
        treated = pair.loc[pair[treatment_col] == 1, outcome_col]
        control = pair.loc[pair[treatment_col] == 0, outcome_col]
        if len(treated) == 1 and len(control) == 1:
            diffs.append(float(treated.iloc[0]) - float(control.iloc[0]))
    d = np.array(diffs)

    n_pos = int(np.sum(d > 0))
    n_neg = int(np.sum(d < 0))
    n_eff = n_pos + n_neg  # non-tied pairs
    if n_eff == 0:
        raise ValueError("No discordant matched pairs; cannot compute bounds.")

    # One-sided test in the direction of the observed effect.
    t_stat = max(n_pos, n_neg)

    gammas = np.array(gamma_range, dtype=float)
    p_lower = np.empty_like(gammas)
    p_upper = np.empty_like(gammas)
    for i, g in enumerate(gammas):
        p_plus_hi = g / (1.0 + g)
        p_plus_lo = 1.0 / (1.0 + g)
        # P(X >= t_stat) under each bounding probability.
        p_upper[i] = stats.binom.sf(t_stat - 1, n_eff, p_plus_hi)
        p_lower[i] = stats.binom.sf(t_stat - 1, n_eff, p_plus_lo)

    crossed = np.where(p_upper > 0.05)[0]
    critical_gamma = float(gammas[crossed[0]]) if crossed.size > 0 else None

    return RosenbaumResult(
        gamma=gammas,
        p_value_lower=p_lower,
        p_value_upper=p_upper,
        critical_gamma=critical_gamma,
    )


# --------------------------------------------------------------------------- #
# E-value
# --------------------------------------------------------------------------- #
def _evalue_from_rr(rr: float) -> float:
    """E-value for a risk ratio (VanderWeele & Ding 2017)."""
    if rr < 1.0:
        rr = 1.0 / rr
    if rr <= 1.0:
        return 1.0
    return float(rr + np.sqrt(rr * (rr - 1.0)))


def e_value(
    estimate: float,
    std_error: float,
    null_value: float = 0.0,
) -> EValueResult:
    """E-value for an effect expressed on the log (ratio) scale.

    ``estimate`` and ``std_error`` are on the log scale (e.g. a log risk ratio,
    log hazard ratio, or a regression coefficient), so ``null_value=0`` is the
    null (risk ratio of 1). The observed risk ratio is ``exp(estimate)``.

    Returns the E-value for the point estimate and for the 95% confidence limit
    closest to the null. A higher E-value means a stronger (and so less likely)
    unmeasured confounder would be required to explain away the result. If the
    confidence interval already includes the null, its E-value is 1.

    Reference: VanderWeele & Ding (2017), *Annals of Internal Medicine*.
    """
    rr = float(np.exp(estimate))
    point_evalue = _evalue_from_rr(rr)

    z = stats.norm.ppf(0.975)
    lo_log = estimate - z * std_error
    hi_log = estimate + z * std_error
    rr_lo = float(np.exp(lo_log))
    rr_hi = float(np.exp(hi_log))

    null_rr = float(np.exp(null_value))
    if rr >= null_rr:
        # Effect above the null: the relevant limit is the lower one.
        ci_evalue = 1.0 if rr_lo <= null_rr else _evalue_from_rr(rr_lo)
    else:
        # Effect below the null: the relevant limit is the upper one.
        ci_evalue = 1.0 if rr_hi >= null_rr else _evalue_from_rr(rr_hi)

    return EValueResult(
        point_estimate_evalue=point_evalue,
        confidence_interval_evalue=ci_evalue,
        observed_risk_ratio=rr,
    )


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def plot_sensitivity_curve(rosenbaum_result: RosenbaumResult) -> Figure:
    """Plot Rosenbaum p-value bounds against gamma."""
    r = rosenbaum_result
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(r.gamma, r.p_value_upper, marker="o", label="upper bound", color="#c44e52")
    ax.plot(r.gamma, r.p_value_lower, marker="o", label="lower bound", color="#4c72b0")
    ax.axhline(0.05, color="k", ls="--", alpha=0.7, label="p = 0.05")
    if r.critical_gamma is not None:
        ax.axvline(r.critical_gamma, color="0.4", ls=":",
                   label=f"critical Γ = {r.critical_gamma:.2f}")
    ax.set_xlabel("Γ (unmeasured-confounding strength)")
    ax.set_ylabel("p-value bound")
    ax.set_title("Rosenbaum sensitivity analysis")
    ax.legend()
    fig.tight_layout()
    return fig
