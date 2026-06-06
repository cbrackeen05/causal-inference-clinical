"""Difference-in-Differences estimator.

DiD estimates a causal effect by comparing the *change* in outcomes over time
for a treated group against the change for a control group. Identification rests
on the **parallel trends** assumption: absent treatment, the two groups would
have moved in parallel.

The canonical 2x2 regression is::

    outcome = a + b1*treated + b2*post + b3*(treated x post) + e

where ``b3`` (the treated-by-post interaction) is the DiD estimate of the ATT.
Standard errors are clustered at the unit level throughout.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from causal.utils import CausalEstimate, EventStudyResult, ParallelTrendsTest

__all__ = ["DifferenceInDifferences"]


class DifferenceInDifferences:
    """Two-group difference-in-differences with unit-clustered inference."""

    def __init__(
        self,
        data: pd.DataFrame,
        unit_col: str,
        time_col: str,
        outcome_col: str,
        treatment_col: str,
        treatment_time: int,
    ):
        self.unit_col = unit_col
        self.time_col = time_col
        self.outcome_col = outcome_col
        self.treatment_col = treatment_col
        self.treatment_time = treatment_time

        df = data.copy()
        df["_post"] = (df[time_col] >= treatment_time).astype(int)
        df["_treated"] = df[treatment_col].astype(int)
        df["_rel"] = df[time_col] - treatment_time
        self.data = df

    # ------------------------------------------------------------------ #
    def estimate(self, covariates: list[str] | None = None) -> CausalEstimate:
        """Estimate the DiD coefficient (ATT) via OLS with clustered SEs."""
        formula = f"{self.outcome_col} ~ _treated + _post + _treated:_post"
        if covariates:
            formula += " + " + " + ".join(covariates)

        res = smf.ols(formula, data=self.data).fit(
            cov_type="cluster", cov_kwds={"groups": self.data[self.unit_col]}
        )

        term = "_treated:_post"
        est = float(res.params[term])
        se = float(res.bse[term])
        ci_lo, ci_hi = (float(x) for x in res.conf_int().loc[term])
        p = float(res.pvalues[term])

        treated_units = self.data.loc[self.data["_treated"] == 1, self.unit_col].nunique()
        control_units = self.data.loc[self.data["_treated"] == 0, self.unit_col].nunique()

        return CausalEstimate(
            method="Difference-in-differences (ATT)",
            estimate=est,
            std_error=se,
            confidence_interval=(ci_lo, ci_hi),
            p_value=p,
            n_treated=int(treated_units),
            n_control=int(control_units),
            assumptions=[
                "Parallel trends: absent treatment, treated and control move together",
                "No anticipation of treatment before treatment_time",
                "Stable composition of treated/control groups over time",
            ],
        )

    # ------------------------------------------------------------------ #
    def plot_parallel_trends(self) -> Figure:
        """Plot mean outcome over time for treated vs control groups."""
        means = (
            self.data.groupby([self.time_col, "_treated"])[self.outcome_col]
            .mean()
            .unstack()
        )
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(means.index, means[0], marker="o", label="control", color="#4c72b0")
        ax.plot(means.index, means[1], marker="o", label="treated", color="#c44e52")
        ax.axvline(self.treatment_time, color="k", ls="--", alpha=0.7, label="treatment start")
        ax.set_xlabel(self.time_col)
        ax.set_ylabel(f"mean {self.outcome_col}")
        ax.set_title("Parallel trends check")
        ax.legend()
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------ #
    def test_parallel_trends(self, n_pre_periods: int | None = None) -> ParallelTrendsTest:
        """Test for a differential *linear* pre-trend between the groups.

        Fits ``outcome ~ treated * time`` on the pre-treatment data and tests
        whether the treated-by-time interaction (the difference in pre-trend
        slopes) is zero. p > 0.05 -> fail to reject parallel trends.

        A single-coefficient test is used rather than a fully saturated
        per-period interaction: with cluster-robust SEs and a modest number of
        treated units, the joint Wald test is badly anti-conservative (it
        over-rejects), whereas the linear-slope test is well calibrated and
        directly interpretable.
        """
        pre = self.data[self.data[self.time_col] < self.treatment_time].copy()
        if n_pre_periods is not None:
            keep = sorted(pre[self.time_col].unique())[-n_pre_periods:]
            pre = pre[pre[self.time_col].isin(keep)]

        n_periods = pre[self.time_col].nunique()
        if n_periods < 2:
            raise ValueError(
                "Need at least 2 pre-treatment periods to test parallel trends."
            )

        formula = f"{self.outcome_col} ~ _treated * {self.time_col}"
        res = smf.ols(formula, data=pre).fit(
            cov_type="cluster", cov_kwds={"groups": pre[self.unit_col]}
        )

        term = f"_treated:{self.time_col}"
        f_test = res.f_test(f"{term} = 0")
        f_stat = float(np.squeeze(f_test.fvalue))
        p_value = float(np.squeeze(f_test.pvalue))
        passes = p_value > 0.05

        if passes:
            interpretation = (
                f"Fail to reject parallel trends (F={f_stat:.2f}, p={p_value:.3f}). "
                "Pre-treatment trends are statistically indistinguishable; the DiD "
                "assumption is supported."
            )
        else:
            interpretation = (
                f"Reject parallel trends (F={f_stat:.2f}, p={p_value:.3f}). "
                "Pre-treatment trends differ between groups; DiD estimates may be biased."
            )

        return ParallelTrendsTest(
            f_statistic=f_stat,
            p_value=p_value,
            passes=passes,
            interpretation=interpretation,
        )

    # ------------------------------------------------------------------ #
    def event_study(self) -> EventStudyResult:
        """Estimate per-period (lead/lag) effects relative to treatment.

        The period just before treatment (relative period -1) is the omitted
        reference. Pre-treatment coefficients near zero validate parallel trends;
        post-treatment coefficients trace out the dynamic treatment effect.
        """
        rel_term = "C(_rel, Treatment(reference=-1))"
        formula = f"{self.outcome_col} ~ _treated + {rel_term} + _treated:{rel_term}"
        res = smf.ols(formula, data=self.data).fit(
            cov_type="cluster", cov_kwds={"groups": self.data[self.unit_col]}
        )
        conf = res.conf_int()

        pattern = re.compile(r"_treated:" + re.escape(rel_term) + r"\[T\.(-?\d+)\]")
        periods, coefs, lows, highs = [], [], [], []
        for name in res.params.index:
            m = pattern.fullmatch(name)
            if m:
                periods.append(int(m.group(1)))
                coefs.append(float(res.params[name]))
                lows.append(float(conf.loc[name, 0]))
                highs.append(float(conf.loc[name, 1]))

        # Add the reference period (-1) explicitly at zero.
        periods.append(-1)
        coefs.append(0.0)
        lows.append(0.0)
        highs.append(0.0)

        order = np.argsort(periods)
        rel = np.array(periods)[order]
        coef = np.array(coefs)[order]
        lo = np.array(lows)[order]
        hi = np.array(highs)[order]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.errorbar(
            rel, coef, yerr=[coef - lo, hi - coef], fmt="o-", capsize=3, color="#c44e52"
        )
        ax.axhline(0, color="k", lw=1)
        ax.axvline(-0.5, color="k", ls="--", alpha=0.7, label="treatment start")
        ax.set_xlabel("period relative to treatment")
        ax.set_ylabel("estimated effect")
        ax.set_title("Event study")
        ax.legend()
        fig.tight_layout()

        return EventStudyResult(
            relative_periods=rel,
            coefficients=coef,
            conf_lower=lo,
            conf_upper=hi,
            figure=fig,
        )
