"""Synthetic Control method for single-treated-unit settings.

When only one unit is treated (one region, one hospital), difference-in-
differences lacks power. Synthetic control instead builds a weighted average of
untreated "donor" units that tracks the treated unit's *pre-treatment*
trajectory, then reads the treatment effect off the post-treatment gap between
the treated unit and its synthetic counterpart.

Donor weights are non-negative and sum to one (a convex combination), found by
minimising pre-treatment prediction error. Because there is no classical
standard error, inference is done with placebo (permutation) tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from causal.utils import CausalEstimate, PlaceboTestResult

__all__ = ["SyntheticControl"]


def _solve_weights(treated: np.ndarray, donors: np.ndarray) -> np.ndarray:
    """Find convex weights w (>=0, sum 1) minimising ||treated - donors @ w||^2.

    ``treated`` is a length-m feature vector; ``donors`` is (m x J).
    """
    n_donors = donors.shape[1]

    def objective(w: np.ndarray) -> float:
        resid = treated - donors @ w
        return float(resid @ resid)

    w0 = np.full(n_donors, 1.0 / n_donors)
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0)] * n_donors
    res = minimize(
        objective, w0, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    w = np.clip(res.x, 0.0, None)
    total = w.sum()
    return w / total if total > 0 else w0


class SyntheticControl:
    """Synthetic control estimator for a single treated unit."""

    def __init__(
        self,
        data: pd.DataFrame,
        unit_col: str,
        time_col: str,
        outcome_col: str,
        treated_unit: object,
        treatment_time: int,
        predictor_cols: list[str] | None = None,
    ):
        self.unit_col = unit_col
        self.time_col = time_col
        self.outcome_col = outcome_col
        self.treated_unit = treated_unit
        self.treatment_time = treatment_time
        self.predictor_cols = predictor_cols

        wide = data.pivot_table(index=time_col, columns=unit_col, values=outcome_col)
        self._wide = wide.sort_index()
        self._times = self._wide.index.to_numpy()
        self._pre_mask = self._times < treatment_time
        if self._pre_mask.sum() < 2:
            raise ValueError("Need at least 2 pre-treatment periods.")
        self.donor_units = [u for u in self._wide.columns if u != treated_unit]
        if len(self.donor_units) < 1:
            raise ValueError("Need at least one donor unit.")

        # Optional time-invariant predictors: pre-treatment mean per unit.
        self._predictor_means: dict[object, np.ndarray] | None = None
        if predictor_cols:
            pre = data[data[time_col] < treatment_time]
            means = pre.groupby(unit_col)[predictor_cols].mean()
            self._predictor_means = {u: means.loc[u].to_numpy() for u in means.index}

        self._weights: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    def _features(self, unit: object) -> np.ndarray:
        feats = self._wide.loc[self._pre_mask, unit].to_numpy()
        if self._predictor_means is not None:
            feats = np.concatenate([feats, self._predictor_means[unit]])
        return feats

    def _matching_matrices(
        self, treated_unit: object, donor_units: list[object]
    ) -> tuple[np.ndarray, np.ndarray]:
        treated = self._features(treated_unit)
        donors = np.column_stack([self._features(u) for u in donor_units])
        # Standardize each feature (row) to comparable scale across all units.
        allvals = np.column_stack([treated, donors])
        scale = allvals.std(axis=1)
        scale[scale == 0] = 1.0
        return treated / scale, donors / scale.reshape(-1, 1)

    # ------------------------------------------------------------------ #
    def fit(self) -> "SyntheticControl":
        treated_f, donors_f = self._matching_matrices(self.treated_unit, self.donor_units)
        self._weights = _solve_weights(treated_f, donors_f)
        return self

    def _require_fit(self) -> np.ndarray:
        if self._weights is None:
            self.fit()
        assert self._weights is not None
        return self._weights

    def weights(self) -> pd.Series:
        w = self._require_fit()
        return pd.Series(w, index=pd.Index(self.donor_units, name=self.unit_col), name="weight")

    def synthetic(self) -> pd.Series:
        """The synthetic control outcome series over all periods."""
        w = self._require_fit()
        synth = self._wide[self.donor_units].to_numpy() @ w
        return pd.Series(synth, index=self._wide.index, name="synthetic")

    def pre_treatment_fit(self) -> float:
        """Root mean squared prediction error over the pre-treatment period."""
        gap = self._wide[self.treated_unit] - self.synthetic()
        pre = gap.to_numpy()[self._pre_mask]
        return float(np.sqrt(np.mean(pre**2)))

    # ------------------------------------------------------------------ #
    def estimate(self) -> CausalEstimate:
        """Average post-treatment gap (ATT). Inference is via placebo_tests()."""
        gap = self._wide[self.treated_unit] - self.synthetic()
        post = gap.to_numpy()[~self._pre_mask]
        att = float(np.mean(post))
        return CausalEstimate(
            method="Synthetic control (post-treatment ATT)",
            estimate=att,
            std_error=float("nan"),
            confidence_interval=(float("nan"), float("nan")),
            p_value=float("nan"),
            n_treated=1,
            n_control=len(self.donor_units),
            assumptions=[
                "The treated unit's pre-treatment path lies in the convex hull of donors",
                "No interference: donors are unaffected by the treatment",
                "No anticipation before treatment_time",
                "Inference requires placebo tests (no classical standard error)",
            ],
        )

    def effects(self) -> pd.Series:
        """Per-period treatment effect (treated minus synthetic)."""
        return (self._wide[self.treated_unit] - self.synthetic()).rename("effect")

    # ------------------------------------------------------------------ #
    def plot(self) -> Figure:
        """Actual treated unit vs synthetic control over time."""
        synth = self.synthetic()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(self._times, self._wide[self.treated_unit].to_numpy(),
                marker="o", label=f"treated ({self.treated_unit})", color="#c44e52")
        ax.plot(self._times, synth.to_numpy(), ls="--",
                label="synthetic control", color="#4c72b0")
        ax.axvline(self.treatment_time, color="k", ls=":", alpha=0.7, label="treatment start")
        ax.set_xlabel(self.time_col)
        ax.set_ylabel(self.outcome_col)
        ax.set_title("Synthetic control fit")
        ax.legend()
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------ #
    def _rmspe_ratio(self, treated_unit: object, donor_units: list[object]) -> tuple[float, np.ndarray]:
        """Post/pre RMSPE ratio and the gap path for one (placebo) unit."""
        treated_f, donors_f = self._matching_matrices(treated_unit, donor_units)
        w = _solve_weights(treated_f, donors_f)
        synth = self._wide[donor_units].to_numpy() @ w
        gap = self._wide[treated_unit].to_numpy() - synth
        pre_rmspe = np.sqrt(np.mean(gap[self._pre_mask] ** 2))
        post_rmspe = np.sqrt(np.mean(gap[~self._pre_mask] ** 2))
        ratio = post_rmspe / pre_rmspe if pre_rmspe > 0 else np.inf
        return float(ratio), gap

    def placebo_tests(self) -> PlaceboTestResult:
        """Run synthetic control on each donor as a placebo for inference.

        If the treated unit's post/pre RMSPE ratio is extreme relative to the
        placebo distribution, the estimated effect is unlikely to be noise.
        """
        treated_ratio, treated_gap = self._rmspe_ratio(self.treated_unit, self.donor_units)

        placebo_ratios: list[float] = []
        placebo_gaps: dict[object, np.ndarray] = {}
        for u in self.donor_units:
            others = [self.treated_unit] + [d for d in self.donor_units if d != u]
            ratio, gap = self._rmspe_ratio(u, others)
            placebo_ratios.append(ratio)
            placebo_gaps[u] = gap

        n_ge = sum(1 for r in placebo_ratios if r >= treated_ratio)
        p_value = (n_ge + 1) / (len(placebo_ratios) + 1)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        for u, gap in placebo_gaps.items():
            ax.plot(self._times, gap, color="0.7", lw=0.8, alpha=0.7)
        ax.plot(self._times, treated_gap, color="#c44e52", lw=2,
                label=f"treated ({self.treated_unit})")
        ax.axhline(0, color="k", lw=1)
        ax.axvline(self.treatment_time, color="k", ls=":", alpha=0.7, label="treatment start")
        ax.set_xlabel(self.time_col)
        ax.set_ylabel(f"gap in {self.outcome_col}")
        ax.set_title("Placebo gaps (treated vs donors)")
        ax.legend()
        fig.tight_layout()

        return PlaceboTestResult(
            treated_post_rmspe_ratio=treated_ratio,
            placebo_ratios=placebo_ratios,
            p_value=p_value,
            figure=fig,
        )
