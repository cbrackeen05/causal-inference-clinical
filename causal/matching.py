"""Propensity-score matching and inverse-probability weighting.

When treatment is not randomly assigned, treated and control units differ on
observed covariates. These estimators construct a valid comparison by modelling
the probability of treatment (the *propensity score*) and then either matching
treated units to similar controls or reweighting the sample so that covariates
are balanced.

* :class:`PropensityScoreModel` -- fit/inspect propensity scores and overlap.
* :class:`PropensityScoreMatching` -- nearest-neighbour matching + ATT.
* :class:`InverseProbabilityWeighting` -- IPW estimate of the ATE.
* :func:`plot_balance` -- love plot of covariate balance before/after matching.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from causal.utils import (
    CausalEstimate,
    MatchingResult,
    normal_confidence_interval,
    standardized_mean_difference,
)

__all__ = [
    "PropensityScoreModel",
    "PropensityScoreMatching",
    "InverseProbabilityWeighting",
    "plot_balance",
]

_PS_CLIP = 1e-3


def _as_frame(covariates: pd.DataFrame) -> pd.DataFrame:
    return covariates.reset_index(drop=True).astype(float)


def _fit_propensity(X: pd.DataFrame, treatment: np.ndarray) -> np.ndarray:
    """Fit a logistic propensity model and return clipped propensity scores."""
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logit", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(X.to_numpy(), treatment)
    ps = model.predict_proba(X.to_numpy())[:, 1]
    return np.clip(ps, _PS_CLIP, 1.0 - _PS_CLIP)


# --------------------------------------------------------------------------- #
# Propensity score model
# --------------------------------------------------------------------------- #
class PropensityScoreModel:
    """Logistic-regression model of the probability of treatment."""

    def __init__(self, covariates: pd.DataFrame, treatment: pd.Series):
        self.covariates = _as_frame(covariates)
        self.treatment = treatment.reset_index(drop=True).to_numpy().astype(int)
        self._pipeline: Pipeline | None = None
        self._scores: np.ndarray | None = None

    def fit(self) -> "PropensityScoreModel":
        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                ("logit", LogisticRegression(max_iter=1000)),
            ]
        )
        X = self.covariates.to_numpy()
        pipeline.fit(X, self.treatment)
        self._pipeline = pipeline
        self._scores = np.clip(
            pipeline.predict_proba(X)[:, 1], _PS_CLIP, 1.0 - _PS_CLIP
        )
        return self

    def scores(self) -> np.ndarray:
        if self._scores is None:
            self.fit()
        assert self._scores is not None
        return self._scores

    def plot_overlap(self, bins: int = 30) -> Figure:
        """Plot propensity-score distributions for treated vs control.

        If the two distributions do not overlap, there is no common support and
        matching/weighting cannot work -- this is the key diagnostic to check.
        """
        ps = self.scores()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(ps[self.treatment == 1], bins=bins, alpha=0.6, label="treated", color="#c44e52")
        ax.hist(ps[self.treatment == 0], bins=bins, alpha=0.6, label="control", color="#4c72b0")
        ax.set_xlabel("propensity score")
        ax.set_ylabel("count")
        ax.set_title("Propensity score overlap (common support)")
        ax.legend()
        fig.tight_layout()
        return fig


# --------------------------------------------------------------------------- #
# Propensity score matching
# --------------------------------------------------------------------------- #
class PropensityScoreMatching:
    """Nearest-neighbour propensity-score matching with a caliper."""

    def __init__(
        self,
        covariates: pd.DataFrame,
        treatment: pd.Series,
        outcome: pd.Series,
        caliper: float = 0.05,
        replacement: bool = False,
    ):
        self.covariates = _as_frame(covariates)
        self.treatment = treatment.reset_index(drop=True).to_numpy().astype(int)
        self.outcome = outcome.reset_index(drop=True).to_numpy().astype(float)
        self.caliper = caliper
        self.replacement = replacement
        self.propensity = _fit_propensity(self.covariates, self.treatment)
        self._result: MatchingResult | None = None
        # Per matched pair: indices and outcomes, set during match().
        self._pair_treated: list[int] = []
        self._pair_control: list[int] = []

    def match(self) -> MatchingResult:
        ps = self.propensity
        treated_idx = np.where(self.treatment == 1)[0]
        control_idx = list(np.where(self.treatment == 0)[0])

        pair_treated: list[int] = []
        pair_control: list[int] = []
        available = control_idx.copy()

        # Greedy nearest-neighbour matching on the propensity score.
        for t in treated_idx:
            if not available:
                break
            pool = np.array(available)
            dist = np.abs(ps[pool] - ps[t])
            j = int(np.argmin(dist))
            if dist[j] <= self.caliper:
                c = int(pool[j])
                pair_treated.append(int(t))
                pair_control.append(c)
                if not self.replacement:
                    available.remove(c)

        self._pair_treated = pair_treated
        self._pair_control = pair_control

        n_treated = int(len(treated_idx))
        n_matched = len(pair_treated)
        n_unmatched = n_treated - n_matched

        # Balance: SMD per covariate before (full sample) and after (matched).
        smds: dict[str, dict[str, float]] = {}
        full_t = self.treatment == 1
        full_c = self.treatment == 0
        for col in self.covariates.columns:
            x = self.covariates[col].to_numpy()
            before = standardized_mean_difference(x[full_t], x[full_c])
            if n_matched > 0:
                after = standardized_mean_difference(
                    x[pair_treated], x[pair_control]
                )
            else:
                after = float("nan")
            smds[col] = {"before": before, "after": after}

        before_mean = np.mean([abs(v["before"]) for v in smds.values()])
        after_mean = (
            np.mean([abs(v["after"]) for v in smds.values()])
            if n_matched > 0
            else float("inf")
        )
        balance_improved = bool(after_mean < before_mean)

        matched_data = self._build_matched_frame(pair_treated, pair_control)
        result = MatchingResult(
            matched_data=matched_data,
            n_treated=n_treated,
            n_matched=n_matched,
            n_unmatched=n_unmatched,
            standardized_mean_differences=smds,
            balance_improved=balance_improved,
        )
        self._result = result
        return result

    def _build_matched_frame(
        self, pair_treated: list[int], pair_control: list[int]
    ) -> pd.DataFrame:
        rows = []
        for pair_id, (t, c) in enumerate(zip(pair_treated, pair_control)):
            for role, idx in (("treated", t), ("control", c)):
                row = self.covariates.iloc[idx].to_dict()
                row.update(
                    {
                        "treatment": int(self.treatment[idx]),
                        "outcome": float(self.outcome[idx]),
                        "propensity": float(self.propensity[idx]),
                        "pair_id": pair_id,
                        "role": role,
                    }
                )
                rows.append(row)
        return pd.DataFrame(rows)

    def estimate_att(self) -> CausalEstimate:
        """Average Treatment effect on the Treated from the matched sample."""
        if self._result is None:
            self.match()
        if not self._pair_treated:
            raise ValueError(
                "No treated units could be matched within the caliper "
                f"({self.caliper}). Check overlap with PropensityScoreModel."
            )

        y_t = self.outcome[self._pair_treated]
        y_c = self.outcome[self._pair_control]
        diffs = y_t - y_c
        est = float(np.mean(diffs))
        n = len(diffs)
        se = float(np.std(diffs, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        if se and not np.isnan(se) and se > 0:
            z = est / se
            p = float(2 * (1 - stats.norm.cdf(abs(z))))
            ci = normal_confidence_interval(est, se)
        else:
            p = float("nan")
            ci = (float("nan"), float("nan"))

        return CausalEstimate(
            method="Propensity score matching (ATT)",
            estimate=est,
            std_error=se,
            confidence_interval=ci,
            p_value=p,
            n_treated=n,
            n_control=len(set(self._pair_control)),
            assumptions=[
                "Conditional ignorability (no unmeasured confounders)",
                "Common support / overlap between treated and control",
                "Correctly specified propensity model",
            ],
        )


# --------------------------------------------------------------------------- #
# Inverse probability weighting
# --------------------------------------------------------------------------- #
class InverseProbabilityWeighting:
    """Inverse-probability-of-treatment weighting estimate of the ATE."""

    def __init__(
        self,
        covariates: pd.DataFrame,
        treatment: pd.Series,
        outcome: pd.Series,
    ):
        self.covariates = _as_frame(covariates)
        self.treatment = treatment.reset_index(drop=True).to_numpy().astype(int)
        self.outcome = outcome.reset_index(drop=True).to_numpy().astype(float)

    def _ate_once(
        self, X: pd.DataFrame, t: np.ndarray, y: np.ndarray, stabilized: bool
    ) -> tuple[float, np.ndarray]:
        ps = _fit_propensity(X, t)
        if stabilized:
            p_treat = t.mean()
            w = t * (p_treat / ps) + (1 - t) * ((1 - p_treat) / (1 - ps))
        else:
            w = t / ps + (1 - t) / (1 - ps)
        mu1 = np.sum(w * t * y) / np.sum(w * t)
        mu0 = np.sum(w * (1 - t) * y) / np.sum(w * (1 - t))
        return float(mu1 - mu0), w

    def weights(self, stabilized: bool = True) -> np.ndarray:
        _, w = self._ate_once(
            self.covariates, self.treatment, self.outcome, stabilized
        )
        return w

    def estimate_ate(
        self, stabilized: bool = True, n_boot: int = 300, seed: int = 0
    ) -> CausalEstimate:
        """Estimate the ATE; standard error/CI via the bootstrap.

        ``stabilized=True`` uses stabilized weights (recommended) to reduce
        variance from extreme propensity scores.
        """
        est, _ = self._ate_once(
            self.covariates, self.treatment, self.outcome, stabilized
        )

        rng = np.random.default_rng(seed)
        n = len(self.treatment)
        boot = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            Xb = self.covariates.iloc[idx]
            boot[b], _ = self._ate_once(
                Xb, self.treatment[idx], self.outcome[idx], stabilized
            )
        se = float(np.std(boot, ddof=1))
        ci = normal_confidence_interval(est, se)
        z = est / se if se > 0 else float("nan")
        p = float(2 * (1 - stats.norm.cdf(abs(z)))) if se > 0 else float("nan")

        n_treated = int(self.treatment.sum())
        return CausalEstimate(
            method="Inverse probability weighting (ATE)"
            + (" [stabilized]" if stabilized else ""),
            estimate=est,
            std_error=se,
            confidence_interval=ci,
            p_value=p,
            n_treated=n_treated,
            n_control=n - n_treated,
            assumptions=[
                "Conditional ignorability (no unmeasured confounders)",
                "Positivity (every unit has non-zero chance of each arm)",
                "Correctly specified propensity model",
            ],
        )

    def plot_weight_distribution(self, stabilized: bool = True, bins: int = 40) -> Figure:
        """Plot the IPW weight distribution; long tails flag poor overlap."""
        w = self.weights(stabilized)
        t = self.treatment
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(w[t == 1], bins=bins, alpha=0.6, label="treated", color="#c44e52")
        ax.hist(w[t == 0], bins=bins, alpha=0.6, label="control", color="#4c72b0")
        ax.set_xlabel("IPW weight")
        ax.set_ylabel("count")
        ax.set_title(
            f"Weight distribution ({'stabilized' if stabilized else 'unstabilized'})"
        )
        ax.legend()
        fig.tight_layout()
        return fig


# --------------------------------------------------------------------------- #
# Love plot
# --------------------------------------------------------------------------- #
def plot_balance(matching_result: MatchingResult, threshold: float = 0.1) -> Figure:
    """Love plot: absolute standardized mean difference before/after matching.

    Each covariate gets a dot for |SMD| before and after; the dashed line marks
    the conventional adequate-balance threshold (0.1).
    """
    smds = matching_result.standardized_mean_differences
    covs = list(smds.keys())
    before = [abs(smds[c]["before"]) for c in covs]
    after = [abs(smds[c]["after"]) for c in covs]
    y = np.arange(len(covs))

    fig, ax = plt.subplots(figsize=(7, 0.5 * len(covs) + 2))
    ax.scatter(before, y, label="before matching", facecolors="none", edgecolors="#c44e52", s=70)
    ax.scatter(after, y, label="after matching", color="#4c72b0", s=70)
    ax.axvline(threshold, color="k", ls="--", alpha=0.7, label=f"threshold = {threshold}")
    ax.set_yticks(y)
    ax.set_yticklabels(covs)
    ax.set_xlabel("|standardized mean difference|")
    ax.set_title("Covariate balance (love plot)")
    ax.legend()
    fig.tight_layout()
    return fig
