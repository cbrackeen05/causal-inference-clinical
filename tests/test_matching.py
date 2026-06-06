"""Tests for propensity-score matching and inverse-probability weighting."""

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from causal.data.simulate import simulate_observational
from causal.matching import (
    InverseProbabilityWeighting,
    PropensityScoreMatching,
    PropensityScoreModel,
    plot_balance,
)

TRUTH = 0.05
COVS = ["age", "severity", "comorbidities"]


@pytest.fixture
def obs_data():
    df = simulate_observational(n=3000, treatment_effect=TRUTH, confounding_strength=2.0, seed=42)
    return df[COVS], df["treatment"], df["outcome"], df


def _naive(df: pd.DataFrame) -> float:
    return float(df.loc[df.treatment == 1, "outcome"].mean() - df.loc[df.treatment == 0, "outcome"].mean())


# --------------------------------------------------------------------------- #
# PropensityScoreModel
# --------------------------------------------------------------------------- #
def test_propensity_scores_in_unit_interval(obs_data):
    cov, t, _y, _ = obs_data
    ps = PropensityScoreModel(cov, t).fit().scores()
    assert ps.shape == (len(t),)
    assert np.all((ps > 0) & (ps < 1))
    # Sicker patients are more likely treated -> treated have higher mean PS.
    assert ps[t.to_numpy() == 1].mean() > ps[t.to_numpy() == 0].mean()


def test_overlap_plot_runs(obs_data):
    cov, t, _y, _ = obs_data
    fig = PropensityScoreModel(cov, t).fit().plot_overlap()
    assert isinstance(fig, Figure)


# --------------------------------------------------------------------------- #
# PropensityScoreMatching
# --------------------------------------------------------------------------- #
def test_psm_recovers_truth_and_improves_balance(obs_data):
    cov, t, y, df = obs_data
    psm = PropensityScoreMatching(cov, t, y, caliper=0.05)
    result = psm.match()
    att = psm.estimate_att()

    # Recovers the true effect within the confidence interval...
    lo, hi = att.confidence_interval
    assert lo <= TRUTH <= hi
    # ...while the biased naive estimate falls outside that interval.
    assert _naive(df) > hi
    # Matching improved covariate balance overall.
    assert result.balance_improved
    # Severity was the main confounder and is now well balanced.
    assert abs(result.standardized_mean_differences["severity"]["after"]) < 0.1


def test_matching_result_counts_consistent(obs_data):
    cov, t, y, _ = obs_data
    result = PropensityScoreMatching(cov, t, y, caliper=0.05).match()
    assert result.n_matched + result.n_unmatched == result.n_treated
    assert result.n_matched > 0
    # matched_data holds two rows (treated + control) per matched pair.
    assert len(result.matched_data) == 2 * result.n_matched


def test_matching_with_replacement_runs(obs_data):
    cov, t, y, _ = obs_data
    result = PropensityScoreMatching(cov, t, y, caliper=0.05, replacement=True).match()
    assert result.n_matched > 0


def test_love_plot_runs(obs_data):
    cov, t, y, _ = obs_data
    result = PropensityScoreMatching(cov, t, y, caliper=0.05).match()
    assert isinstance(plot_balance(result), Figure)


def test_no_common_support_raises():
    # Perfectly separated groups: no controls within any reasonable caliper.
    rng = np.random.default_rng(0)
    n = 100
    severity = np.concatenate([np.full(n, 10.0), np.full(n, 0.0)])
    noise = rng.normal(0, 0.1, 2 * n)
    cov = pd.DataFrame({"severity": severity, "noise": noise})
    treatment = pd.Series([1] * n + [0] * n)
    outcome = pd.Series(rng.binomial(1, 0.2, 2 * n).astype(float))

    psm = PropensityScoreMatching(cov, treatment, outcome, caliper=0.05)
    result = psm.match()
    assert result.n_matched == 0
    with pytest.raises(ValueError, match="caliper"):
        psm.estimate_att()


# --------------------------------------------------------------------------- #
# InverseProbabilityWeighting
# --------------------------------------------------------------------------- #
def test_ipw_recovers_truth(obs_data):
    cov, t, y, _ = obs_data
    ate = InverseProbabilityWeighting(cov, t, y).estimate_ate(stabilized=True, n_boot=200, seed=0)
    lo, hi = ate.confidence_interval
    assert lo <= TRUTH <= hi
    assert ate.std_error > 0


def test_ipw_close_to_psm(obs_data):
    cov, t, y, _ = obs_data
    att = PropensityScoreMatching(cov, t, y, caliper=0.05).estimate_att()
    ate = InverseProbabilityWeighting(cov, t, y).estimate_ate(n_boot=200, seed=0)
    # Two valid methods on the same data should broadly agree.
    assert abs(att.estimate - ate.estimate) < 0.03


def test_stabilized_weights_have_smaller_spread(obs_data):
    cov, t, y, _ = obs_data
    ipw = InverseProbabilityWeighting(cov, t, y)
    assert ipw.weights(stabilized=True).std() < ipw.weights(stabilized=False).std()


def test_weight_distribution_plot_runs(obs_data):
    cov, t, y, _ = obs_data
    fig = InverseProbabilityWeighting(cov, t, y).plot_weight_distribution()
    assert isinstance(fig, Figure)
