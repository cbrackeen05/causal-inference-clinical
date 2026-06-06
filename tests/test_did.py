"""Tests for the difference-in-differences estimator."""

import numpy as np
import pytest
from matplotlib.figure import Figure

from causal.data.simulate import simulate_panel
from causal.did import DifferenceInDifferences

EFFECT = -0.08


def _did(df, treatment_time=10):
    return DifferenceInDifferences(df, "unit", "time", "outcome", "treatment", treatment_time)


@pytest.fixture
def clean_panel():
    return simulate_panel(
        n_units=50, n_periods=20, treatment_period=10,
        treatment_effect=EFFECT, n_treated_units=12, seed=42,
    )


@pytest.fixture
def nonparallel_panel():
    return simulate_panel(
        n_units=50, n_periods=20, treatment_period=10,
        treatment_effect=EFFECT, n_treated_units=12, treated_trend=0.01, seed=42,
    )


# --------------------------------------------------------------------------- #
def test_did_recovers_true_effect(clean_panel):
    est = _did(clean_panel).estimate()
    lo, hi = est.confidence_interval
    assert lo <= EFFECT <= hi
    assert est.p_value < 0.05
    assert est.n_treated == 12
    assert est.n_control == 38


def test_did_with_covariates_runs(clean_panel):
    df = clean_panel.copy()
    rng = np.random.default_rng(0)
    df["x"] = rng.normal(size=len(df))
    est = _did(df).estimate(covariates=["x"])
    lo, hi = est.confidence_interval
    assert lo <= EFFECT <= hi


def test_parallel_trends_passes_on_clean_data(clean_panel):
    pt = _did(clean_panel).test_parallel_trends()
    assert pt.passes
    assert pt.p_value > 0.05
    assert "parallel trends" in pt.interpretation.lower()


def test_parallel_trends_fails_when_nonparallel(nonparallel_panel):
    pt = _did(nonparallel_panel).test_parallel_trends()
    assert not pt.passes
    assert pt.p_value < 0.05


def test_parallel_trends_needs_two_pre_periods(clean_panel):
    with pytest.raises(ValueError, match="pre-treatment period"):
        _did(clean_panel).test_parallel_trends(n_pre_periods=1)


def test_event_study_pre_periods_near_zero(clean_panel):
    es = _did(clean_panel).event_study()
    pre = es.coefficients[es.relative_periods < 0]
    post = es.coefficients[es.relative_periods >= 0]
    # Pre-treatment leads should be ~0 (validates parallel trends visually).
    assert np.max(np.abs(pre)) < 0.02
    # Post-treatment effect recovers the truth on average.
    assert abs(post.mean() - EFFECT) < 0.02
    # Reference period (-1) is included and pinned to zero.
    assert 0.0 in es.coefficients[es.relative_periods == -1]


def test_event_study_arrays_aligned(clean_panel):
    es = _did(clean_panel).event_study()
    n = len(es.relative_periods)
    assert len(es.coefficients) == n
    assert len(es.conf_lower) == n
    assert len(es.conf_upper) == n
    assert np.all(np.diff(es.relative_periods) > 0)  # sorted, unique


def test_plots_run(clean_panel):
    did = _did(clean_panel)
    assert isinstance(did.plot_parallel_trends(), Figure)
    assert isinstance(did.event_study().figure, Figure)
