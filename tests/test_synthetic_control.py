"""Tests for the synthetic control estimator."""

import numpy as np
import pytest
from matplotlib.figure import Figure

from causal.data.simulate import simulate_panel
from causal.synthetic_control import SyntheticControl

EFFECT = -0.08


@pytest.fixture
def sc():
    df = simulate_panel(
        n_units=20, n_periods=20, treatment_unit=0, treatment_period=10,
        treatment_effect=EFFECT, n_treated_units=1, seed=42,
    )
    return SyntheticControl(df, "unit", "time", "outcome", treated_unit=0, treatment_time=10).fit()


# --------------------------------------------------------------------------- #
def test_weights_are_convex(sc):
    w = sc.weights()
    assert np.isclose(w.sum(), 1.0)
    assert (w >= -1e-9).all()
    assert len(w) == 19  # all donor units


def test_pre_treatment_fit_is_low(sc):
    # On clean simulated data the synthetic control tracks the treated unit well.
    assert sc.pre_treatment_fit() < 0.02


def test_recovers_treatment_effect(sc):
    est = sc.estimate()
    assert abs(est.estimate - EFFECT) < 0.02
    assert est.n_treated == 1
    assert est.n_control == 19
    # No classical SE for synthetic control.
    assert np.isnan(est.std_error)


def test_effects_series_length(sc):
    eff = sc.effects()
    assert len(eff) == 20
    # Pre-treatment effects are near zero; post-treatment near the true effect.
    pre = eff.to_numpy()[:10]
    post = eff.to_numpy()[10:]
    assert np.max(np.abs(pre)) < 0.02
    assert abs(post.mean() - EFFECT) < 0.02


def test_placebo_tests_structure_and_significance(sc):
    pl = sc.placebo_tests()
    assert len(pl.placebo_ratios) == 19
    assert 0.0 < pl.p_value <= 1.0
    # The real treatment effect should stand out from the placebo distribution.
    assert pl.p_value <= 0.10
    assert pl.treated_post_rmspe_ratio > max(pl.placebo_ratios) - 1e-9
    assert isinstance(pl.figure, Figure)


def test_plot_runs(sc):
    assert isinstance(sc.plot(), Figure)


def test_few_donors_edge_case():
    # Only 2 donor units available.
    df = simulate_panel(
        n_units=3, n_periods=20, treatment_unit=0, treatment_period=10,
        treatment_effect=EFFECT, n_treated_units=1, seed=1,
    )
    sc = SyntheticControl(df, "unit", "time", "outcome", treated_unit=0, treatment_time=10).fit()
    w = sc.weights()
    assert len(w) == 2
    assert np.isclose(w.sum(), 1.0)
    assert np.isfinite(sc.estimate().estimate)


def test_requires_two_pre_periods():
    df = simulate_panel(n_units=5, n_periods=20, treatment_period=1, seed=0)
    with pytest.raises(ValueError, match="pre-treatment period"):
        SyntheticControl(df, "unit", "time", "outcome", treated_unit=0, treatment_time=1)
