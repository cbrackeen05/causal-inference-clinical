"""Tests for the synthetic data generators."""

import numpy as np
import pandas as pd

from causal.data.simulate import (
    simulate_observational,
    simulate_panel,
    simulate_rct,
)

CROSS_SECTION_COLUMNS = {"age", "severity", "comorbidities", "treatment", "outcome"}


def _naive_diff(df: pd.DataFrame) -> float:
    treated = df.loc[df.treatment == 1, "outcome"].mean()
    control = df.loc[df.treatment == 0, "outcome"].mean()
    return float(treated - control)


# --------------------------------------------------------------------------- #
# simulate_rct
# --------------------------------------------------------------------------- #
def test_rct_schema_and_types():
    df = simulate_rct(n=500, seed=0)
    assert set(df.columns) == CROSS_SECTION_COLUMNS
    assert len(df) == 500
    assert set(df["treatment"].unique()) <= {0, 1}
    assert set(df["outcome"].unique()) <= {0, 1}


def test_rct_is_balanced_and_unbiased():
    df = simulate_rct(n=5000, treatment_effect=0.05, seed=42)
    # Randomized -> roughly 50/50 assignment and a naive estimate near the truth.
    assert 0.45 < df["treatment"].mean() < 0.55
    assert abs(_naive_diff(df) - 0.05) < 0.03


# --------------------------------------------------------------------------- #
# simulate_observational
# --------------------------------------------------------------------------- #
def test_observational_schema():
    df = simulate_observational(n=500, seed=0)
    assert set(df.columns) == CROSS_SECTION_COLUMNS
    assert len(df) == 500


def test_observational_has_confounding_and_biased_naive_estimate():
    df = simulate_observational(n=4000, treatment_effect=0.05, seed=42)
    # Confounding by indication: treated patients are sicker on average.
    sev_treated = df.loc[df.treatment == 1, "severity"].mean()
    sev_control = df.loc[df.treatment == 0, "severity"].mean()
    assert sev_treated > sev_control + 1.0
    # The naive comparison is biased well above the true effect of 0.05.
    assert _naive_diff(df) - 0.05 > 0.05


def test_rct_naive_is_much_closer_to_truth_than_observational():
    truth = 0.05
    rct_err = abs(_naive_diff(simulate_rct(n=4000, treatment_effect=truth, seed=1)) - truth)
    obs_err = abs(
        _naive_diff(simulate_observational(n=4000, treatment_effect=truth, seed=1)) - truth
    )
    assert rct_err < obs_err


def test_observational_confounding_strength_zero_reduces_bias():
    truth = 0.05
    strong = simulate_observational(n=4000, treatment_effect=truth, confounding_strength=2.0, seed=3)
    none = simulate_observational(n=4000, treatment_effect=truth, confounding_strength=0.0, seed=3)
    assert abs(_naive_diff(none) - truth) < abs(_naive_diff(strong) - truth)


# --------------------------------------------------------------------------- #
# simulate_panel
# --------------------------------------------------------------------------- #
def test_panel_schema_and_grid():
    df = simulate_panel(n_units=30, n_periods=12, seed=0)
    assert list(df.columns) == ["unit", "time", "outcome", "treatment"]
    assert len(df) == 30 * 12
    # Balanced panel: every unit observed in every period.
    assert df.groupby("unit")["time"].nunique().eq(12).all()


def test_panel_single_treated_unit_by_default():
    df = simulate_panel(seed=0)
    assert df.loc[df.treatment == 1, "unit"].nunique() == 1


def test_panel_multiple_treated_units():
    df = simulate_panel(n_units=40, n_treated_units=8, seed=0)
    assert df.loc[df.treatment == 1, "unit"].nunique() == 8
    # Treatment indicator is on for treated units in every period.
    treated_units = df.loc[df.treatment == 1, "unit"].unique()
    for u in treated_units:
        assert (df.loc[df.unit == u, "treatment"] == 1).all()


def test_panel_recovers_did_effect():
    effect = -0.08
    df = simulate_panel(
        n_units=40, n_periods=20, treatment_period=10,
        treatment_effect=effect, n_treated_units=8, seed=42,
    )
    tr, co = df[df.treatment == 1], df[df.treatment == 0]
    did = (
        (tr[tr.time >= 10].outcome.mean() - tr[tr.time < 10].outcome.mean())
        - (co[co.time >= 10].outcome.mean() - co[co.time < 10].outcome.mean())
    )
    assert abs(did - effect) < 0.02


def test_panel_treated_trend_breaks_parallel_trends():
    # With a treated-specific trend, the pre-treatment treated/control gap drifts.
    df = simulate_panel(
        n_units=40, n_periods=20, treatment_period=10,
        n_treated_units=8, treated_trend=0.01, seed=42,
    )
    pre = df[df.time < 10]
    gaps = (
        pre[pre.treatment == 1].groupby("time")["outcome"].mean()
        - pre[pre.treatment == 0].groupby("time")["outcome"].mean()
    )
    # Gap at the last pre-period is clearly larger than at the first.
    assert gaps.iloc[-1] - gaps.iloc[0] > 0.05


# --------------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------------- #
def test_seed_reproducibility():
    a = simulate_rct(n=200, seed=7)
    b = simulate_rct(n=200, seed=7)
    c = simulate_rct(n=200, seed=8)
    pd.testing.assert_frame_equal(a, b)
    assert not np.array_equal(a["outcome"].to_numpy(), c["outcome"].to_numpy())
