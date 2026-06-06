"""Tests for sensitivity analysis (Rosenbaum bounds and E-values)."""

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from scipy import stats

from causal.sensitivity import e_value, plot_sensitivity_curve, rosenbaum_bounds


def _matched_pairs(n_pos: int, n_neg: int) -> pd.DataFrame:
    """Build a matched_data frame with given numbers of +/- pair differences."""
    rows = []
    pid = 0
    for _ in range(n_pos):
        rows += [
            {"pair_id": pid, "treatment": 1, "outcome": 1.0},
            {"pair_id": pid, "treatment": 0, "outcome": 0.0},
        ]
        pid += 1
    for _ in range(n_neg):
        rows += [
            {"pair_id": pid, "treatment": 1, "outcome": 0.0},
            {"pair_id": pid, "treatment": 0, "outcome": 1.0},
        ]
        pid += 1
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# E-value
# --------------------------------------------------------------------------- #
def test_evalue_matches_vanderweele_reference():
    # VanderWeele & Ding (2017): RR=3.9 -> E-value 7.26; lower CI limit 1.8 -> 3.0.
    se = (np.log(3.9) - np.log(1.8)) / stats.norm.ppf(0.975)
    ev = e_value(np.log(3.9), se)
    assert ev.point_estimate_evalue == pytest.approx(7.26, abs=0.01)
    assert ev.confidence_interval_evalue == pytest.approx(3.00, abs=0.01)
    assert ev.observed_risk_ratio == pytest.approx(3.9, abs=1e-6)


def test_evalue_symmetric_for_protective_effect():
    # RR=2 and RR=0.5 are equally far from the null -> same E-value.
    up = e_value(np.log(2.0), 0.1).point_estimate_evalue
    down = e_value(np.log(0.5), 0.1).point_estimate_evalue
    assert up == pytest.approx(down, abs=1e-9)


def test_evalue_ci_crossing_null_is_one():
    ev = e_value(np.log(1.2), 0.5)  # wide interval spanning RR=1
    assert ev.confidence_interval_evalue == 1.0
    assert ev.point_estimate_evalue > 1.0


def test_evalue_null_effect_is_one():
    ev = e_value(0.0, 0.1)
    assert ev.point_estimate_evalue == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Rosenbaum bounds
# --------------------------------------------------------------------------- #
def test_rosenbaum_gamma_one_matches_sign_test():
    md = _matched_pairs(n_pos=14, n_neg=6)
    rb = rosenbaum_bounds(md, "outcome", "treatment", gamma_range=[1.0])
    expected = stats.binom.sf(14 - 1, 20, 0.5)  # one-sided sign test
    assert rb.p_value_upper[0] == pytest.approx(expected, abs=1e-12)
    assert rb.p_value_lower[0] == pytest.approx(rb.p_value_upper[0], abs=1e-12)


def test_rosenbaum_upper_bound_monotonic_nondecreasing():
    md = _matched_pairs(n_pos=18, n_neg=2)
    rb = rosenbaum_bounds(md, "outcome", "treatment")
    assert np.all(np.diff(rb.p_value_upper) >= -1e-12)
    # Lower bound is always <= upper bound.
    assert np.all(rb.p_value_lower <= rb.p_value_upper + 1e-12)


def test_rosenbaum_critical_gamma():
    md = _matched_pairs(n_pos=18, n_neg=2)
    rb = rosenbaum_bounds(md, "outcome", "treatment")
    if rb.critical_gamma is not None:
        idx = int(np.where(rb.gamma == rb.critical_gamma)[0][0])
        assert rb.p_value_upper[idx] > 0.05
        if idx > 0:
            assert rb.p_value_upper[idx - 1] <= 0.05


def test_rosenbaum_requires_pair_column():
    md = pd.DataFrame({"treatment": [1, 0], "outcome": [1.0, 0.0]})
    with pytest.raises(ValueError, match="pair_id"):
        rosenbaum_bounds(md, "outcome", "treatment")


def test_sensitivity_curve_plot_runs():
    md = _matched_pairs(n_pos=15, n_neg=5)
    rb = rosenbaum_bounds(md, "outcome", "treatment")
    assert isinstance(plot_sensitivity_curve(rb), Figure)
