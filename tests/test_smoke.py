"""Smoke test: the package imports and the shared result type is usable."""

import causal
from causal import CausalEstimate
from causal.utils import (
    logit,
    normal_confidence_interval,
    standardized_mean_difference,
)


def test_package_has_version():
    assert isinstance(causal.__version__, str)
    assert causal.__version__


def test_causal_estimate_constructs_and_summarizes():
    est = CausalEstimate(
        method="smoke",
        estimate=0.05,
        std_error=0.01,
        confidence_interval=(0.03, 0.07),
        p_value=0.001,
        n_treated=100,
        n_control=100,
        assumptions=["no unmeasured confounding"],
    )
    assert est.estimate == 0.05
    summary = est.summary()
    assert "smoke" in summary
    assert "no unmeasured confounding" in summary


def test_helpers_basic_behaviour():
    # logit(0.5) == 0
    assert abs(logit([0.5])[0]) < 1e-9
    # identical groups -> zero SMD
    assert standardized_mean_difference([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    lo, hi = normal_confidence_interval(0.0, 1.0)
    assert lo < 0 < hi
