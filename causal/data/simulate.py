"""Synthetic data generators for tests and the case-study notebook.

Three scenarios, all framed around a clinical outcome (30-day readmission):

* :func:`simulate_rct` -- a clean randomized trial, the gold standard against
  which observational estimators are validated.
* :func:`simulate_observational` -- cross-sectional data with confounding, where
  a naive comparison is biased but matching / IPW should recover the truth.
* :func:`simulate_panel` -- long-format panel data with parallel pre-trends, for
  difference-in-differences and synthetic control.

All generators take a ``seed`` and use :func:`numpy.random.default_rng` so that
results are reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["simulate_rct", "simulate_observational", "simulate_panel"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _patient_covariates(
    rng: np.random.Generator, n: int
) -> dict[str, np.ndarray]:
    """Shared pre-treatment patient covariates used by the cross-sectional sims."""
    age = rng.normal(60.0, 15.0, n)
    severity = np.clip(rng.normal(5.0, 2.0, n), 0.0, None)
    comorbidities = rng.poisson(2.0, n).astype(float)
    return {"age": age, "severity": severity, "comorbidities": comorbidities}


def simulate_rct(
    n: int = 1000,
    treatment_effect: float = 0.05,
    baseline_rate: float = 0.20,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate a clean randomized controlled trial.

    Treatment is assigned independently of every covariate, so a naive
    difference in mean outcomes recovers ``treatment_effect`` in expectation.
    Used as the gold standard to validate that observational methods work.

    Returns a DataFrame with columns ``age, severity, comorbidities,
    treatment, outcome`` (outcome is binary: 1 = readmitted within 30 days).
    """
    rng = np.random.default_rng(seed)
    cov = _patient_covariates(rng, n)

    # Randomized assignment -- independent of covariates.
    treatment = rng.binomial(1, 0.5, n)

    # Outcome depends on treatment and (balanced) covariates. Because treatment
    # is random, the covariate terms do not bias the naive comparison.
    p = (
        baseline_rate
        + treatment_effect * treatment
        + 0.002 * (cov["age"] - 60.0)
        + 0.010 * (cov["severity"] - 5.0)
    )
    p = np.clip(p, 0.01, 0.99)
    outcome = rng.binomial(1, p)

    return pd.DataFrame(
        {
            "age": cov["age"],
            "severity": cov["severity"],
            "comorbidities": cov["comorbidities"],
            "treatment": treatment,
            "outcome": outcome,
        }
    )


def simulate_observational(
    n: int = 1000,
    treatment_effect: float = 0.05,
    confounding_strength: float = 2.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate observational data with confounding by indication.

    Disease ``severity`` drives *both* the probability of treatment and the
    outcome, so treated and control patients differ systematically. A naive
    comparison is biased; propensity-score matching / IPW should recover
    ``treatment_effect`` by balancing severity (and the other covariates).

    ``confounding_strength`` scales how strongly severity pushes patients into
    treatment (0 reproduces an RCT-like assignment).

    Returns a DataFrame with the same columns as :func:`simulate_rct`.
    """
    rng = np.random.default_rng(seed)
    cov = _patient_covariates(rng, n)

    severity_z = (cov["severity"] - 5.0) / 2.0
    age_z = (cov["age"] - 60.0) / 15.0

    # Sicker (and somewhat older) patients are more likely to be treated.
    logit_p = confounding_strength * severity_z + 0.5 * age_z
    treatment = rng.binomial(1, _sigmoid(logit_p))

    # Severity also raises the readmission rate -> it is a confounder.
    p = (
        0.20
        + treatment_effect * treatment
        + 0.06 * severity_z
        + 0.02 * age_z
    )
    p = np.clip(p, 0.01, 0.99)
    outcome = rng.binomial(1, p)

    return pd.DataFrame(
        {
            "age": cov["age"],
            "severity": cov["severity"],
            "comorbidities": cov["comorbidities"],
            "treatment": treatment,
            "outcome": outcome,
        }
    )


def simulate_panel(
    n_units: int = 50,
    n_periods: int = 20,
    treatment_unit: int = 0,
    treatment_period: int = 10,
    treatment_effect: float = -0.08,
    n_treated_units: int = 1,
    treated_trend: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate long-format panel data with parallel pre-trends.

    The data-generating process is two-way additive::

        outcome[i, t] = unit_effect[i] + time_effect[t]
                        + treatment_effect * 1{treated_i and t >= treatment_period}
                        + noise

    A common ``time_effect`` makes the treated and control trajectories parallel
    before treatment (so difference-in-differences is valid), and the treated
    unit's level sits inside the range of the donor units (so synthetic control
    can reconstruct it).

    Parameters
    ----------
    treatment_unit : index of the first treated unit.
    n_treated_units : how many units are treated (>=1). Units
        ``treatment_unit .. treatment_unit + n_treated_units - 1`` receive
        treatment. Use 1 for synthetic control, several for DiD.
    treated_trend : if non-zero, adds a treated-group-specific linear time trend,
        deliberately *violating* parallel trends (useful for negative tests).

    Returns a DataFrame with columns ``unit, time, outcome, treatment`` where
    ``treatment`` is 1 for treated units in *all* periods (the post indicator is
    derived from ``treatment_period``).
    """
    rng = np.random.default_rng(seed)

    base_rate = 0.20
    unit_effect = base_rate + rng.normal(0.0, 0.03, n_units)
    # Put the treated unit(s) at the centre of the donor distribution so a
    # convex combination of donors can match them (needed for synthetic control).
    treated_ids = [
        (treatment_unit + k) % n_units for k in range(n_treated_units)
    ]
    donor_mean = float(
        np.mean([unit_effect[i] for i in range(n_units) if i not in treated_ids])
    )
    for i in treated_ids:
        unit_effect[i] = donor_mean

    # Common time effect (a smooth random walk) -> parallel trends.
    time_effect = np.cumsum(rng.normal(0.0, 0.01, n_periods))

    rows = []
    treated_set = set(treated_ids)
    for i in range(n_units):
        is_treated = i in treated_set
        for t in range(n_periods):
            post = t >= treatment_period
            y = unit_effect[i] + time_effect[t] + rng.normal(0.0, 0.01)
            if is_treated and post:
                y += treatment_effect
            if is_treated and treated_trend != 0.0:
                y += treated_trend * t
            rows.append(
                {
                    "unit": i,
                    "time": t,
                    "outcome": y,
                    "treatment": int(is_treated),
                }
            )

    return pd.DataFrame(rows)
