# causal-inference-clinical

[![CI](https://github.com/cbrackeen05/causal-inference-clinical/actions/workflows/ci.yml/badge.svg)](https://github.com/cbrackeen05/causal-inference-clinical/actions/workflows/ci.yml)

Quasi-experimental and causal-inference methods for settings where a randomized
controlled trial isn't feasible. In the real world, treatments and interventions
are rarely assigned at random — patients who get a new protocol are sicker,
regions that adopt a program are better resourced, products roll out to whole
populations at once. A naive treated-vs-untreated comparison in those settings is
**confounded**, not causal. This library provides the tools to estimate causal
effects rigorously *anyway*, and to be honest about the assumptions each one
requires.

It's framed around a clinical example — estimating the effect of a care-
coordination intervention on 30-day hospital readmissions — but the methods are
general-purpose and apply to any non-randomized evaluation.

## Installation

```bash
git clone https://github.com/cbrackeen05/causal-inference-clinical.git
cd causal-inference-clinical
pip install -e ".[dev]"
```

Requires Python 3.10+. Core dependencies: numpy, pandas, scipy, scikit-learn,
statsmodels, matplotlib.

## Quick start

Recover a treatment effect from confounded observational data with propensity
score matching:

```python
from causal.data.simulate import simulate_observational
from causal.matching import PropensityScoreMatching

df = simulate_observational(n=2000, treatment_effect=-0.05, seed=7)
psm = PropensityScoreMatching(
    covariates=df[["age", "severity", "comorbidities"]],
    treatment=df["treatment"],
    outcome=df["outcome"],
    caliper=0.05,
)
psm.match()                       # balance treated and control on covariates
print(psm.estimate_att().summary())
```

```
Method:   Propensity score matching (ATT)
Estimate: -0.0822 (SE 0.0237)
95% CI:   [-0.1288, -0.0357]
p-value:  0.0005359
N:        450 treated, 450 control
Assumptions:
  - Conditional ignorability (no unmeasured confounders)
  - Common support / overlap between treated and control
  - Correctly specified propensity model
```

The true effect is −0.05; a naive comparison on this data is biased toward zero
(or even the wrong sign) because treated patients are systematically sicker.

## Methods overview

Every estimator returns a `CausalEstimate` (effect, standard error, confidence
interval, p-value, and a plain-English list of the assumptions it relies on).

### Propensity score matching & IPW — `causal.matching`
- **Use when:** you have many treated and control units with rich pre-treatment
  covariates.
- **Key assumptions:** no unmeasured confounders (conditional ignorability) and
  common support (overlap) between groups.
- **Limitation:** can only adjust for *measured* confounders; poor overlap leaves
  treated units unmatched or produces extreme IPW weights.

### Difference-in-Differences — `causal.did`
- **Use when:** you have panel data — multiple units measured before and after a
  treatment that some units receive.
- **Key assumption:** parallel trends (absent treatment, treated and control
  groups would have moved together). Checked here visually and with a
  differential-trend test.
- **Limitation:** invalid if the groups were already diverging pre-treatment;
  inference needs enough treated clusters for the clustered standard errors.

### Synthetic control — `causal.synthetic_control`
- **Use when:** a single (or a few) treated unit(s) with many candidate donor
  units and a long pre-treatment history.
- **Key assumption:** the treated unit's pre-treatment path lies within the
  convex hull of the donors, with no interference and no anticipation.
- **Limitation:** no classical standard error — inference relies on placebo
  (permutation) tests, which have limited power with few donors.

### Sensitivity analysis — `causal.sensitivity`
- **Use when:** you want to quantify robustness to *unmeasured* confounding
  (which no design can rule out from the data alone).
- **Provides:** Rosenbaum bounds for matched studies and the E-value
  (VanderWeele & Ding, 2017).
- **Limitation:** describes how strong a hidden confounder would have to be — it
  cannot prove one doesn't exist.

### Which method when

| Scenario | Recommended method |
|---|---|
| Many treated and control units, rich covariates | Propensity score matching or IPW |
| Panel data, multiple units, parallel trends plausible | Difference-in-differences |
| Single treated unit, multiple control units | Synthetic control |
| Assessing robustness to unmeasured confounding | Rosenbaum bounds / E-value |
| Randomization is feasible | Just run the RCT |

## Case study

[`notebooks/case_study.ipynb`](notebooks/case_study.ipynb) works through a single
cohesive story: a hospital network's care-coordination intervention, evaluated
end-to-end with all four method families. The naive comparison suggests the
intervention does nothing; matching, IPW, DiD, and synthetic control — resting on
*different* assumptions — all agree it reduces readmissions, and sensitivity
analysis shows the result is reasonably robust to unmeasured confounding. That
triangulation is what makes a causal claim credible without an RCT.

## Why this matters at scale

At companies like Netflix, not every product question can be answered with a
randomized A/B test. Price changes, content launches, and infrastructure rollouts
often affect entire user populations simultaneously, so there is no clean control
group to randomize. These methods provide the statistical framework for drawing
causal conclusions when randomization isn't available — the same framework used
by platform experimentation teams for quasi-experimental analysis. The clinical
framing here is incidental; the machinery is identical.

## Development

```bash
pytest tests/ --cov=causal --cov-report=term-missing   # run the test suite
mypy causal/                                            # type-check
```

CI runs the tests and type-checks across Python 3.10–3.12 on every push and pull
request (see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

## References

- Rosenbaum & Rubin (1983). *The central role of the propensity score in
  observational studies for causal effects.* Biometrika 70(1), 41–55.
- Abadie & Gardeazabal (2003). *The economic costs of conflict: A case study of
  the Basque Country.* American Economic Review 93(1), 113–132. (synthetic control)
- VanderWeele & Ding (2017). *Sensitivity analysis in observational research:
  introducing the E-value.* Annals of Internal Medicine 167(4), 268–274.
- Angrist & Pischke (2009). *Mostly Harmless Econometrics.* Princeton University
  Press. (difference-in-differences)
```
