"""causal-inference-clinical: quasi-experimental and causal inference methods.

Estimators for settings where randomized controlled trials are not feasible:
propensity-score matching and IPW, difference-in-differences, synthetic control,
and sensitivity analysis for unmeasured confounding.
"""

from __future__ import annotations

from causal.utils import (
    CausalEstimate,
    EValueResult,
    EventStudyResult,
    MatchingResult,
    ParallelTrendsTest,
    PlaceboTestResult,
    RosenbaumResult,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "CausalEstimate",
    "MatchingResult",
    "ParallelTrendsTest",
    "EventStudyResult",
    "PlaceboTestResult",
    "RosenbaumResult",
    "EValueResult",
]
