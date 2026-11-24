"""Bank cost-matrix evaluation (B6).

No real Deutsche Bank cost data exists, so costs are a RATIO swept as a
parameter: C_FN / C_FP = cost of missing an exploited vulnerability relative
to the cost of remediating one that would never have been exploited. Cost is
reported per 1,000 CVEs in units of C_FP, so policies compare directly.
"""

import numpy as np


def expected_cost(y_true, decide, cost_ratio: float) -> float:
    """Cost per 1,000 CVEs (unit = one unnecessary remediation)."""
    y = np.asarray(y_true).astype(bool)
    d = np.asarray(decide).astype(bool)
    false_negatives = np.sum(y & ~d)
    false_positives = np.sum(~y & d)
    return float((cost_ratio * false_negatives + false_positives) / len(y) * 1000)


def best_threshold(y_true, y_prob, cost_ratio: float) -> float:
    """Threshold minimising cost on the given (validation) data."""
    candidates = np.unique(np.concatenate([np.quantile(y_prob, np.linspace(0, 1, 2001)), [np.inf]]))
    costs = [expected_cost(y_true, y_prob >= t, cost_ratio) for t in candidates]
    return float(candidates[int(np.argmin(costs))])


def bayes_threshold(cost_ratio: float) -> float:
    """Optimal threshold for a perfectly calibrated probability: 1 / (1 + ratio)."""
    return 1.0 / (1.0 + cost_ratio)
