"""Statistical tests and agreement measures.

Study C will add Friedman / Nemenyi / Wilcoxon here; A3 needs inter-rater
agreement. Cohen's kappa itself comes from scikit-learn - only the bootstrap
interval around it is implemented here, because CLAUDE.md requires that no
result is reported as a single point estimate.
"""

import logging

import numpy as np
from sklearn.metrics import cohen_kappa_score

logger = logging.getLogger(__name__)


def kappa_with_ci(
    rater_a,
    rater_b,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
    weights: str | None = None,
) -> dict:
    """Cohen's kappa with a percentile bootstrap confidence interval.

    Returns kappa, the CI bounds, n, and the bootstrap standard deviation.
    `weights` may be None (nominal), "linear" or "quadratic" for ordinal
    scales.
    """
    a = np.asarray(rater_a)
    b = np.asarray(rater_b)
    n = len(a)
    if n == 0:
        return {"kappa": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0, "std": np.nan}

    kappa = cohen_kappa_score(a, b, weights=weights)

    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        # A resample can be single-class, leaving kappa undefined; skip those.
        if len(set(a[idx])) < 2 and len(set(b[idx])) < 2:
            continue
        samples.append(cohen_kappa_score(a[idx], b[idx], weights=weights))

    if not samples:
        return {"kappa": kappa, "ci_low": np.nan, "ci_high": np.nan, "n": n, "std": np.nan}

    alpha = (1 - confidence) / 2
    return {
        "kappa": float(kappa),
        "ci_low": float(np.quantile(samples, alpha)),
        "ci_high": float(np.quantile(samples, 1 - alpha)),
        "n": int(n),
        "std": float(np.std(samples)),
    }


def interpret_kappa(kappa: float) -> str:
    """Landis & Koch (1977) descriptive bands, for reporting only."""
    if np.isnan(kappa):
        return "undefined"
    if kappa < 0:
        return "worse than chance"
    if kappa < 0.20:
        return "slight"
    if kappa < 0.40:
        return "fair"
    if kappa < 0.60:
        return "moderate"
    if kappa < 0.80:
        return "substantial"
    return "almost perfect"
