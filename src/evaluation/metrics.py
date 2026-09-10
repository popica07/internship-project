"""The locked evaluation metrics (CLAUDE.md, Evaluation Protocol).

Primary:     PR-AUC, Precision@k          (k = 10, 50, 100)
Ranking:     NDCG@k, MAP, MRR, Recall@k
Calibration: Brier score, Expected Calibration Error
Secondary:   ROC-AUC - reported, never used to choose a model

ROC-AUC is deliberately demoted. At the L1 base rate (1,695 KEV of 387,494
CVEs, ~1:228) a model can look strong on ROC-AUC while being useless at the top
of the ranking, which is the only part a security engineer ever reads.

Everything here takes (y_true, y_score) and returns plain floats, so Study B/C
can call them uniformly across model families.
"""

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

DEFAULT_K_VALUES: tuple[int, ...] = (10, 50, 100)


def _top_k_indices(y_score: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(y_score))
    return np.argsort(y_score)[::-1][:k]


def precision_at_k(y_true, y_score, k: int) -> float:
    """Share of the top-k that are positive - the analyst's-eye metric."""
    y_true = np.asarray(y_true)
    idx = _top_k_indices(np.asarray(y_score), k)
    return float(y_true[idx].mean()) if len(idx) else float("nan")


def recall_at_k(y_true, y_score, k: int) -> float:
    """Share of all positives captured in the top-k."""
    y_true = np.asarray(y_true)
    total_positives = y_true.sum()
    if total_positives == 0:
        return float("nan")
    idx = _top_k_indices(np.asarray(y_score), k)
    return float(y_true[idx].sum() / total_positives)


def dcg_at_k(relevance: np.ndarray, k: int) -> float:
    relevance = relevance[:k]
    discounts = np.log2(np.arange(2, len(relevance) + 2))
    return float((relevance / discounts).sum())


def ndcg_at_k(y_true, y_score, k: int) -> float:
    """Normalised discounted cumulative gain - position-weighted ranking quality."""
    y_true = np.asarray(y_true, dtype=float)
    ranked = y_true[np.argsort(np.asarray(y_score))[::-1]]
    ideal = np.sort(y_true)[::-1]
    ideal_dcg = dcg_at_k(ideal, k)
    return dcg_at_k(ranked, k) / ideal_dcg if ideal_dcg > 0 else float("nan")


def mean_reciprocal_rank(y_true, y_score) -> float:
    """1 / rank of the first positive."""
    y_true = np.asarray(y_true)
    ranked = y_true[np.argsort(np.asarray(y_score))[::-1]]
    hits = np.flatnonzero(ranked)
    return float(1.0 / (hits[0] + 1)) if len(hits) else 0.0


def average_precision(y_true, y_score) -> float:
    """MAP over a single ranking == average precision."""
    y_true = np.asarray(y_true)
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, np.asarray(y_score)))


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """ECE with equal-width bins: mean |confidence - accuracy| gap."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (y_prob > lower) & (y_prob <= upper)
        if not in_bin.any():
            continue
        ece += in_bin.mean() * abs(y_true[in_bin].mean() - y_prob[in_bin].mean())
    return float(ece)


def evaluate(y_true, y_score, k_values: tuple[int, ...] = DEFAULT_K_VALUES, is_probability: bool = True) -> dict:
    """The full locked metric set for one model on one split."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    results = {
        "pr_auc": average_precision(y_true, y_score),  # PRIMARY
        "mrr": mean_reciprocal_rank(y_true, y_score),
        "map": average_precision(y_true, y_score),
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "base_rate": float(y_true.mean()),
    }
    for k in k_values:
        results[f"precision_at_{k}"] = precision_at_k(y_true, y_score, k)  # PRIMARY
        results[f"recall_at_{k}"] = recall_at_k(y_true, y_score, k)
        results[f"ndcg_at_{k}"] = ndcg_at_k(y_true, y_score, k)

    # Secondary only - never a selection criterion (see module docstring).
    results["roc_auc"] = (
        float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan")
    )

    if is_probability:
        results["brier"] = float(brier_score_loss(y_true, np.clip(y_score, 0, 1)))
        results["ece"] = expected_calibration_error(y_true, np.clip(y_score, 0, 1))

    return results
