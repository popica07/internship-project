"""B2 - KEV membership prediction (binary, strict temporal split).

Design fixed in the Study B pre-registration addendum:
- split T = 2023-01-01; train labels = KEV as of T; test labels = KEV today;
- point-in-time features (src/features.py); test CVEs already in KEV on their
  scoring date are excluded, and every baseline is re-scored on that same set;
- primary model HistGradientBoosting, reference logistic regression, 10 seeds;
- sensitivities: train labels as of today; + NVD reference-tag features.

H1 (pre-registered): a learned model beats CVSS-score ordering on PR-AUC and
Precision@100. Decided by a paired bootstrap over test CVEs of both differences
(PR-AUC and P@100), plus the seed-level comparison.

Usage:  python -m src.models.b2_kev
"""

import logging
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import evaluate
from src.features import FeatureBuilder, load_l1, temporal_task
from src.models.baselines import SEEDS, tie_broken_rank

logger = logging.getLogger(__name__)

SPLIT_DATE = "2023-01-01"
OUTPUT_DIR = Path("outputs")
CACHE_DIR = Path("data/processed/cache")
PREDICTIONS_PATH = CACHE_DIR / "b2_test_predictions.pkl"
REPORT_METRICS = ["pr_auc", "precision_at_10", "precision_at_50", "precision_at_100",
                  "recall_at_100", "ndcg_at_100", "mrr", "roc_auc", "brier", "ece"]
N_BOOTSTRAP = 500


def make_gbm(seed: int) -> HistGradientBoostingClassifier:
    # Fixed capacity, no early stopping: early stopping would carve a RANDOM
    # validation split out of time-ordered training data. max_features < 1 is
    # what makes the seed matter (feature subsampling per split).
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, max_features=0.5, early_stopping=False, random_state=seed,
    )


def make_logreg() -> object:
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=3000),
    )


def build_matrices(use_reference_tags: bool = False) -> dict:
    df = load_l1()
    task = temporal_task(df, SPLIT_DATE)
    builder = FeatureBuilder(use_epss=True, use_reference_tags=use_reference_tags).fit(task["train"])
    task["X_train"] = builder.transform(task["train"], task["train_dates"])
    task["X_test"] = builder.transform(task["test"], task["test_dates"])
    for name in ("X_train", "X_test"):
        values = task[name].to_numpy(float)
        if np.isinf(values).any():
            raise ValueError(f"{name} contains inf")
        # Only EPSS may be missing (no archive / not yet scored); text and CVSS
        # NaNs would mean a broken transform (e.g. a NaN SVD), so fail loudly.
        unexpected = task[name].drop(columns=["epss_pit", "epss_pit_pct", "cvss_score",
                                              "cvss_severity", "cvss_version"]).isna().sum()
        if unexpected.any():
            raise ValueError(f"{name} has NaNs in {list(unexpected[unexpected > 0].index)}")
    return task


def fit_predict(model_name: str, X_train, y_train, X_test, seed: int) -> np.ndarray:
    model = make_gbm(seed) if model_name == "gbm" else make_logreg()
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def baseline_scores(task: dict) -> dict[str, np.ndarray]:
    """Point-in-time baselines on the B2 evaluation set (raw, untied)."""
    test = task["test"]
    v3 = test["cvss_score"].where(test["cvss_version"].astype(str).str.startswith("3"))
    return {
        "cvss_v3": v3.fillna(-np.inf).to_numpy(float),
        "epss_pit": task["X_test"]["epss_pit"].fillna(-np.inf).to_numpy(float),
        "random": np.zeros(len(test)),
        "epss_current": test["epss_score"].fillna(-np.inf).to_numpy(float),  # leaky reference
    }


def _precision_at_k(y, score, k: int = 100) -> float:
    return float(y[np.argsort(-score, kind="stable")[:k]].mean())


def paired_bootstrap(y, score_a, score_b, n: int = N_BOOTSTRAP, seed: int = 0) -> dict:
    """a - b over CVE resamples, for both H1 metrics (PR-AUC and P@100).

    Returns the observed difference, the 95% CI and P(diff <= 0) per metric.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    metrics = {"pr_auc": average_precision_score, "p_at_100": _precision_at_k}
    diffs = {m: [] for m in metrics}
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        for m, f in metrics.items():
            diffs[m].append(f(y[idx], score_a[idx]) - f(y[idx], score_b[idx]))
    out = {}
    for m, f in metrics.items():
        d = np.array(diffs[m])
        out.update({f"{m}_diff": float(f(y, score_a) - f(y, score_b)),
                    f"{m}_ci_low": float(np.quantile(d, 0.025)),
                    f"{m}_ci_high": float(np.quantile(d, 0.975)),
                    f"{m}_p_not_better": float((d <= 0).mean())})
    return out


def run() -> dict:
    from src.tracking import log_experiment

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Building point-in-time features")
    task = build_matrices(use_reference_tags=True)
    tag_columns = ["public_exploit", "has_patch"]
    X_train_tags, X_test_tags = task["X_train"], task["X_test"]
    X_train = X_train_tags.drop(columns=tag_columns)
    X_test = X_test_tags.drop(columns=tag_columns)
    task["X_test"] = X_test
    y_test = task["y_test"].to_numpy()

    variants = {
        # name: (model, X_train, y_train, X_test)
        "gbm": ("gbm", X_train, task["y_train"], X_test),
        "logreg": ("logreg", X_train, task["y_train"], X_test),
        "gbm_labels_today": ("gbm", X_train, task["y_train_today"], X_test),
        "gbm_plus_reference_tags": ("gbm", X_train_tags, task["y_train"], X_test_tags),
    }

    rows, predictions = [], {}
    for name, (model_name, Xtr, ytr, Xte) in variants.items():
        seeds = SEEDS if model_name == "gbm" else [0]  # logreg is deterministic
        for seed in seeds:
            logger.info("B2 %s seed %d", name, seed)
            prob = fit_predict(model_name, Xtr, ytr, Xte, seed)
            predictions[f"{name}__s{seed}"] = prob
            rows.append({"scorer": name, "seed": seed, **evaluate(y_test, prob, is_probability=True)})

    for name, primary in baseline_scores(task).items():
        for seed in SEEDS:
            rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
            score = tie_broken_rank(primary, rng)
            predictions.setdefault(f"{name}__s{seed}", score)
            rows.append({"scorer": name, "seed": seed, **evaluate(y_test, score, is_probability=False)})

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "b2_results_by_seed.csv", index=False)
    grouped = results.groupby("scorer")[REPORT_METRICS]
    summary = (grouped.mean().round(4).astype(str) + " ± " + grouped.std(ddof=0).round(4).astype(str))
    summary = summary.loc[grouped.mean()["pr_auc"].sort_values(ascending=False).index]
    summary.to_csv(OUTPUT_DIR / "b2_summary.csv")

    # H1: seed-averaged GBM vs CVSS v3 (and vs point-in-time EPSS, reported).
    gbm_mean = np.mean([predictions[f"gbm__s{s}"] for s in SEEDS], axis=0)
    cvss_mean = np.mean([predictions[f"cvss_v3__s{s}"] for s in SEEDS], axis=0)
    epss_pit_mean = np.mean([predictions[f"epss_pit__s{s}"] for s in SEEDS], axis=0)
    h1 = {
        "gbm_vs_cvss_v3": paired_bootstrap(y_test, gbm_mean, cvss_mean),
        "gbm_vs_epss_pit": paired_bootstrap(y_test, gbm_mean, epss_pit_mean),
    }
    pd.DataFrame(h1).T.to_csv(OUTPUT_DIR / "b2_h1_bootstrap.csv")

    meta = task["test"][["cve_id", "published_date"]].assign(
        scoring_date=task["test_dates"].to_numpy(), y=y_test
    ).reset_index(drop=True)
    pd.concat([meta, pd.DataFrame(predictions)], axis=1).to_pickle(PREDICTIONS_PATH)

    exclusions = {k: task[k] for k in ("n_excluded_already_known", "n_excluded_kev_already_known",
                                       "n_excluded_no_snapshot")}
    means = grouped.mean()
    stds = grouped.std(ddof=0)
    for scorer in means.index:
        metrics = {f"{m}_mean": means.loc[scorer, m] for m in REPORT_METRICS}
        metrics.update({f"{m}_std": stds.loc[scorer, m] for m in REPORT_METRICS})
        if scorer == "gbm":
            metrics.update({f"h1_{k}_{s}": v for k, d in h1.items() for s, v in d.items()})
            metrics.update(exclusions)
        log_experiment(
            "B2", run_name=f"B2_{scorer}",
            params={"scorer": scorer, "split_date": SPLIT_DATE, "n_train": len(X_train),
                    "n_test": len(y_test), "n_test_kev": int(y_test.sum()),
                    "label_rule": "today" if scorer.endswith("labels_today") else "KEV as of T"},
            metrics=metrics,
            artifacts=sorted(OUTPUT_DIR.glob("b2_*")) if scorer == "gbm" else None,
            tags={"leaky_reference": str(scorer == "epss_current")},
        )

    return {"results": results, "summary": summary, "h1": h1, "exclusions": exclusions,
            "n_train": len(X_train), "n_train_kev": int(task["y_train"].sum()),
            "n_test": len(y_test), "n_test_kev": int(y_test.sum())}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out = run()
    print(out["summary"].to_string())
    print(pd.DataFrame(out["h1"]).T)
    print(out["exclusions"])
