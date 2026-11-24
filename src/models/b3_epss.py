"""B3 - EPSS regression (can public CVE data reproduce EPSS?).

Design fixed in the Study B pre-registration addendum:
- target = logit(EPSS) as of the split date T = 2023-01-01 (known at T);
- EPSS features are excluded (EPSS is the target); features are otherwise
  B2's primary set: CVSS score / severity / version + description TF-IDF/SVD;
- evaluated (a) as a regression: Spearman rho against TODAY's EPSS on the test
  CVEs, and (b) as a KEV ranker on the B2 evaluation set (same exclusions);
- HistGradientBoostingRegressor, 10 seeds.

Training CVEs with no EPSS on T (not yet scored / rejected) have no target and
are dropped; the count is reported.

Usage:  python -m src.models.b3_epss
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from src.evaluation.metrics import evaluate
from src.features import FeatureBuilder, epss_as_of, load_l1, temporal_task
from src.models.baselines import SEEDS

logger = logging.getLogger(__name__)

SPLIT_DATE = "2023-01-01"
OUTPUT_DIR = Path("outputs")
PREDICTIONS_PATH = Path("data/processed/cache/b3_test_predictions.pkl")
EPS = 1e-5
REPORT_METRICS = ["spearman_vs_epss_today", "pr_auc", "precision_at_100", "recall_at_100",
                  "ndcg_at_100", "mrr", "roc_auc"]


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def make_gbm(seed: int) -> HistGradientBoostingRegressor:
    # Same capacity and no-early-stopping rationale as B2 (src/models/b2_kev.py).
    return HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, max_features=0.5, early_stopping=False, random_state=seed,
    )


def run() -> dict:
    from src.tracking import log_experiment

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    task = temporal_task(load_l1(), SPLIT_DATE)
    builder = FeatureBuilder(use_epss=False).fit(task["train"])
    X_train = builder.transform(task["train"])
    X_test = builder.transform(task["test"])

    target_t = epss_as_of(task["train"]["cve_id"], task["train_dates"])["epss_pit"]
    has_target = target_t.notna()
    n_dropped = int((~has_target).sum())
    X_fit, y_fit = X_train[has_target], logit(target_t[has_target].to_numpy())

    y_kev = task["y_test"].to_numpy()
    epss_today = task["test"]["epss_score"].to_numpy(float)
    today_known = ~np.isnan(epss_today)

    rows, preds = [], []
    for seed in SEEDS:
        logger.info("B3 gbm seed %d", seed)
        pred = make_gbm(seed).fit(X_fit, y_fit).predict(X_test)
        preds.append(pred)
        rho = spearmanr(pred[today_known], epss_today[today_known]).statistic
        rows.append({"scorer": "b3_gbm", "seed": seed, "spearman_vs_epss_today": rho,
                     **evaluate(y_kev, pred, is_probability=False)})

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"cve_id": task["test"]["cve_id"].to_numpy(),
                  **{f"b3_gbm__s{s}": p for s, p in zip(SEEDS, preds)}}).to_pickle(PREDICTIONS_PATH)

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "b3_results_by_seed.csv", index=False)
    summary = pd.DataFrame({
        "mean": results[REPORT_METRICS].mean(), "std": results[REPORT_METRICS].std(ddof=0),
    }).round(4)
    summary.to_csv(OUTPUT_DIR / "b3_summary.csv")

    counts = {"n_train_with_target": int(has_target.sum()), "n_train_no_epss_at_T": n_dropped,
              "n_test": len(y_kev), "n_test_kev": int(y_kev.sum()),
              "n_test_with_epss_today": int(today_known.sum())}
    pd.Series(counts).to_csv(OUTPUT_DIR / "b3_counts.csv", header=["value"])

    metrics = {f"{m}_mean": summary.loc[m, "mean"] for m in REPORT_METRICS}
    metrics.update({f"{m}_std": summary.loc[m, "std"] for m in REPORT_METRICS})
    metrics.update(counts)
    log_experiment("B3", run_name="B3_gbm",
                   params={"split_date": SPLIT_DATE, "target": "logit(EPSS as of T)",
                           "features": "cvss + text (no EPSS)"},
                   metrics=metrics, artifacts=sorted(OUTPUT_DIR.glob("b3_*")))
    return {"results": results, "summary": summary, "counts": counts}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out = run()
    print(out["summary"].to_string())
    print(out["counts"])
