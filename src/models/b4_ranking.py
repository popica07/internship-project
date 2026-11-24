"""B4 - pure ranking evaluation on the monthly triage queue.

Design fixed in the Study B pre-registration addendum:
- every scorer ranks each MONTHLY batch of test CVEs - the CVEs sharing one
  scoring date (first EPSS snapshot on/after publication), i.e. the queue an
  analyst actually works that month;
- per-month metrics: P@10, NDCG@10, MAP, MRR;
- Friedman across scorers with months as blocks, Nemenyi post-hoc, and a
  Wilcoxon signed-rank test for B2 (GBM) vs CVSS v3;
- months with no KEV positive are excluded (metrics undefined); count reported.

Inputs are the cached per-seed test predictions of B2 (models + tie-broken
baselines) and B3. For each scorer and month the metric is computed per seed
and then averaged over seeds, so baseline tie-breaking enters as noise, not as
a single lucky ordering.

`epss_current` is a leaky reference (A6): reported in the table, excluded from
the Friedman/Nemenyi comparison of fair scorers.

Usage:  python -m src.models.b4_ranking   (after b2_kev and b3_epss)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import average_precision, mean_reciprocal_rank, ndcg_at_k, precision_at_k
from src.evaluation.statistical import friedman_nemenyi, wilcoxon_pair
from src.models import b2_kev, b3_epss

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs")
METRICS = {
    "precision_at_10": lambda y, s: precision_at_k(y, s, 10),
    "ndcg_at_10": lambda y, s: ndcg_at_k(y, s, 10),
    "map": average_precision,
    "mrr": mean_reciprocal_rank,
}
SCORERS = ["gbm", "logreg", "b3_gbm", "epss_pit", "cvss_v3", "random", "epss_current"]
LEAKY = {"epss_current"}


def load_predictions() -> pd.DataFrame:
    b2 = pd.read_pickle(b2_kev.PREDICTIONS_PATH)
    b3 = pd.read_pickle(b3_epss.PREDICTIONS_PATH)
    merged = b2.merge(b3, on="cve_id", how="left", validate="one_to_one")
    if merged.filter(like="b3_gbm__").isna().any().any():
        raise ValueError("B3 predictions do not cover the B2 evaluation set")
    return merged


def per_month(preds: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    rows, n_empty = [], 0
    for month, batch in preds.groupby("scoring_date"):
        y = batch["y"].to_numpy()
        if y.sum() == 0:
            n_empty += 1
            continue
        for scorer in SCORERS:
            columns = [c for c in batch.columns if c.startswith(f"{scorer}__s")]
            values = {m: np.mean([f(y, batch[c].to_numpy()) for c in columns]) for m, f in METRICS.items()}
            rows.append({"month": month, "scorer": scorer, "n": len(y), "n_kev": int(y.sum()),
                         "n_seeds": len(columns), **values})
    return pd.DataFrame(rows), n_empty


def run() -> dict:
    from src.tracking import log_experiment

    preds = load_predictions()
    monthly, n_empty = per_month(preds)
    monthly.to_csv(OUTPUT_DIR / "b4_monthly_metrics.csv", index=False)

    summary = monthly.groupby("scorer")[list(METRICS)].agg(["mean", "std"]).round(4)
    summary = summary.loc[monthly.groupby("scorer")["map"].mean().sort_values(ascending=False).index]
    summary.to_csv(OUTPUT_DIR / "b4_summary.csv")

    tests, stats_rows = {}, []
    fair = [s for s in SCORERS if s not in LEAKY]
    for metric in METRICS:
        wide = monthly.pivot(index="month", columns="scorer", values=metric)
        fn = friedman_nemenyi(wide[fair])
        wx = wilcoxon_pair(wide["gbm"], wide["cvss_v3"])
        fn["nemenyi"].to_csv(OUTPUT_DIR / f"b4_nemenyi_{metric}.csv")
        tests[metric] = {"friedman": fn, "wilcoxon_gbm_vs_cvss_v3": wx}
        stats_rows.append({"metric": metric, "friedman_stat": fn["statistic"],
                           "friedman_p": fn["p_value"], "n_months": fn["n_blocks"],
                           "mean_ranks": fn["mean_ranks"].round(2).to_dict(),
                           "nemenyi_p_gbm_vs_cvss_v3": float(fn["nemenyi"].loc["gbm", "cvss_v3"]),
                           "wilcoxon_p_gbm_vs_cvss_v3": wx["p_value"],
                           "median_diff_gbm_minus_cvss_v3": wx["median_diff"]})
    stats = pd.DataFrame(stats_rows)
    stats.to_csv(OUTPUT_DIR / "b4_tests.csv", index=False)

    n_months = monthly["month"].nunique()
    metrics = {f"{m}_{scorer}_mean": float(monthly.loc[monthly.scorer == scorer, m].mean())
               for m in METRICS for scorer in SCORERS}
    for row in stats_rows:
        metrics.update({f"friedman_p_{row['metric']}": row["friedman_p"],
                        f"wilcoxon_p_gbm_vs_cvss_{row['metric']}": row["wilcoxon_p_gbm_vs_cvss_v3"]})
    log_experiment("B4", run_name="B4_monthly_ranking",
                   params={"batch": "scoring month", "n_months": n_months,
                           "n_months_no_kev_excluded": n_empty, "scorers": ",".join(SCORERS)},
                   metrics=metrics, artifacts=sorted(OUTPUT_DIR.glob("b4_*")))
    return {"summary": summary, "stats": stats, "n_months": n_months, "n_empty": n_empty}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out = run()
    print(out["summary"].to_string())
    print(out["stats"].drop(columns="mean_ranks").to_string())
    print(out["stats"][["metric", "mean_ranks"]].to_string())
    print("months used", out["n_months"], "excluded (no KEV)", out["n_empty"])
