"""C6 - statistical comparison across every Study C final model.

Design fixed in Study C addendum 2 (commit 59bd4b1), before any C6 run:
- methods: C1-R (6), C2 (6), C3 (3) + B2 baselines CVSS v3 and
  point-in-time EPSS (17);
- (1) global test metrics, 10 seeds, mean ± std;
- (2) monthly MAP (primary), P@10, NDCG@10, NDCG@100, MRR over the 46 monthly
  batches; Friedman + Nemenyi; critical-difference diagrams (alpha 0.05);
- (3) Wilcoxon of every method vs the best mean-rank method and vs
  point-in-time EPSS, Holm-adjusted per metric.

Usage:  python -m src.models.c6_statistics   (after C1-R, C2, C3)
"""

import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import scikit_posthocs as sp  # noqa: E402

from src.evaluation.metrics import evaluate  # noqa: E402
from src.evaluation.statistical import friedman_nemenyi, wilcoxon_pair  # noqa: E402
from src.models import b2_kev, c2_ltr, classical, study_c_data  # noqa: E402
from src.models.baselines import SEEDS  # noqa: E402

logger = logging.getLogger(__name__)
OUTPUT_DIR = Path("outputs")
CACHE_DIR = Path("data/processed/cache")
MONTHLY = c2_ltr.MONTHLY_METRICS  # P@10, NDCG@10, MAP, MRR, NDCG@100
PRIMARY = "map"
GLOBAL = ["pr_auc", "precision_at_10", "precision_at_100", "ndcg_at_100", "mrr", "roc_auc"]


def collect() -> tuple[dict, np.ndarray, np.ndarray]:
    data = study_c_data.load("full")
    y = data["y_eval"]
    methods = {}
    c1r = pd.read_pickle(CACHE_DIR / "c1r_test_predictions.pkl")
    for f in classical.FAMILIES:
        methods[f"C1R:{f}"] = [c1r[f"{f}_tuned__s{s}"].to_numpy() for s in SEEDS]
    c2 = pd.read_pickle(CACHE_DIR / "c2_test_predictions.pkl")
    for c in c2_ltr.CONTENDERS:
        methods[f"C2:{c}"] = [c2[f"{c}__s{s}"].to_numpy() for s in SEEDS]
    c3 = pd.read_pickle(CACHE_DIR / "c3_test_predictions.pkl")
    for m in sorted({c.split("__")[0] for c in c3.columns if "__s" in c}):
        methods[f"C3:{m}"] = [c3[f"{m}__s{s}"].to_numpy() for s in SEEDS]
    b2 = pd.read_pickle(b2_kev.PREDICTIONS_PATH)
    for b in ("cvss_v3", "epss_pit"):
        methods[f"baseline:{b}"] = [b2[f"{b}__s{s}"].to_numpy() for s in SEEDS]
    for frame in (c1r, c2, c3, b2):
        if not (frame["cve_id"].to_numpy() == data["cve_eval"]).all():
            raise ValueError("cached predictions misaligned with the Study C test set")
    return methods, y, data["group_eval"]


def holm(p: pd.Series) -> pd.Series:
    order = p.sort_values().index
    m = len(p)
    adj, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[k]))
        adj[k] = running
    return pd.Series(adj).reindex(p.index)


def cd_figure(ranks: pd.Series, nemenyi: pd.DataFrame, metric: str, n_blocks: int, path: Path):
    fig, ax = plt.subplots(figsize=(10, 0.32 * len(ranks) + 1.5))
    # copy: scikit-posthocs 0.16 overwrites the p-value matrix in place with 0/1 flags
    sp.critical_difference_diagram(ranks, nemenyi.copy(), ax=ax, alpha=0.05)
    ax.set_title(f"Critical-difference diagram — monthly {metric} ({n_blocks} months, Nemenyi α = 0.05)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run() -> dict:
    from src.tracking import log_experiment

    methods, y, months = collect()
    rows = [{"method": name, "seed": seed, **evaluate(y, s, is_probability=False)}
            for name, seeds in methods.items() for seed, s in zip(SEEDS, seeds)]
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "c6_global_by_seed.csv", index=False)
    g = results.groupby("method")[GLOBAL]
    summary = g.mean().round(4).astype(str) + " ± " + g.std(ddof=0).round(4).astype(str)
    summary = summary.loc[g.mean()["pr_auc"].sort_values(ascending=False).index]
    summary.to_csv(OUTPUT_DIR / "c6_global_summary.csv")

    month_values = np.unique(months)
    month_idx = {m: np.flatnonzero(months == m) for m in month_values if y[months == m].sum()}
    monthly_rows = []
    for name, seeds in methods.items():
        for m, idx in month_idx.items():
            monthly_rows.append({"method": name, "month": m, **{
                k: float(np.mean([fn(y[idx], s[idx]) for s in seeds])) for k, fn in MONTHLY.items()}})
        logger.info("C6 monthly %s done", name)
    monthly = pd.DataFrame(monthly_rows)
    monthly.to_csv(OUTPUT_DIR / "c6_monthly_metrics.csv", index=False)

    friedman, wilcoxon_rows = {}, []
    for metric in MONTHLY:
        wide = monthly.pivot(index="month", columns="method", values=metric)
        fn = friedman_nemenyi(wide)
        fn["nemenyi"].to_csv(OUTPUT_DIR / f"c6_nemenyi_{metric}.csv")
        cd_figure(fn["mean_ranks"], fn["nemenyi"], metric, fn["n_blocks"], OUTPUT_DIR / f"c6_cd_{metric}.png")
        best = fn["mean_ranks"].index[0]
        friedman[metric] = {"statistic": fn["statistic"], "p_value": fn["p_value"], "n_months": fn["n_blocks"],
                            "best_mean_rank": best, **{f"rank_{k}": v for k, v in fn["mean_ranks"].items()}}
        for ref in (best, "baseline:epss_pit"):
            pvals, diffs = {}, {}
            for other in wide.columns:
                if other == ref:
                    continue
                w = wilcoxon_pair(wide[other], wide[ref])
                pvals[other], diffs[other] = w["p_value"], w["median_diff"]
            adj = holm(pd.Series(pvals))
            for other in pvals:
                wilcoxon_rows.append({"metric": metric, "reference": ref, "method": other,
                                      "median_diff_vs_ref": diffs[other], "p_value": pvals[other],
                                      "p_holm": float(adj[other]),
                                      "nemenyi_p": float(fn["nemenyi"].loc[other, ref])})
    friedman = pd.DataFrame(friedman).T
    friedman.to_csv(OUTPUT_DIR / "c6_friedman.csv")
    wilcoxon = pd.DataFrame(wilcoxon_rows)
    wilcoxon.to_csv(OUTPUT_DIR / "c6_wilcoxon_holm.csv", index=False)

    ranks = monthly.pivot(index="month", columns="method", values=PRIMARY).rank(axis=1, ascending=False).mean()
    log_experiment("C6", run_name="C6_statistical_comparison",
                   params={"n_methods": len(methods), "n_months": len(month_idx), "primary_monthly_metric": PRIMARY},
                   metrics={**{f"friedman_p_{m}": float(r["p_value"]) for m, r in friedman.iterrows()},
                            **{f"rank_{PRIMARY}_{k}".replace(":", "_"): float(v) for k, v in ranks.items()},
                            **{f"pr_auc_{k}".replace(":", "_"): float(v) for k, v in g.mean()["pr_auc"].items()}},
                   artifacts=sorted(OUTPUT_DIR.glob("c6_*")), tags={"protocol": "v2"})
    return {"summary": summary, "friedman": friedman, "wilcoxon": wilcoxon}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    pd.set_option("display.width", 250)
    out = run()
    print(out["summary"].to_string())
    print(out["friedman"][["statistic", "p_value", "n_months", "best_mean_rank"]].to_string())
    w = out["wilcoxon"]
    print(w[w.metric == PRIMARY].round(4).to_string())
