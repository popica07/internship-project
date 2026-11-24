"""C2 - learning-to-rank: pointwise vs pairwise vs listwise objectives (H3).

Design fixed in Study C addendum 2 (reports/analysis_plan_preregistration.md,
commit 59bd4b1), registered before any C2 model:
- six gradient-boosted contenders that differ only in the objective;
- training queries = publication months; evaluation queries = scoring months;
- XGBoost contenders re-use the exact C1 XGBoost draws (+ default), LightGBM
  contenders the C1 LightGBM draws; one fit each on the v2 fold;
- selection metric for ALL contenders: mean monthly validation NDCG@100;
- H3: best-validation LambdaMART vs best-validation pointwise contender,
  per-month test NDCG@100 (46 months), month bootstrap CI of the mean
  difference > 0.

Usage:  python -m src.models.c2_ltr --stage tune|final
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker
from xgboost import XGBClassifier, XGBRanker

from src.evaluation.metrics import evaluate, ndcg_at_k
from src.evaluation.statistical import friedman_nemenyi, wilcoxon_pair
from src.models import b2_kev, classical, study_c_data
from src.models.b4_ranking import METRICS as B4_METRICS
from src.models.baselines import SEEDS
from src.models.c1_revised import configs_from_c1

logger = logging.getLogger(__name__)
OUTPUT_DIR = Path("outputs")
CACHE_DIR = Path("data/processed/cache")
N_JOBS = classical.N_JOBS
TRUNCATION = 100

CONTENDERS = {  # name: (library, objective type)
    "xgb_binary": ("xgboost", "pointwise"),
    "lgbm_binary": ("lightgbm", "pointwise"),
    "xgb_pairwise": ("xgboost", "pairwise"),
    "xgb_lambdamart": ("xgboost", "listwise"),
    "lgbm_lambdamart": ("lightgbm", "listwise"),
    "lgbm_xendcg": ("lightgbm", "listwise"),
}
POINTWISE = ["xgb_binary", "lgbm_binary"]
LAMBDAMART = ["xgb_lambdamart", "lgbm_lambdamart"]
MONTHLY_METRICS = {**B4_METRICS, "ndcg_at_100": lambda y, s: ndcg_at_k(y, s, 100)}
N_MONTH_BOOTSTRAP = 2000


def make(contender: str, config: dict | None, seed: int):
    c = dict(config or {})
    library = CONTENDERS[contender][0]
    if library == "lightgbm" and "subsample" in c:
        c["subsample_freq"] = 1
    if contender == "xgb_binary":
        return XGBClassifier(tree_method="hist", n_jobs=N_JOBS, random_state=seed, **c)
    if contender == "lgbm_binary":
        return LGBMClassifier(n_jobs=N_JOBS, random_state=seed, verbose=-1, **c)
    if contender == "xgb_pairwise":
        return XGBRanker(objective="rank:pairwise", lambdarank_pair_method="mean",
                         lambdarank_num_pair_per_sample=8, tree_method="hist", n_jobs=N_JOBS,
                         random_state=seed, **c)
    if contender == "xgb_lambdamart":
        return XGBRanker(objective="rank:ndcg", lambdarank_pair_method="topk",
                         lambdarank_num_pair_per_sample=TRUNCATION, tree_method="hist", n_jobs=N_JOBS,
                         random_state=seed, **c)
    if contender == "lgbm_lambdamart":
        return LGBMRanker(objective="lambdarank", lambdarank_truncation_level=TRUNCATION, n_jobs=N_JOBS,
                          random_state=seed, verbose=-1, **c)
    if contender == "lgbm_xendcg":
        return LGBMRanker(objective="rank_xendcg", n_jobs=N_JOBS, random_state=seed, verbose=-1, **c)
    raise ValueError(contender)


def fit_score(contender: str, config: dict | None, seed: int, data: dict) -> np.ndarray:
    """Fit on group-sorted training rows; return scores for the evaluation rows."""
    order = np.argsort(data["group_train"], kind="stable")
    X, y, g = data["X_train"].iloc[order], data["y_train"][order], data["group_train"][order]
    model = make(contender, config, seed)
    if CONTENDERS[contender][1] == "pointwise":
        model.fit(X, y)
        return model.predict_proba(data["X_eval"])[:, 1]
    if CONTENDERS[contender][0] == "xgboost":
        model.fit(X, y, qid=pd.factorize(g)[0])
    else:
        model.fit(X, y, group=pd.Series(g).value_counts(sort=False).loc[pd.unique(g)].to_numpy())
    return model.predict(data["X_eval"])


def monthly(y: np.ndarray, score: np.ndarray, months: np.ndarray, metric) -> pd.Series:
    out = {}
    for m in np.unique(months):
        idx = months == m
        if y[idx].sum():
            out[m] = metric(y[idx], score[idx])
    return pd.Series(out)


def month_bootstrap(diff: pd.Series, n: int = N_MONTH_BOOTSTRAP, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    d = diff.to_numpy()
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return {"mean_diff": float(d.mean()), "ci_low": float(np.quantile(boots, 0.025)),
            "ci_high": float(np.quantile(boots, 0.975)), "p_not_better": float((boots <= 0).mean()),
            "n_months": len(d)}


# ---------------------------------------------------------------- tuning
def tune() -> pd.DataFrame:
    fold = study_c_data.load("v2fold")
    draws = configs_from_c1()
    rows, val_preds = [], {}
    for contender, (library, _) in CONTENDERS.items():
        for r in draws[draws.family == library].itertuples():
            start = time.time()
            s = fit_score(contender, json.loads(r.config), 0, fold)
            ndcg = monthly(fold["y_eval"], s, fold["group_eval"], MONTHLY_METRICS["ndcg_at_100"])
            val_preds[f"{contender}__{r.config_id}"] = s
            rows.append({"contender": contender, "config_id": r.config_id, "config": r.config,
                         "val_ndcg_at_100": float(ndcg.mean()), "seconds": round(time.time() - start, 1)})
            logger.info("C2 tune %s %s val NDCG@100 %.4f (%.0fs)", contender, r.config_id,
                        rows[-1]["val_ndcg_at_100"], rows[-1]["seconds"])
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "c2_tuning.csv", index=False)
    pd.DataFrame(val_preds).assign(y=fold["y_eval"]).to_pickle(CACHE_DIR / "c2_val_predictions.pkl")
    return out


def selected(tuning: pd.DataFrame) -> pd.DataFrame:
    searched = tuning[tuning.config_id.astype(str) != "default"]
    return searched.loc[searched.groupby("contender")["val_ndcg_at_100"].idxmax()].set_index("contender")


# ---------------------------------------------------------------- final
def final() -> dict:
    from src.tracking import log_experiment

    tuning = pd.read_csv(OUTPUT_DIR / "c2_tuning.csv", dtype={"config": str}, keep_default_na=False)
    chosen = selected(tuning)
    chosen.to_csv(OUTPUT_DIR / "c2_selected_configs.csv")
    pointwise = max(POINTWISE, key=lambda c: chosen.loc[c, "val_ndcg_at_100"])
    lambdamart = max(LAMBDAMART, key=lambda c: chosen.loc[c, "val_ndcg_at_100"])
    logger.info("Selected on validation: pointwise %s, LambdaMART %s", pointwise, lambdamart)

    data = study_c_data.load("full")
    y, months = data["y_eval"], data["group_eval"]
    preds, rows = {}, []
    for contender in CONTENDERS:
        config = json.loads(chosen.loc[contender, "config"])
        for seed in SEEDS:
            logger.info("C2 final %s seed %d", contender, seed)
            s = fit_score(contender, config, seed, data)
            preds[f"{contender}__s{seed}"] = s
            rows.append({"contender": contender, "objective": CONTENDERS[contender][1], "seed": seed,
                         **evaluate(y, s, is_probability=CONTENDERS[contender][1] == "pointwise")})
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "c2_results_by_seed.csv", index=False)
    metrics = ["pr_auc", "precision_at_10", "precision_at_100", "ndcg_at_100", "mrr", "roc_auc"]
    g = results.groupby("contender")[metrics]
    summary = (g.mean().round(4).astype(str) + " ± " + g.std(ddof=0).round(4).astype(str))
    summary = summary.loc[g.mean()["ndcg_at_100"].sort_values(ascending=False).index]
    summary.to_csv(OUTPUT_DIR / "c2_summary.csv")

    # Monthly metrics, per seed then averaged over seeds.
    monthly_rows = []
    for contender in CONTENDERS:
        per_metric = {m: pd.concat([monthly(y, preds[f"{contender}__s{s}"], months, fn) for s in SEEDS], axis=1).mean(axis=1)
                      for m, fn in MONTHLY_METRICS.items()}
        frame = pd.DataFrame(per_metric)
        frame.index.name = "month"
        monthly_rows.append(frame.reset_index().assign(contender=contender))
    monthly_df = pd.concat(monthly_rows, ignore_index=True)
    monthly_df.to_csv(OUTPUT_DIR / "c2_monthly_metrics.csv", index=False)

    wide = monthly_df.pivot(index="month", columns="contender", values="ndcg_at_100")
    diff = wide[lambdamart] - wide[pointwise]
    h3 = {"lambdamart": lambdamart, "pointwise": pointwise, **month_bootstrap(diff),
          **{f"wilcoxon_{k}": v for k, v in wilcoxon_pair(wide[lambdamart], wide[pointwise]).items()}}
    h3["H3_supported"] = bool(h3["ci_low"] > 0)
    pd.Series(h3).to_csv(OUTPUT_DIR / "c2_h3.csv", header=["value"])

    mean_score = {c: np.mean([preds[f"{c}__s{s}"] for s in SEEDS], axis=0) for c in CONTENDERS}
    secondary = {f"{a}_vs_{b}": b2_kev.paired_bootstrap(y, mean_score[a], mean_score[b])
                 for a, b in [(lambdamart, pointwise), ("xgb_pairwise", pointwise), ("lgbm_xendcg", pointwise)]}
    secondary["best_lambdamart_vs_epss_pit"] = b2_kev.paired_bootstrap(y, mean_score[lambdamart],
                                                                      study_c_data.epss_pit_test_scores())
    tests = pd.DataFrame(secondary).T
    tests.to_csv(OUTPUT_DIR / "c2_secondary_tests.csv")

    friedman = {}
    for m in ("ndcg_at_100", "map"):
        fn = friedman_nemenyi(monthly_df.pivot(index="month", columns="contender", values=m))
        fn["nemenyi"].to_csv(OUTPUT_DIR / f"c2_nemenyi_{m}.csv")
        friedman[m] = {"statistic": fn["statistic"], "p_value": fn["p_value"], "n_months": fn["n_blocks"],
                       **{f"rank_{k}": v for k, v in fn["mean_ranks"].items()}}
    friedman = pd.DataFrame(friedman).T
    friedman.to_csv(OUTPUT_DIR / "c2_friedman.csv")

    pd.DataFrame({"cve_id": data["cve_eval"], "scoring_date": data["scoring_date_eval"], "y": y, **preds}).to_pickle(
        CACHE_DIR / "c2_test_predictions.pkl")

    means, stds = g.mean(), g.std(ddof=0)
    for contender in CONTENDERS:
        m = {f"{k}_mean": means.loc[contender, k] for k in metrics}
        m.update({f"{k}_std": stds.loc[contender, k] for k in metrics})
        m["val_ndcg_at_100"] = float(chosen.loc[contender, "val_ndcg_at_100"])
        m["monthly_ndcg_at_100_mean"] = float(wide[contender].mean())
        is_summary = contender == lambdamart
        if is_summary:
            m.update({f"H3_{k}": float(v) for k, v in h3.items() if isinstance(v, (int, float, bool))})
        log_experiment("C2", run_name=f"C2_{contender}",
                       params={"contender": contender, "objective": CONTENDERS[contender][1],
                               "config": chosen.loc[contender, "config"], "selection": "v2 fold, monthly NDCG@100",
                               "pointwise_selected": pointwise, "lambdamart_selected": lambdamart},
                       metrics=m, artifacts=sorted(OUTPUT_DIR.glob("c2_*")) if is_summary else None,
                       tags={"protocol": "v2"})
    return {"summary": summary, "h3": h3, "tests": tests, "friedman": friedman}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["tune", "final"], required=True)
    if parser.parse_args().stage == "tune":
        print(tune().groupby("contender").val_ndcg_at_100.describe().round(4))
    else:
        out = final()
        print(out["summary"].to_string())
        print(pd.Series(out["h3"]).to_string())
        print(out["tests"].round(4).to_string())
        print(out["friedman"].round(4).to_string())
