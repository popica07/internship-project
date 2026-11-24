"""C1 - classical ML suite: 6 families, tuned on a time-ordered validation fold.

Design fixed in the Study C pre-registration addendum
(reports/analysis_plan_preregistration.md, registered before any C model):
- data / features / labels / evaluation set exactly as B2;
- random search, 20 configurations per family (sampled with seed 0), each
  fitted once on CVEs published < 2022-01-01 (KEV as of 2022-01-01) and scored
  on 2022 CVEs (KEV as of 2023-01-01); best validation PR-AUC is selected;
- the selected and the default configuration are refitted on all training
  data (< 2023-01-01) with 10 seeds and evaluated once on test;
- H2: best-validation GBM family vs LR and vs DT, paired bootstrap of PR-AUC;
- H1b: best-validation model overall vs point-in-time EPSS, PR-AUC and P@100;
- Friedman + Nemenyi across families on B4's monthly batches.

No class weighting, no early stopping (see addendum).

Usage:  python -m src.models.classical              (full run)
        python -m src.models.classical --tune-only  (validation search only)
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.evaluation.labels import kev_label_asof
from src.evaluation.metrics import evaluate
from src.evaluation.statistical import friedman_nemenyi
from src.features import FeatureBuilder, load_l1, temporal_task
from src.models import b2_kev
from src.models.b4_ranking import METRICS as MONTHLY_METRICS
from src.models.baselines import SEEDS

logger = logging.getLogger(__name__)

SPLIT_DATE = "2023-01-01"
VAL_SPLIT, VAL_END = "2022-01-01", "2023-01-01"
N_CONFIGS = 20
SEARCH_SEED = 0
FAMILIES = ["logreg", "decision_tree", "random_forest", "xgboost", "lightgbm", "catboost"]
GBM_FAMILIES = ["xgboost", "lightgbm", "catboost"]
OUTPUT_DIR = Path("outputs")
CACHE_DIR = Path("data/processed/cache")
PREDICTIONS_PATH = CACHE_DIR / "c1_test_predictions.pkl"
N_JOBS = 8


# ---------------------------------------------------------------- search spaces
def _log_uniform(rng, low, high):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _log_int(rng, low, high):
    return int(round(_log_uniform(rng, low, high)))


def sample_config(family: str, rng: np.random.Generator) -> dict:
    """One draw from the pre-registered space for `family`."""
    if family == "logreg":
        return {"C": _log_uniform(rng, 1e-3, 10)}
    if family == "decision_tree":
        return {"max_depth": int(rng.integers(3, 21)), "min_samples_leaf": _log_int(rng, 10, 500),
                "criterion": str(rng.choice(["gini", "entropy"]))}
    if family == "random_forest":
        return {"max_depth": [None, 10, 20, 30][int(rng.integers(4))],
                "min_samples_leaf": _log_int(rng, 1, 50),
                "max_features": ["sqrt", 0.3, 0.5][int(rng.integers(3))]}
    if family == "xgboost":
        return {"n_estimators": int(rng.integers(200, 801)), "learning_rate": _log_uniform(rng, 0.01, 0.2),
                "max_depth": int(rng.integers(3, 11)), "subsample": float(rng.uniform(0.5, 1)),
                "colsample_bytree": float(rng.uniform(0.5, 1)), "min_child_weight": _log_uniform(rng, 1, 20),
                "reg_lambda": _log_uniform(rng, 0.1, 10)}
    if family == "lightgbm":
        return {"n_estimators": int(rng.integers(200, 801)), "learning_rate": _log_uniform(rng, 0.01, 0.2),
                "num_leaves": _log_int(rng, 15, 127), "min_child_samples": _log_int(rng, 10, 200),
                "subsample": float(rng.uniform(0.5, 1)), "colsample_bytree": float(rng.uniform(0.5, 1)),
                "reg_lambda": _log_uniform(rng, 0.1, 10)}
    if family == "catboost":
        return {"iterations": int(rng.integers(200, 801)), "learning_rate": _log_uniform(rng, 0.01, 0.2),
                "depth": int(rng.integers(4, 9)), "l2_leaf_reg": _log_uniform(rng, 1, 10)}
    raise ValueError(family)


def _impute_scale(model):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), model)


def _impute(model):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), model)


def make_model(family: str, config: dict | None, seed: int):
    """config=None gives the library default (reported alongside the tuned model)."""
    c = dict(config or {})
    if family == "logreg":
        return _impute_scale(LogisticRegression(max_iter=3000, random_state=seed, **c))
    if family == "decision_tree":
        return _impute(DecisionTreeClassifier(random_state=seed, **c))
    if family == "random_forest":
        return _impute(RandomForestClassifier(n_estimators=300, n_jobs=N_JOBS, random_state=seed, **c))
    if family == "xgboost":
        return XGBClassifier(tree_method="hist", n_jobs=N_JOBS, random_state=seed, eval_metric="aucpr", **c)
    if family == "lightgbm":
        if "subsample" in c:
            c["subsample_freq"] = 1
        return LGBMClassifier(n_jobs=N_JOBS, random_state=seed, verbose=-1, **c)
    if family == "catboost":
        return CatBoostClassifier(thread_count=N_JOBS, random_seed=seed, verbose=False,
                                  allow_writing_files=False, **c)
    raise ValueError(family)


def predict(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------- data
def build(split: str, test_end: str | None = None) -> dict:
    task = temporal_task(load_l1(), split, test_end=test_end)
    builder = FeatureBuilder(use_epss=True).fit(task["train"])
    task["X_train"] = builder.transform(task["train"], task["train_dates"])
    task["X_test"] = builder.transform(task["test"], task["test_dates"])
    return task


# ---------------------------------------------------------------- tuning
def tune() -> pd.DataFrame:
    task = build(VAL_SPLIT, test_end=VAL_END)
    y_val = kev_label_asof(task["test"], VAL_END).to_numpy()
    logger.info("Validation fold: train %d (%d pos), val %d (%d pos)", len(task["y_train"]),
                task["y_train"].sum(), len(y_val), y_val.sum())
    rows = []
    for family in FAMILIES:
        rng = np.random.default_rng([SEARCH_SEED, FAMILIES.index(family)])
        configs = [None] + [sample_config(family, rng) for _ in range(N_CONFIGS)]
        for i, config in enumerate(configs):
            start = time.time()
            model = make_model(family, config, seed=0).fit(task["X_train"], task["y_train"])
            p = predict(model, task["X_test"])
            rows.append({"family": family, "config_id": "default" if config is None else i,
                         "config": json.dumps(config), "val_pr_auc": average_precision_score(y_val, p),
                         "seconds": round(time.time() - start, 1)})
            logger.info("tune %s %s val PR-AUC %.4f (%.0fs)", family, rows[-1]["config_id"],
                        rows[-1]["val_pr_auc"], rows[-1]["seconds"])
    results = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / "c1_tuning.csv", index=False)
    return results


def selected_configs(tuning: pd.DataFrame) -> dict:
    """Best *searched* config per family (the default is reported, not selectable)."""
    searched = tuning[tuning.config_id != "default"]
    best = searched.loc[searched.groupby("family")["val_pr_auc"].idxmax()]
    return {r.family: {"config": json.loads(r.config), "val_pr_auc": r.val_pr_auc} for r in best.itertuples()}


# ---------------------------------------------------------------- main run
def run(tuning: pd.DataFrame | None = None) -> dict:
    from src.tracking import log_experiment

    tuning = tune() if tuning is None else tuning
    chosen = selected_configs(tuning)
    best_gbm = max(GBM_FAMILIES, key=lambda f: chosen[f]["val_pr_auc"])
    best_overall = max(FAMILIES, key=lambda f: chosen[f]["val_pr_auc"])
    logger.info("Selected on validation: best GBM %s, best overall %s", best_gbm, best_overall)

    task = build(SPLIT_DATE)
    y = task["y_test"].to_numpy()
    rows, preds = [], {}
    for family in FAMILIES:
        for variant, config in (("tuned", chosen[family]["config"]), ("default", None)):
            for seed in SEEDS:
                logger.info("C1 %s %s seed %d", family, variant, seed)
                p = predict(make_model(family, config, seed).fit(task["X_train"], task["y_train"]), task["X_test"])
                preds[f"{family}_{variant}__s{seed}"] = p
                rows.append({"family": family, "variant": variant, "seed": seed,
                             **evaluate(y, p, is_probability=True)})
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "c1_results_by_seed.csv", index=False)

    metrics = ["pr_auc", "precision_at_10", "precision_at_100", "ndcg_at_100", "mrr", "roc_auc", "brier", "ece"]
    grouped = results.groupby(["family", "variant"])[metrics]
    summary = grouped.mean().round(4).astype(str) + " ± " + grouped.std(ddof=0).round(4).astype(str)
    summary = summary.loc[grouped.mean()["pr_auc"].sort_values(ascending=False).index]
    summary.to_csv(OUTPUT_DIR / "c1_summary.csv")

    # Seed-averaged scores for the hypothesis tests.
    mean_score = {f: np.mean([preds[f"{f}_tuned__s{s}"] for s in SEEDS], axis=0) for f in FAMILIES}
    b2 = pd.read_pickle(b2_kev.PREDICTIONS_PATH)
    if not (b2["cve_id"].to_numpy() == task["test"]["cve_id"].to_numpy()).all():
        raise ValueError("C1 test set differs from the B2 evaluation set")
    epss_pit = np.mean([b2[f"epss_pit__s{s}"] for s in SEEDS], axis=0)
    tests = {
        f"H2_{best_gbm}_vs_logreg": b2_kev.paired_bootstrap(y, mean_score[best_gbm], mean_score["logreg"]),
        f"H2_{best_gbm}_vs_decision_tree": b2_kev.paired_bootstrap(y, mean_score[best_gbm], mean_score["decision_tree"]),
        f"H1b_{best_overall}_vs_epss_pit": b2_kev.paired_bootstrap(y, mean_score[best_overall], epss_pit),
    }
    for f in GBM_FAMILIES:  # secondary
        if f != best_gbm:
            tests[f"secondary_{f}_vs_logreg"] = b2_kev.paired_bootstrap(y, mean_score[f], mean_score["logreg"])
            tests[f"secondary_{f}_vs_decision_tree"] = b2_kev.paired_bootstrap(y, mean_score[f], mean_score["decision_tree"])
    tests_df = pd.DataFrame(tests).T
    tests_df.to_csv(OUTPUT_DIR / "c1_hypothesis_tests.csv")
    h2 = all(tests[k]["pr_auc_ci_low"] > 0 for k in tests if k.startswith("H2_"))
    h1b_key = f"H1b_{best_overall}_vs_epss_pit"
    h1b = tests[h1b_key]["pr_auc_ci_low"] > 0 and tests[h1b_key]["p_at_100_ci_low"] > 0

    # Friedman + Nemenyi over monthly batches (tuned models, seed-averaged per month).
    months = task["test_dates"].to_numpy()
    monthly = []
    for month in np.unique(months):
        idx = months == month
        if y[idx].sum() == 0:
            continue
        for f in FAMILIES:
            monthly.append({"month": month, "family": f, **{
                m: np.mean([fn(y[idx], preds[f"{f}_tuned__s{s}"][idx]) for s in SEEDS])
                for m, fn in MONTHLY_METRICS.items()}})
    monthly = pd.DataFrame(monthly)
    monthly.to_csv(OUTPUT_DIR / "c1_monthly_metrics.csv", index=False)
    friedman = {}
    for m in MONTHLY_METRICS:
        fn = friedman_nemenyi(monthly.pivot(index="month", columns="family", values=m))
        fn["nemenyi"].to_csv(OUTPUT_DIR / f"c1_nemenyi_{m}.csv")
        friedman[m] = {"statistic": fn["statistic"], "p_value": fn["p_value"], "n_months": fn["n_blocks"],
                       **{f"rank_{k}": v for k, v in fn["mean_ranks"].items()}}
    pd.DataFrame(friedman).T.to_csv(OUTPUT_DIR / "c1_friedman.csv")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat([b2[["cve_id", "published_date", "scoring_date", "y"]], pd.DataFrame(preds)], axis=1).to_pickle(PREDICTIONS_PATH)
    pd.Series({f: json.dumps(c["config"]) for f, c in chosen.items()}).to_csv(OUTPUT_DIR / "c1_selected_configs.csv",
                                                                             header=["config"])

    means, stds = grouped.mean(), grouped.std(ddof=0)
    for (family, variant) in means.index:
        m = {f"{k}_mean": means.loc[(family, variant), k] for k in metrics}
        m.update({f"{k}_std": stds.loc[(family, variant), k] for k in metrics})
        m["val_pr_auc"] = chosen[family]["val_pr_auc"] if variant == "tuned" else float(
            tuning[(tuning.family == family) & (tuning.config_id == "default")].val_pr_auc.iloc[0])
        is_summary_run = family == best_overall and variant == "tuned"
        if is_summary_run:
            m.update({f"{k}__{s}": v for k, d in tests.items() for s, v in d.items()})
            m.update({"H2_supported": float(h2), "H1b_supported": float(h1b)})
        log_experiment("C1", run_name=f"C1_{family}_{variant}",
                       params={"family": family, "variant": variant, "split_date": SPLIT_DATE,
                               "config": json.dumps(chosen[family]["config"]) if variant == "tuned" else "default",
                               "best_gbm": best_gbm, "best_overall": best_overall},
                       metrics=m, artifacts=sorted(OUTPUT_DIR.glob("c1_*")) if is_summary_run else None)

    return {"tuning": tuning, "chosen": chosen, "best_gbm": best_gbm, "best_overall": best_overall,
            "summary": summary, "tests": tests_df, "H2_supported": h2, "H1b_supported": h1b,
            "friedman": pd.DataFrame(friedman).T}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune-only", action="store_true")
    parser.add_argument("--reuse-tuning", action="store_true", help="use outputs/c1_tuning.csv")
    args = parser.parse_args()
    if args.tune_only:
        print(tune().groupby("family").val_pr_auc.describe())
    else:
        out = run(pd.read_csv(OUTPUT_DIR / "c1_tuning.csv") if args.reuse_tuning else None)
        print(out["summary"].to_string())
        print(out["tests"].round(4).to_string())
        print("best GBM:", out["best_gbm"], "| best overall:", out["best_overall"])
        print("H2 supported:", out["H2_supported"], "| H1b supported:", out["H1b_supported"])
        print(out["friedman"].round(4).to_string())
