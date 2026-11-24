"""C4 - feature ablation and class imbalance (exploratory, no hypothesis).

Design fixed in Study C addendum 2 (commit 59bd4b1), before any C4 model:
- L1 feature groups CVSS / text / EPSS-PIT: each alone, each pair, all;
  C1-R-selected XGBoost (10 seeds) and LR configurations, not re-tuned per
  subset; every set vs "all" by paired bootstrap;
- L1 imbalance on XGBoost (all features): none, scale_pos_weight sqrt(neg/pos)
  and neg/pos, negative undersampling to 1:20 (new sample per seed); LR
  balanced vs unweighted; PR-AUC, P@100, Brier, ECE vs "none";
- L3 (SIMULATED, reported separately): High/Critical target, finding-only vs
  context-only vs all, XGBoost defaults, 10 grouped-by-asset 80/20 splits.

Usage:  python -m src.models.c4_ablation --stage l1|l3
"""

import argparse
import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

from src.evaluation.metrics import evaluate
from src.models import b2_kev, classical, study_c_data
from src.models.baselines import SEEDS

logger = logging.getLogger(__name__)
OUTPUT_DIR = Path("outputs")
CACHE_DIR = Path("data/processed/cache")
L3_PATH = Path("data/simulator/l3_simulated_findings.csv")
UNDERSAMPLE_RATIO = 20
REPORT = ["pr_auc", "precision_at_100", "ndcg_at_100", "roc_auc", "brier", "ece"]


def feature_groups(columns) -> dict:
    return {"cvss": [c for c in columns if c.startswith("cvss_")],
            "text": [c for c in columns if c.startswith("text_")],
            "epss": [c for c in columns if c.startswith("epss_")]}


def feature_sets(groups: dict) -> dict:
    names = list(groups)
    sets = {}
    for r in (1, 2, 3):
        for combo in combinations(names, r):
            sets["+".join(combo) if r < 3 else "all"] = [c for g in combo for c in groups[g]]
    return sets


def c1r_configs() -> dict:
    s = pd.read_csv(OUTPUT_DIR / "c1r_selected_configs.csv", index_col=0)["config"]
    return {f: json.loads(c) for f, c in s.items()}


def _summarise(results: pd.DataFrame, keys: list[str], tag: str) -> pd.DataFrame:
    g = results.groupby(keys)[REPORT]
    summary = g.mean().round(4).astype(str) + " ± " + g.std(ddof=0).round(4).astype(str)
    summary = summary.loc[g.mean()["pr_auc"].sort_values(ascending=False).index]
    summary.to_csv(OUTPUT_DIR / f"{tag}_summary.csv")
    return summary


# ---------------------------------------------------------------- L1
def run_l1() -> dict:
    from src.tracking import log_experiment

    data = study_c_data.load("full")
    y = data["y_eval"]
    configs = c1r_configs()
    sets = feature_sets(feature_groups(data["X_train"].columns))
    rows, preds = [], {}

    def record(kind: str, variant: str, model: str, seed: int, p: np.ndarray):
        preds[f"{kind}|{variant}|{model}__s{seed}"] = p
        rows.append({"kind": kind, "variant": variant, "model": model, "seed": seed,
                     **evaluate(y, p, is_probability=True)})

    for name, cols in sets.items():
        sub = {"X_train": data["X_train"][cols], "X_eval": data["X_eval"][cols]}
        p = classical.predict(classical.make_model("logreg", configs["logreg"], 0).fit(sub["X_train"], data["y_train"]),
                              sub["X_eval"])
        for seed in SEEDS:  # LR is deterministic: one fit, repeated so every cell has 10 seeds
            record("ablation", name, "logreg", seed, p)
        for seed in SEEDS:
            logger.info("C4 ablation %s xgboost seed %d", name, seed)
            m = classical.make_model("xgboost", configs["xgboost"], seed).fit(sub["X_train"], data["y_train"])
            record("ablation", name, "xgboost", seed, classical.predict(m, sub["X_eval"]))

    n_pos = int(data["y_train"].sum())
    n_neg = len(data["y_train"]) - n_pos
    for variant, weight in [("spw_sqrt", np.sqrt(n_neg / n_pos)), ("spw_full", n_neg / n_pos)]:
        for seed in SEEDS:
            logger.info("C4 imbalance %s seed %d", variant, seed)
            m = classical.make_model("xgboost", {**configs["xgboost"], "scale_pos_weight": float(weight)}, seed)
            record("imbalance", variant, "xgboost", seed,
                   classical.predict(m.fit(data["X_train"], data["y_train"]), data["X_eval"]))
    for seed in SEEDS:
        logger.info("C4 imbalance undersample seed %d", seed)
        rng = np.random.default_rng([seed, 4])
        pos = np.flatnonzero(data["y_train"] == 1)
        neg = rng.choice(np.flatnonzero(data["y_train"] == 0), size=UNDERSAMPLE_RATIO * len(pos), replace=False)
        idx = np.sort(np.concatenate([pos, neg]))
        m = classical.make_model("xgboost", configs["xgboost"], seed)
        record("imbalance", f"undersample_1to{UNDERSAMPLE_RATIO}", "xgboost", seed,
               classical.predict(m.fit(data["X_train"].iloc[idx], data["y_train"][idx]), data["X_eval"]))
    p = classical.predict(classical.make_model("logreg", {**configs["logreg"], "class_weight": "balanced"}, 0)
                          .fit(data["X_train"], data["y_train"]), data["X_eval"])
    for seed in SEEDS:
        record("imbalance", "balanced", "logreg", seed, p)

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "c4_l1_results_by_seed.csv", index=False)
    summary = _summarise(results, ["kind", "variant", "model"], "c4_l1")
    pd.DataFrame({"cve_id": data["cve_eval"], "y": y, **preds}).to_pickle(CACHE_DIR / "c4_test_predictions.pkl")

    def mean_of(kind, variant, model):
        return np.mean([preds[f"{kind}|{variant}|{model}__s{s}"] for s in SEEDS], axis=0)

    tests = {}
    for model in ("xgboost", "logreg"):
        ref = mean_of("ablation", "all", model)
        for name in sets:
            if name != "all":
                tests[f"ablation_{model}_{name}_vs_all"] = b2_kev.paired_bootstrap(y, mean_of("ablation", name, model), ref)
        for variant in results[(results.kind == "imbalance") & (results.model == model)].variant.unique():
            tests[f"imbalance_{model}_{variant}_vs_none"] = b2_kev.paired_bootstrap(y, mean_of("imbalance", variant, model), ref)
    tests_df = pd.DataFrame(tests).T
    tests_df.to_csv(OUTPUT_DIR / "c4_l1_tests.csv")
    calib = results[results.kind == "imbalance"].groupby(["variant", "model"])[["brier", "ece"]].mean()
    log_experiment("C4", run_name="C4_l1_ablation_imbalance",
                   params={"feature_sets": ",".join(sets), "undersample_ratio": UNDERSAMPLE_RATIO,
                           "models": "C1-R selected xgboost + logreg (not re-tuned)"},
                   metrics={**{f"pr_auc_{k}_{v}_{m}": float(x) for (k, v, m), x in
                               results.groupby(["kind", "variant", "model"]).pr_auc.mean().items()},
                            **{f"brier_{v}_{m}": float(x) for (v, m), x in calib.brier.items()}},
                   artifacts=sorted(OUTPUT_DIR.glob("c4_l1_*")), tags={"protocol": "v2"})
    return {"summary": summary, "tests": tests_df}


# ---------------------------------------------------------------- L3 (simulated)
FINDING = ["source_layer", "tool", "base_severity", "epss_score", "epss_percentile", "cvss_score", "in_kev",
           "public_exploit"]
CONTEXT = ["criticality_tier", "internet_facing", "pci_scope", "dora_scope", "regulatory_scope", "environment",
           "asset_type", "business_unit", "provider", "principal_count", "admin_principal_count", "blast_radius",
           "has_admin_path", "sla_breached", "recurrence_count", "age_days"]


def run_l3() -> dict:
    from src.tracking import log_experiment

    df = pd.read_csv(L3_PATH, low_memory=False)
    X_all = df[FINDING + CONTEXT].copy()
    for c in X_all.columns:
        if X_all[c].dtype == object:
            X_all[c] = X_all[c].astype("category")
        elif X_all[c].dtype == bool:
            X_all[c] = X_all[c].astype(int)
    targets = {"noisy": df["priority_label"].isin(["High", "Critical"]).astype(int).to_numpy(),
               "clean": df["priority_label_clean"].isin(["High", "Critical"]).astype(int).to_numpy()}
    sets = {"finding_only": FINDING, "context_only": CONTEXT, "all": FINDING + CONTEXT}
    rows = []
    for seed in SEEDS:
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(X_all, groups=df["asset_id"]))
        for target, y in targets.items():
            for name, cols in sets.items():
                m = XGBClassifier(tree_method="hist", enable_categorical=True, n_jobs=classical.N_JOBS, random_state=seed)
                m.fit(X_all.iloc[tr][cols], y[tr])
                p = m.predict_proba(X_all.iloc[te][cols])[:, 1]
                rows.append({"target": target, "feature_set": name, "seed": seed, **evaluate(y[te], p)})
        logger.info("C4 L3 split %d done", seed)
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "c4_l3_results_by_seed.csv", index=False)
    summary = _summarise(results, ["target", "feature_set"], "c4_l3")
    log_experiment("C4", run_name="C4_l3_ablation_SIMULATED",
                   params={"target": "priority_label in {High, Critical}", "split": "10x grouped-by-asset 80/20",
                           "model": "xgboost defaults", "data": "L3 SIMULATED - not ground truth"},
                   metrics={f"pr_auc_{t}_{f}": float(x) for (t, f), x in
                            results.groupby(["target", "feature_set"]).pr_auc.mean().items()},
                   artifacts=sorted(OUTPUT_DIR.glob("c4_l3_*")), tags={"layer": "L3_simulated"})
    return {"summary": summary}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["l1", "l3"], required=True)
    stage = parser.parse_args().stage
    out = run_l1() if stage == "l1" else run_l3()
    print(out["summary"].to_string())
    if "tests" in out:
        print(out["tests"].round(4).to_string())
