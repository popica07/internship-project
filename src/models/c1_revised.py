"""C1-R - selection protocol v2 + selection-reliability diagnostic.

Registered 2026-11-24 after C1 (commit e21fe21), before any run here.
- v2 fold: train on CVEs published < 2021-05-01 (features as of 2021-05-01),
  validate on 2021-05-01..2022-12-31 (first-snapshot scoring, already-known
  excluded), labels KEV as of T = 2023-01-01 for both: 652 / 145 positives;
- same 20 configs per family as C1 (+ default), one fit each, select on
  validation PR-AUC;
- diagnostic: all 126 configs fitted once (seed 0) on the full training set
  and scored on test, ONLY to measure whether validation rank predicts test
  rank (Spearman, per family and pooled) for the C1 fold and the v2 fold.
  Never used for selection;
- final: v2-selected config per family, 10 seeds, H2/H1b re-tested exactly
  as registered (classical.run with tag "c1r").

Usage:  python -m src.models.c1_revised --stage tune|diagnostic|final|analyse
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from src.evaluation.labels import kev_label_asof
from src.features import FeatureBuilder, load_l1, temporal_task
from src.models import classical

logger = logging.getLogger(__name__)
OUTPUT_DIR = Path("outputs")
FOLD_SPLIT, FOLD_END, LABEL_ASOF = "2021-05-01", "2023-01-01", "2023-01-01"


def configs_from_c1() -> pd.DataFrame:
    """The exact C1 search (same draws), so only the fold differs."""
    # keep_default_na=False: the default config is stored as the string "null", not NaN.
    t = pd.read_csv(OUTPUT_DIR / "c1_tuning.csv", dtype={"config": str}, keep_default_na=False)
    return t[["family", "config_id", "config"]]


def tune_v2() -> pd.DataFrame:
    task = temporal_task(load_l1(), FOLD_SPLIT, test_end=FOLD_END)
    y_tr = kev_label_asof(task["train"], LABEL_ASOF).to_numpy()
    y_val = kev_label_asof(task["test"], LABEL_ASOF).to_numpy()
    builder = FeatureBuilder(use_epss=True).fit(task["train"])
    X_tr = builder.transform(task["train"], task["train_dates"])
    X_val = builder.transform(task["test"], task["test_dates"])
    logger.info("v2 fold: train %d (%d pos), val %d (%d pos)", len(y_tr), y_tr.sum(), len(y_val), y_val.sum())
    rows = []
    for r in configs_from_c1().itertuples():
        model = classical.make_model(r.family, json.loads(r.config), seed=0).fit(X_tr, y_tr)
        score = average_precision_score(y_val, classical.predict(model, X_val))
        rows.append({"family": r.family, "config_id": r.config_id, "config": r.config, "val_pr_auc": score})
        logger.info("v2 tune %s %s val PR-AUC %.4f", r.family, r.config_id, score)
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "c1r_tuning.csv", index=False)
    return out


def diagnostic() -> pd.DataFrame:
    task = classical.build(classical.SPLIT_DATE)
    y = task["y_test"].to_numpy()
    rows = []
    for r in configs_from_c1().itertuples():
        model = classical.make_model(r.family, json.loads(r.config), seed=0).fit(task["X_train"], task["y_train"])
        score = average_precision_score(y, classical.predict(model, task["X_test"]))
        rows.append({"family": r.family, "config_id": r.config_id, "test_pr_auc_seed0": score})
        logger.info("diagnostic %s %s test PR-AUC %.4f", r.family, r.config_id, score)
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "c1r_diagnostic_test.csv", index=False)
    return out


def analyse() -> pd.DataFrame:
    v1 = pd.read_csv(OUTPUT_DIR / "c1_tuning.csv").rename(columns={"val_pr_auc": "val_c1_fold"})
    v2 = pd.read_csv(OUTPUT_DIR / "c1r_tuning.csv").rename(columns={"val_pr_auc": "val_v2_fold"})
    te = pd.read_csv(OUTPUT_DIR / "c1r_diagnostic_test.csv")
    keys = ["family", "config_id"]
    for d in (v1, v2, te):
        d["config_id"] = d["config_id"].astype(str)
    m = v1[keys + ["val_c1_fold"]].merge(v2[keys + ["val_v2_fold"]], on=keys).merge(te, on=keys, validate="one_to_one")
    m.to_csv(OUTPUT_DIR / "c1r_selection_reliability_configs.csv", index=False)
    rows = []
    for family, part in list(m.groupby("family")) + [("pooled", m)]:
        row = {"family": family, "n_configs": len(part)}
        for fold in ("val_c1_fold", "val_v2_fold"):
            rho = spearmanr(part[fold], part["test_pr_auc_seed0"])
            row[f"spearman_{fold}"] = rho.statistic
            row[f"p_{fold}"] = rho.pvalue
            # test PR-AUC of the config this fold would select (searched configs only)
            searched = part[part.config_id != "default"]
            pick = searched.loc[searched.groupby("family")[fold].idxmax()] if family != "pooled" else None
            if pick is not None:
                row[f"test_of_pick_{fold}"] = float(pick["test_pr_auc_seed0"].iloc[0])
        if family != "pooled":
            searched = part[part.config_id != "default"]
            row["test_best_possible"] = float(searched["test_pr_auc_seed0"].max())
            row["test_median_config"] = float(searched["test_pr_auc_seed0"].median())
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "c1r_selection_reliability.csv", index=False)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["tune", "diagnostic", "final", "analyse"], required=True)
    stage = parser.parse_args().stage
    if stage == "tune":
        print(tune_v2().groupby("family").val_pr_auc.describe().round(4))
    elif stage == "diagnostic":
        print(diagnostic().groupby("family").test_pr_auc_seed0.describe().round(4))
    elif stage == "analyse":
        print(analyse().round(4).to_string())
    else:
        out = classical.run(pd.read_csv(OUTPUT_DIR / "c1r_tuning.csv"), tag="c1r", variants=("tuned",))
        print(out["summary"].to_string())
        print(out["tests"].round(4).to_string())
        print("best GBM:", out["best_gbm"], "| best overall:", out["best_overall"])
        print("H2 supported:", out["H2_supported"], "| H1b supported:", out["H1b_supported"])
        print(out["friedman"].round(4).to_string())
