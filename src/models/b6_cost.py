"""B6 - cost-sensitive evaluation of the B2 model.

Design fixed in the Study B pre-registration addendum:
- no bank cost data exists, so the ratio C_FN / C_FP is swept over
  {10, 50, 100, 500}; cost is per 1,000 test CVEs in units of C_FP
  (src/evaluation/cost_matrix.py);
- thresholds are chosen on a VALIDATION year, never on test: a model trained on
  CVEs published < 2022-01-01 (KEV as of 2022-01-01) scores 2022 CVEs, labelled
  with KEV as of 2023-01-01 (what was knowable when B2's model was built).
  The cost-minimising threshold there is applied to B2's GBM on test, seed by
  seed (same seed for the validation and test model);
- compared policies: B2 GBM at the validated threshold, CVSS >= 7, CVSS >= 9,
  remediate-all, remediate-none, and a fixed monthly remediation budget (top-k
  of each monthly batch, k in {10, 50, 100} - the protocol's k values) for the
  GBM and for CVSS v3 ordering;
- calibration (Brier, ECE, reliability curve) of B2 is reported first.

Also reported, labelled as a reference: the GBM at the Bayes threshold
1 / (1 + ratio), which is optimal only if the probabilities are calibrated.

Usage:  python -m src.models.b6_cost   (after b2_kev)
"""

import logging
import zlib
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402

from src.evaluation.cost_matrix import bayes_threshold, best_threshold, expected_cost  # noqa: E402
from src.evaluation.labels import kev_label_asof  # noqa: E402
from src.evaluation.metrics import brier_score_loss, expected_calibration_error  # noqa: E402
from src.features import FeatureBuilder, load_l1, temporal_task  # noqa: E402
from src.models import b2_kev  # noqa: E402
from src.models.baselines import SEEDS, tie_broken_rank  # noqa: E402

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs")
COST_RATIOS = [10, 50, 100, 500]
BUDGETS = [10, 50, 100]
VAL_SPLIT, VAL_END = "2022-01-01", "2023-01-01"


def validation_predictions(df: pd.DataFrame) -> tuple[np.ndarray, list[np.ndarray]]:
    task = temporal_task(df, VAL_SPLIT, test_end=VAL_END)
    y_val = kev_label_asof(task["test"], VAL_END).to_numpy()  # KEV as of B2's split date
    builder = FeatureBuilder(use_epss=True).fit(task["train"])
    X_tr = builder.transform(task["train"], task["train_dates"])
    X_val = builder.transform(task["test"], task["test_dates"])
    probs = []
    for seed in SEEDS:
        logger.info("B6 validation model seed %d", seed)
        probs.append(b2_kev.make_gbm(seed).fit(X_tr, task["y_train"]).predict_proba(X_val)[:, 1])
    logger.info("validation: %d CVEs, %d KEV as of %s", len(y_val), y_val.sum(), VAL_END)
    return y_val, probs


def monthly_top_k(months: np.ndarray, score: np.ndarray, k: int) -> np.ndarray:
    decide = np.zeros(len(score), bool)
    for month in np.unique(months):
        idx = np.flatnonzero(months == month)
        decide[idx[np.argsort(-score[idx], kind="stable")[:k]]] = True
    return decide


def calibration_report(y: np.ndarray, probs: list[np.ndarray]) -> pd.DataFrame:
    rows = [{"seed": s, "brier": brier_score_loss(y, p), "ece": expected_calibration_error(y, p),
             "mean_prob": float(p.mean()), "base_rate": float(y.mean())} for s, p in zip(SEEDS, probs)]
    fig, ax = plt.subplots(figsize=(5, 5))
    frac, mean = calibration_curve(y, np.mean(probs, axis=0), n_bins=10, strategy="quantile")
    ax.plot(mean, frac, "o-", label="B2 GBM (seed mean)")
    lim = max(mean.max(), frac.max()) * 1.05
    ax.plot([0, lim], [0, lim], "--", color="grey", label="perfect calibration")
    ax.set(xlabel="mean predicted probability", ylabel="observed KEV rate",
           title="B2 reliability (test, quantile bins)", xscale="symlog", yscale="symlog",
           xlim=(0, lim), ylim=(0, lim))
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "b6_reliability.png", dpi=150)
    plt.close(fig)
    return pd.DataFrame(rows)


def run() -> dict:
    from src.tracking import log_experiment

    df = load_l1()
    preds = pd.read_pickle(b2_kev.PREDICTIONS_PATH)
    test = preds.merge(df[["cve_id", "cvss_score", "cvss_version"]], on="cve_id", how="left",
                       validate="one_to_one")
    y = test["y"].to_numpy()
    months = test["scoring_date"].to_numpy()
    cvss = test["cvss_score"].to_numpy(float)
    v3 = np.where(test["cvss_version"].astype(str).str.startswith("3"), cvss, -np.inf)
    v3 = np.nan_to_num(v3, nan=-np.inf)
    gbm = [test[f"gbm__s{s}"].to_numpy() for s in SEEDS]

    calib = calibration_report(y, gbm)
    calib.to_csv(OUTPUT_DIR / "b6_calibration.csv", index=False)

    y_val, val_probs = validation_predictions(df)

    rows, thresholds = [], []
    for ratio in COST_RATIOS:
        fixed = {"remediate_all": np.ones(len(y), bool), "remediate_none": np.zeros(len(y), bool),
                 "cvss_ge_7": np.nan_to_num(cvss, nan=-1) >= 7, "cvss_ge_9": np.nan_to_num(cvss, nan=-1) >= 9}
        for name, decide in fixed.items():
            rows.append({"ratio": ratio, "policy": name, "seed": -1, "cost": expected_cost(y, decide, ratio),
                         "n_remediated": int(decide.sum()), "kev_caught": int((decide & (y == 1)).sum())})
        for seed, p_val, p_test in zip(SEEDS, val_probs, gbm):
            t = best_threshold(y_val, p_val, ratio)
            thresholds.append({"ratio": ratio, "seed": seed, "threshold": t,
                               "val_cost": expected_cost(y_val, p_val >= t, ratio)})
            policies = {"gbm_validated_threshold": p_test >= t,
                        "gbm_bayes_threshold_ref": p_test >= bayes_threshold(ratio)}
            rng = np.random.default_rng([seed, zlib.crc32(b"cvss_v3")])
            v3_rank = tie_broken_rank(v3, rng)
            for k in BUDGETS:
                policies[f"gbm_budget_{k}_per_month"] = monthly_top_k(months, p_test, k)
                policies[f"cvss_v3_budget_{k}_per_month"] = monthly_top_k(months, v3_rank, k)
            for name, decide in policies.items():
                rows.append({"ratio": ratio, "policy": name, "seed": seed,
                             "cost": expected_cost(y, decide, ratio), "n_remediated": int(decide.sum()),
                             "kev_caught": int((decide & (y == 1)).sum())})

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "b6_costs_by_seed.csv", index=False)
    pd.DataFrame(thresholds).to_csv(OUTPUT_DIR / "b6_thresholds.csv", index=False)
    summary = (results.groupby(["ratio", "policy"])[["cost", "n_remediated", "kev_caught"]]
               .agg(["mean", "std"]).round(2))
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index().sort_values(["ratio", "cost_mean"])
    summary.to_csv(OUTPUT_DIR / "b6_summary.csv", index=False)
    best = summary.groupby("ratio").first()[["policy", "cost_mean"]]

    log_experiment("B6", run_name="B6_cost_sweep",
                   params={"cost_ratios": ",".join(map(str, COST_RATIOS)), "budgets": ",".join(map(str, BUDGETS)),
                           "validation": f"train<{VAL_SPLIT}, validate {VAL_SPLIT[:4]}, KEV as of {VAL_END}",
                           "n_test": len(y), "n_test_kev": int(y.sum()), "n_val_kev": int(y_val.sum())},
                   metrics={"brier_mean": float(calib.brier.mean()), "ece_mean": float(calib.ece.mean()),
                            **{f"cost_r{r}_{p}": float(c) for (r, p), c in
                               summary.set_index(["ratio", "policy"])["cost_mean"].items()}},
                   artifacts=sorted(OUTPUT_DIR.glob("b6_*")))
    return {"calibration": calib, "summary": summary, "best": best,
            "thresholds": pd.DataFrame(thresholds), "n_val_kev": int(y_val.sum())}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out = run()
    print(out["calibration"].describe().round(5).to_string())
    print(out["summary"].to_string())
    print(out["best"])
    print(out["thresholds"].groupby("ratio").threshold.describe())
