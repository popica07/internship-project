"""B5 - temporal drift: train on publication year N, test on N+1..N+3.

Design fixed in the Study B pre-registration addendum + its 2026-11-24
deviation (KEV launched 2021-11-03, so earlier training years have no
positives under a fixed horizon):
- N in {2021, 2022, 2023, 2024}; test years N+1..N+3, capped at 2025;
  2025 is restricted to CVEs published by 2025-11-23 (last complete horizon);
- LABEL (B5 only, pre-registered): positive iff added to KEV within 365 days of
  publication, for train and test alike. This separates drift from KEV lag
  (B1: recent CVEs have simply not had time to be observed exploited). It is
  deliberately NOT `temporal_labels`, whose as-of-split rule would make every
  later test year look worse purely through label immaturity;
- features: CVSS score / severity / version + description TF-IDF/SVD fitted on
  the training year only; EPSS excluded (no archive before 2021);
- B2's GBM, 10 seeds; CVSS v3 ordering (tie-broken, 10 seeds) on every test
  year as the reference that absorbs year-to-year difficulty.

H5: performance decays as the gap (test year - N) grows. Base rates differ by
year, so PR-AUC is also reported as lift (PR-AUC / base rate) and as the
margin over CVSS v3 in the same test year. The decay test is a Spearman
correlation between gap and margin over the (N, test year) cells, plus
the within-N comparison gap 1 vs the largest gap.

Usage:  python -m src.models.b5_drift
"""

import logging
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.evaluation.metrics import evaluate
from src.features import FeatureBuilder, load_l1
from src.models.b2_kev import make_gbm
from src.models.baselines import SEEDS, tie_broken_rank

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs")
TRAIN_YEARS = [2021, 2022, 2023, 2024]
MAX_GAP = 3
LAST_TEST_YEAR = 2025
HORIZON = pd.Timedelta(days=365)
LAST_COMPLETE_PUBLICATION = pd.Timestamp("2025-11-23", tz="UTC")  # + 365 d = KEV snapshot
REPORT_METRICS = ["pr_auc", "precision_at_100", "ndcg_at_100", "roc_auc"]


def horizon_label(df: pd.DataFrame) -> pd.Series:
    """1 iff added to KEV within 365 days of publication (B5 only)."""
    lag = df["kev_date_added"] - df["published_date"]
    return (df["kev_date_added"].notna() & (lag <= HORIZON)).astype(int)


def year_slice(df: pd.DataFrame, year: int) -> pd.DataFrame:
    part = df[df["published_date"].dt.year == year]
    if year == LAST_TEST_YEAR:
        part = part[part["published_date"] <= LAST_COMPLETE_PUBLICATION]
    return part


def run() -> dict:
    from src.tracking import log_experiment

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_l1()
    rows, sizes = [], []
    for N in TRAIN_YEARS:
        train = year_slice(df, N)
        y_train = horizon_label(train)
        builder = FeatureBuilder(use_epss=False).fit(train)
        X_train = builder.transform(train)
        sizes.append({"year": N, "role": "train", "n": len(train), "n_pos": int(y_train.sum())})
        test_years = [N + g for g in range(1, MAX_GAP + 1) if N + g <= LAST_TEST_YEAR]
        tests = {}
        for year in test_years:
            test = year_slice(df, year)
            if train["published_date"].max() >= test["published_date"].min():
                raise ValueError("temporal leakage between train and test year")
            tests[year] = (builder.transform(test), horizon_label(test).to_numpy(), test)
            sizes.append({"year": year, "role": f"test_for_{N}", "n": len(test),
                          "n_pos": int(tests[year][1].sum())})
        for seed in SEEDS:
            logger.info("B5 N=%d seed %d", N, seed)
            model = make_gbm(seed).fit(X_train, y_train)
            for year, (X_test, y_test, test) in tests.items():
                prob = model.predict_proba(X_test)[:, 1]
                rows.append({"train_year": N, "test_year": year, "gap": year - N, "scorer": "gbm",
                             "seed": seed, **evaluate(y_test, prob, is_probability=True)})
                v3 = test["cvss_score"].where(test["cvss_version"].astype(str).str.startswith("3"))
                rng = np.random.default_rng([seed, year, zlib.crc32(b"cvss_v3")])
                score = tie_broken_rank(v3.fillna(-np.inf).to_numpy(float), rng)
                rows.append({"train_year": N, "test_year": year, "gap": year - N, "scorer": "cvss_v3",
                             "seed": seed, **evaluate(y_test, score, is_probability=False)})

    results = pd.DataFrame(rows)
    results["pr_auc_lift"] = results["pr_auc"] / results["base_rate"]
    results.to_csv(OUTPUT_DIR / "b5_results_by_seed.csv", index=False)
    pd.DataFrame(sizes).drop_duplicates(["year", "role"]).to_csv(OUTPUT_DIR / "b5_sizes.csv", index=False)

    keys = ["train_year", "test_year", "gap", "scorer"]
    cells = results.groupby(keys)[REPORT_METRICS + ["pr_auc_lift", "base_rate", "n_positive"]].agg(["mean", "std"])
    cells.columns = [f"{m}_{s}" for m, s in cells.columns]
    cells = cells.reset_index()
    cells.round(4).to_csv(OUTPUT_DIR / "b5_cells.csv", index=False)

    wide = cells.pivot(index=["train_year", "test_year", "gap"], columns="scorer", values="pr_auc_mean")
    wide["margin_over_cvss"] = wide["gbm"] - wide["cvss_v3"]
    wide = wide.reset_index()
    lift = cells[cells.scorer == "gbm"].set_index(["train_year", "test_year"])["pr_auc_lift_mean"]
    wide["gbm_lift"] = lift.loc[list(zip(wide.train_year, wide.test_year))].to_numpy()
    wide.round(4).to_csv(OUTPUT_DIR / "b5_margin.csv", index=False)

    rho_margin = spearmanr(wide["gap"], wide["margin_over_cvss"])
    rho_lift = spearmanr(wide["gap"], wide["gbm_lift"])
    by_gap = wide.groupby("gap")[["gbm", "cvss_v3", "margin_over_cvss", "gbm_lift"]].mean().round(4)
    by_gap.to_csv(OUTPUT_DIR / "b5_by_gap.csv")
    within = []
    for N, part in wide.groupby("train_year"):
        if part["gap"].nunique() > 1:
            first, last = part.loc[part.gap.idxmin()], part.loc[part.gap.idxmax()]
            within.append({"train_year": N, "gap_last": int(last.gap),
                           "margin_change": float(last.margin_over_cvss - first.margin_over_cvss),
                           "lift_change": float(last.gbm_lift - first.gbm_lift)})
    within = pd.DataFrame(within)
    within.round(4).to_csv(OUTPUT_DIR / "b5_within_N.csv", index=False)
    h5 = {"spearman_gap_vs_margin": float(rho_margin.statistic), "p_gap_vs_margin": float(rho_margin.pvalue),
          "spearman_gap_vs_lift": float(rho_lift.statistic), "p_gap_vs_lift": float(rho_lift.pvalue),
          "n_cells": len(wide), "n_N_margin_fell": int((within.margin_change < 0).sum()),
          "n_N_lift_fell": int((within.lift_change < 0).sum()), "n_N_compared": len(within)}
    pd.Series(h5).to_csv(OUTPUT_DIR / "b5_h5.csv", header=["value"])

    log_experiment("B5", run_name="B5_drift",
                   params={"train_years": ",".join(map(str, TRAIN_YEARS)), "max_gap": MAX_GAP,
                           "label": "KEV within 365 days of publication", "features": "cvss + text (no EPSS)"},
                   metrics={**h5, **{f"gap{g}_{c}": float(by_gap.loc[g, c])
                                     for g in by_gap.index for c in by_gap.columns}},
                   artifacts=sorted(OUTPUT_DIR.glob("b5_*")))
    return {"cells": cells, "margin": wide, "by_gap": by_gap, "within": within, "h5": h5,
            "sizes": pd.DataFrame(sizes)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out = run()
    print(out["sizes"].to_string())
    print(out["margin"].to_string())
    print(out["by_gap"].to_string())
    print(out["within"].to_string())
    print(out["h5"])
