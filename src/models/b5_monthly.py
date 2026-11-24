"""B5-M (exploratory) - monthly drift on the B5 design.

Pre-registered in the Study C addendum. Same B5 models (train year N, 365-day
KEV label, CVSS + text, 10 seeds) and test years; the margin GBM PR-AUC minus
CVSS v3 PR-AUC (tie-broken, same seeds) is computed per test publication month.
Months with < 3 positives are dropped (count reported). Decay per N = slope of
the margin on months since the end of year N, with a bootstrap CI over months.
The registered B5 result remains the H5 test.

Usage:  python -m src.models.b5_monthly
"""

import logging
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.features import FeatureBuilder, load_l1
from src.models.b2_kev import make_gbm
from src.models.b5_drift import LAST_TEST_YEAR, MAX_GAP, TRAIN_YEARS, horizon_label, year_slice
from src.models.baselines import SEEDS, tie_broken_rank

logger = logging.getLogger(__name__)
OUTPUT_DIR = Path("outputs")
MIN_POSITIVES = 3
N_BOOT = 2000


def run() -> dict:
    from src.tracking import log_experiment

    df = load_l1()
    rows, n_dropped = [], 0
    for N in TRAIN_YEARS:
        train = year_slice(df, N)
        builder = FeatureBuilder(use_epss=False).fit(train)
        X_train, y_train = builder.transform(train), horizon_label(train)
        test = pd.concat([year_slice(df, N + g) for g in range(1, MAX_GAP + 1) if N + g <= LAST_TEST_YEAR])
        X_test, y_test = builder.transform(test), horizon_label(test).to_numpy()
        month = test["published_date"].dt.tz_localize(None).dt.to_period("M").to_numpy()
        v3 = test["cvss_score"].where(test["cvss_version"].astype(str).str.startswith("3")).fillna(-np.inf).to_numpy(float)
        gbm = [make_gbm(s).fit(X_train, y_train).predict_proba(X_test)[:, 1] for s in SEEDS]
        logger.info("B5-M N=%d fitted", N)
        for m in np.unique(month):
            idx = month == m
            if y_test[idx].sum() < MIN_POSITIVES:
                n_dropped += 1
                continue
            g = np.mean([average_precision_score(y_test[idx], p[idx]) for p in gbm])
            c = np.mean([average_precision_score(
                y_test[idx], tie_broken_rank(v3[idx], np.random.default_rng([s, zlib.crc32(str(m).encode())])))
                for s in SEEDS])
            months_after = (m.year - N - 1) * 12 + m.month
            rows.append({"train_year": N, "month": str(m), "months_after": months_after,
                         "n": int(idx.sum()), "n_pos": int(y_test[idx].sum()), "gbm": g, "cvss_v3": c,
                         "margin": g - c})
    monthly = pd.DataFrame(rows)
    monthly.to_csv(OUTPUT_DIR / "b5m_monthly.csv", index=False)

    slopes = []
    rng = np.random.default_rng(0)
    for N, part in monthly.groupby("train_year"):
        x, yv = part["months_after"].to_numpy(float), part["margin"].to_numpy()
        slope = np.polyfit(x, yv, 1)[0]
        boot = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(x), len(x))
            if np.unique(x[i]).size > 1:
                boot.append(np.polyfit(x[i], yv[i], 1)[0])
        slopes.append({"train_year": N, "n_months": len(part), "slope_per_month": slope,
                       "ci_low": np.quantile(boot, 0.025), "ci_high": np.quantile(boot, 0.975),
                       "slope_per_year": slope * 12})
    slopes = pd.DataFrame(slopes)
    slopes.to_csv(OUTPUT_DIR / "b5m_slopes.csv", index=False)
    log_experiment("B5", run_name="B5M_monthly_drift_exploratory",
                   params={"min_positives": MIN_POSITIVES, "n_months_dropped": n_dropped, "exploratory": True},
                   metrics={f"slope_N{int(r.train_year)}{k}": float(getattr(r, k.lstrip("_")))
                            for r in slopes.itertuples() for k in ("_slope_per_year", "_ci_low", "_ci_high")},
                   artifacts=sorted(OUTPUT_DIR.glob("b5m_*")), tags={"exploratory": "true"})
    return {"monthly": monthly, "slopes": slopes, "n_dropped": n_dropped}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out = run()
    print(out["slopes"].round(5).to_string())
    print("months dropped (<3 positives):", out["n_dropped"])
