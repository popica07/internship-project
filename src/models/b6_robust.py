"""B6-R (exploratory) - bagged threshold for the cost-sensitive policy.

Pre-registered in the Study C addendum, with its 2026-11-24 deviation: the
threshold is the MEDIAN of the cost-minimising thresholds over 200 bootstrap
resamples of the validation fold (a mean-cost rule would reproduce the primary
threshold exactly; see the addendum). Same validation models, seeds, ratios and
test predictions as B6. Exploratory: the primary B6 table is unchanged.

Usage:  python -m src.models.b6_robust   (after b2_kev)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.cost_matrix import best_threshold, expected_cost
from src.features import load_l1
from src.models import b2_kev
from src.models.b6_cost import COST_RATIOS, validation_predictions
from src.models.baselines import SEEDS

logger = logging.getLogger(__name__)
OUTPUT_DIR = Path("outputs")
N_RESAMPLES = 200


def bagged_threshold(y, p, ratio, rng) -> float:
    picks = []
    while len(picks) < N_RESAMPLES:
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        picks.append(best_threshold(y[idx], p[idx], ratio))
    return float(np.median(picks))


def run() -> dict:
    from src.tracking import log_experiment

    df = load_l1()
    preds = pd.read_pickle(b2_kev.PREDICTIONS_PATH)
    y = preds["y"].to_numpy()
    y_val, val_probs = validation_predictions(df)
    rows = []
    for ratio in COST_RATIOS:
        for seed, p_val in zip(SEEDS, val_probs):
            p_test = preds[f"gbm__s{seed}"].to_numpy()
            rng = np.random.default_rng([seed, ratio])
            for rule, t in (("validated", best_threshold(y_val, p_val, ratio)),
                            ("bagged", bagged_threshold(y_val, p_val, ratio, rng))):
                decide = p_test >= t
                rows.append({"ratio": ratio, "seed": seed, "rule": rule, "threshold": t,
                             "cost": expected_cost(y, decide, ratio), "n_remediated": int(decide.sum()),
                             "kev_caught": int((decide & (y == 1)).sum())})
            logger.info("B6-R ratio %d seed %d done", ratio, seed)
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "b6r_by_seed.csv", index=False)
    summary = results.groupby(["ratio", "rule"])[["threshold", "cost", "n_remediated", "kev_caught"]].agg(
        ["mean", "std"]).round(4)
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary.to_csv(OUTPUT_DIR / "b6r_summary.csv")
    log_experiment("B6", run_name="B6R_bagged_threshold_exploratory",
                   params={"n_resamples": N_RESAMPLES, "rule": "median of per-resample argmin", "exploratory": True},
                   metrics={f"r{r}_{rule}_{m}": float(summary.loc[(r, rule), m])
                            for r, rule in summary.index for m in ("cost_mean", "cost_std", "threshold_std")},
                   artifacts=sorted(OUTPUT_DIR.glob("b6r_*")), tags={"exploratory": "true"})
    return {"summary": summary}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(run()["summary"].to_string())
