"""Shared, cached feature matrices for Study C (C2-C6).

Two tasks, built exactly as C1-R built them and cached so every C experiment
sees byte-identical inputs:

- "full":   train on CVEs published < T = 2023-01-01 (KEV as of T), test on the
            B2 evaluation set (first-snapshot scoring, already-known excluded);
- "v2fold": selection protocol v2 (registered e21fe21) - train on CVEs
            published < 2021-05-01, validate on 2021-05-01..2022-12-31, labels
            KEV as of T for both.

Each task also carries query groups for learning-to-rank (C2):
- training rows all share one scoring date, so their groups are publication
  months (each month's new CVEs = one queue);
- evaluation rows are grouped by scoring date (B4's monthly batches).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.labels import kev_label_asof
from src.features import FeatureBuilder, load_l1, temporal_task
from src.models import classical
from src.models.c1_revised import FOLD_END, FOLD_SPLIT, LABEL_ASOF

logger = logging.getLogger(__name__)
CACHE_DIR = Path("data/processed/cache")


def _month(ts: pd.Series) -> np.ndarray:
    return ts.dt.tz_convert(None).dt.to_period("M").astype(str).to_numpy()


def _build(name: str) -> dict:
    if name == "full":
        task = classical.build(classical.SPLIT_DATE)
        y_train, y_eval = task["y_train"].to_numpy(), task["y_test"].to_numpy()
        X_train, X_eval = task["X_train"], task["X_test"]
    elif name == "v2fold":
        task = temporal_task(load_l1(), FOLD_SPLIT, test_end=FOLD_END)
        y_train = kev_label_asof(task["train"], LABEL_ASOF).to_numpy()
        y_eval = kev_label_asof(task["test"], LABEL_ASOF).to_numpy()
        builder = FeatureBuilder(use_epss=True).fit(task["train"])
        X_train = builder.transform(task["train"], task["train_dates"])
        X_eval = builder.transform(task["test"], task["test_dates"])
    else:
        raise ValueError(name)
    return {
        "X_train": X_train.reset_index(drop=True), "y_train": np.asarray(y_train, int),
        "X_eval": X_eval.reset_index(drop=True), "y_eval": np.asarray(y_eval, int),
        "group_train": _month(task["train"]["published_date"]),
        "group_eval": _month(task["test_dates"]),
        "cve_eval": task["test"]["cve_id"].to_numpy(),
        "scoring_date_eval": task["test_dates"].to_numpy(),
    }


def load(name: str) -> dict:
    path = CACHE_DIR / f"c_matrices_{name}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    data = _build(name)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(data, path)
    logger.info("%s: train %d (%d pos, %d groups), eval %d (%d pos, %d groups)", name,
                len(data["y_train"]), data["y_train"].sum(), len(np.unique(data["group_train"])),
                len(data["y_eval"]), data["y_eval"].sum(), len(np.unique(data["group_eval"])))
    return data


def epss_pit_test_scores() -> np.ndarray:
    """Seed-averaged tie-broken point-in-time EPSS baseline from B2, aligned to 'full'."""
    from src.models import b2_kev
    from src.models.baselines import SEEDS

    b2 = pd.read_pickle(b2_kev.PREDICTIONS_PATH)
    if not (b2["cve_id"].to_numpy() == load("full")["cve_eval"]).all():
        raise ValueError("cached full task differs from the B2 evaluation set")
    return np.mean([b2[f"epss_pit__s{s}"] for s in SEEDS], axis=0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for n in ("full", "v2fold"):
        d = load(n)
        print(n, d["X_train"].shape, d["y_train"].sum(), d["X_eval"].shape, d["y_eval"].sum(),
              len(np.unique(d["group_train"])), len(np.unique(d["group_eval"])))
