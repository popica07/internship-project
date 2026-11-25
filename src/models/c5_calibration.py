"""C5 - calibration of every Study C final model (descriptive).

Design fixed in Study C addendum 2 (commit 59bd4b1), before any C5 run:
- models: C1-R (6), C2 (6), C3 (3), C4 imbalance variants (4);
- raw Brier and ECE (locked: 10 equal-width bins) for probabilistic models,
  plus an adaptive ECE (15 equal-mass bins), labelled as added;
- Platt (logistic on logit(p), or on the raw score for ranking objectives) and
  isotonic, fitted on v2-fold validation predictions of the same configuration
  trained on the v2 fold (seed 0), applied to every test seed;
- reliability diagrams (equal-mass bins, log axes); Platt vs raw Brier by
  paired bootstrap over test CVEs (seed-averaged probabilities).

Usage:  python -m src.models.c5_calibration   (after C1-R, C2, C3, C4)
"""

import json
import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from src.evaluation.metrics import average_precision, brier_score_loss, expected_calibration_error  # noqa: E402
from src.models import c2_ltr, classical, study_c_data  # noqa: E402
from src.models.baselines import SEEDS  # noqa: E402
from src.models.c4_ablation import UNDERSAMPLE_RATIO, c1r_configs  # noqa: E402

logger = logging.getLogger(__name__)
OUTPUT_DIR = Path("outputs")
CACHE_DIR = Path("data/processed/cache")
EPS = 1e-7
N_BOOTSTRAP = 500


def adaptive_ece(y, p, n_bins: int = 15) -> float:
    order = np.argsort(p, kind="stable")
    ece = 0.0
    for chunk in np.array_split(order, n_bins):
        ece += len(chunk) / len(p) * abs(y[chunk].mean() - p[chunk].mean())
    return float(ece)


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Platt:
    def __init__(self, is_probability: bool):
        self.is_probability = is_probability

    def _x(self, s):
        return (_logit(s) if self.is_probability else np.asarray(s, float)).reshape(-1, 1)

    def fit(self, s, y):
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(self._x(s), y)
        return self

    def __call__(self, s):
        return self.lr.predict_proba(self._x(s))[:, 1]


# ---------------------------------------------------------------- inputs
def fold_predictions_classical(fold: dict) -> dict:
    """Seed-0 fold fits for C1-R families and the C4 imbalance variants."""
    out = {}
    configs = c1r_configs()
    for family in classical.FAMILIES:
        logger.info("C5 fold fit %s", family)
        m = classical.make_model(family, configs[family], 0).fit(fold["X_train"], fold["y_train"])
        out[f"C1R:{family}"] = classical.predict(m, fold["X_eval"])
    n_pos = int(fold["y_train"].sum())
    n_neg = len(fold["y_train"]) - n_pos
    for variant, w in [("spw_sqrt", np.sqrt(n_neg / n_pos)), ("spw_full", n_neg / n_pos)]:
        m = classical.make_model("xgboost", {**configs["xgboost"], "scale_pos_weight": float(w)}, 0)
        out[f"C4:xgboost_{variant}"] = classical.predict(m.fit(fold["X_train"], fold["y_train"]), fold["X_eval"])
    rng = np.random.default_rng([0, 4])
    pos = np.flatnonzero(fold["y_train"] == 1)
    neg = rng.choice(np.flatnonzero(fold["y_train"] == 0), size=UNDERSAMPLE_RATIO * len(pos), replace=False)
    idx = np.sort(np.concatenate([pos, neg]))
    m = classical.make_model("xgboost", configs["xgboost"], 0).fit(fold["X_train"].iloc[idx], fold["y_train"][idx])
    out[f"C4:xgboost_undersample_1to{UNDERSAMPLE_RATIO}"] = classical.predict(m, fold["X_eval"])
    m = classical.make_model("logreg", {**configs["logreg"], "class_weight": "balanced"}, 0)
    out["C4:logreg_balanced"] = classical.predict(m.fit(fold["X_train"], fold["y_train"]), fold["X_eval"])
    return out


def collect() -> tuple[dict, dict, dict, np.ndarray, np.ndarray]:
    """Returns test scores {model: [10 seeds]}, fold scores {model: array}, is_probability, y_test, y_fold."""
    fold = study_c_data.load("v2fold")
    test, val, is_prob = {}, {}, {}
    c1r = pd.read_pickle(CACHE_DIR / "c1r_test_predictions.pkl")
    for f in classical.FAMILIES:
        test[f"C1R:{f}"] = [c1r[f"{f}_tuned__s{s}"].to_numpy() for s in SEEDS]
        is_prob[f"C1R:{f}"] = True
    c4 = pd.read_pickle(CACHE_DIR / "c4_test_predictions.pkl")
    for col_prefix, name in [("imbalance|spw_sqrt|xgboost", "C4:xgboost_spw_sqrt"),
                             ("imbalance|spw_full|xgboost", "C4:xgboost_spw_full"),
                             (f"imbalance|undersample_1to{UNDERSAMPLE_RATIO}|xgboost",
                              f"C4:xgboost_undersample_1to{UNDERSAMPLE_RATIO}"),
                             ("imbalance|balanced|logreg", "C4:logreg_balanced")]:
        test[name] = [c4[f"{col_prefix}__s{s}"].to_numpy() for s in SEEDS]
        is_prob[name] = True
    val.update(fold_predictions_classical(fold))

    c2 = pd.read_pickle(CACHE_DIR / "c2_test_predictions.pkl")
    c2_val = pd.read_pickle(CACHE_DIR / "c2_val_predictions.pkl")
    c2_sel = pd.read_csv(OUTPUT_DIR / "c2_selected_configs.csv", index_col=0)
    for c in c2_ltr.CONTENDERS:
        test[f"C2:{c}"] = [c2[f"{c}__s{s}"].to_numpy() for s in SEEDS]
        val[f"C2:{c}"] = c2_val[f"{c}__{c2_sel.loc[c, 'config_id']}"].to_numpy()
        is_prob[f"C2:{c}"] = c2_ltr.CONTENDERS[c][1] == "pointwise"

    c3 = pd.read_pickle(CACHE_DIR / "c3_test_predictions.pkl")
    c3_val = pd.read_pickle(CACHE_DIR / "c3_val_predictions.pkl")
    c3_sel = pd.read_csv(OUTPUT_DIR / "c3_selected_configs.csv", index_col=0)
    for m in c3_sel.index:
        test[f"C3:{m}"] = [c3[f"{m}__s{s}"].to_numpy() for s in SEEDS]
        val[f"C3:{m}"] = c3_val[f"{m}__{c3_sel.loc[m, 'config_id']}"].to_numpy()
        is_prob[f"C3:{m}"] = True

    y_test = study_c_data.load("full")["y_eval"]
    for frame in (c1r, c2, c3):
        if len(frame) != len(y_test) or not (frame["y"].to_numpy() == y_test).all():
            raise ValueError("cached test predictions misaligned")
    return test, val, is_prob, y_test, fold["y_eval"]


# ---------------------------------------------------------------- run
def _metrics(y, p) -> dict:
    return {"brier": brier_score_loss(y, p), "ece": expected_calibration_error(y, p),
            "adaptive_ece": adaptive_ece(y, p), "pr_auc": average_precision(y, p), "mean_prob": float(p.mean())}


def brier_bootstrap(y, a, b, n=N_BOOTSTRAP, seed=0) -> dict:
    d = (a - y) ** 2 - (b - y) ** 2  # >0 means b is better
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return {"brier_raw_minus_platt": float(d.mean()), "ci_low": float(np.quantile(boots, 0.025)),
            "ci_high": float(np.quantile(boots, 0.975))}


def reliability_figure(y, curves: dict, path: Path, title: str):
    names = list(curves)
    cols = 4
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows), squeeze=False)
    for ax, name in zip(axes.flat, names):
        for label, p in curves[name].items():
            order = np.argsort(p, kind="stable")
            chunks = np.array_split(order, 15)
            ax.plot([p[c].mean() for c in chunks], [y[c].mean() for c in chunks], "o-", ms=3, label=label)
        lim = 1e-1
        ax.plot([1e-5, lim], [1e-5, lim], "--", color="grey", lw=1)
        ax.set(xscale="log", xlim=(1e-5, lim), title=name, xlabel="mean predicted", ylabel="observed KEV rate")
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.set_ylim(0, lim)
        ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run() -> dict:
    from src.tracking import log_experiment

    test, val, is_prob, y, y_fold = collect()
    rows, boot_rows, curves = [], [], {}
    for name, seeds in test.items():
        platt = Platt(is_prob[name]).fit(val[name], y_fold)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(val[name], y_fold)
        for seed, s in zip(SEEDS, seeds):
            methods = {"platt": platt(s), "isotonic": iso.predict(s)}
            if is_prob[name]:
                methods["raw"] = s
            for method, p in methods.items():
                rows.append({"model": name, "method": method, "seed": seed, **_metrics(y, p)})
        mean_s = np.mean(seeds, axis=0)
        curves[name] = {"platt": platt(mean_s), "isotonic": iso.predict(mean_s)}
        if is_prob[name]:
            curves[name] = {"raw": mean_s, **curves[name]}
            boot_rows.append({"model": name, **brier_bootstrap(y, mean_s, platt(mean_s))})
        logger.info("C5 %s done", name)
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "c5_results_by_seed.csv", index=False)
    g = results.groupby(["model", "method"])[["brier", "ece", "adaptive_ece", "pr_auc", "mean_prob"]]
    summary = g.mean().round(6).astype(str) + " ± " + g.std(ddof=0).round(6).astype(str)
    summary.to_csv(OUTPUT_DIR / "c5_summary.csv")
    boots = pd.DataFrame(boot_rows).set_index("model")
    boots.to_csv(OUTPUT_DIR / "c5_platt_vs_raw_brier.csv")
    for study in ("C1R", "C2", "C3", "C4"):
        part = {k: v for k, v in curves.items() if k.startswith(study + ":")}
        reliability_figure(y, part, OUTPUT_DIR / f"c5_reliability_{study.lower()}.png",
                           f"{study} reliability on test (15 equal-mass bins, seed-mean scores)")
    means = g.mean()
    log_experiment("C5", run_name="C5_calibration",
                   params={"calibrator_fit": "v2-fold validation, seed-0 fold model", "base_rate": float(y.mean()),
                           "n_models": len(test)},
                   metrics={f"{k}_{m}_{meth}".replace(":", "_"): float(v) for (m, meth), row in means.iterrows()
                            for k, v in row[["brier", "adaptive_ece"]].items()},
                   artifacts=sorted(OUTPUT_DIR.glob("c5_*")), tags={"protocol": "v2"})
    return {"summary": summary, "boots": boots, "means": means}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out = run()
    pd.set_option("display.width", 250)
    print(out["means"].round(6).to_string())
    print(out["boots"].to_string())
