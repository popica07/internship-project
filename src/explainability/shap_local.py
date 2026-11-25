"""D2 - local explainability (descriptive). Design: Study D addendum (c0a3ee1).

SHAP (log-odds) and LIME (P(KEV)) for both models on E-rand ∪ E-top ∪ E-kev,
cached for D5; LIME local R^2; nine case studies chosen by a fixed rule:
from E-top under M-XGB the three highest-ranked KEV hits and the three
highest-ranked non-KEV findings, from E-kev the three KEV positives ranked
lowest within their month.

Usage:  python -m src.explainability.shap_local
"""

import logging
import time

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.explainability import common as c, lime_local  # noqa: E402
from src.models.baselines import SEEDS  # noqa: E402

logger = logging.getLogger(__name__)
OUT = c.OUTPUT_DIR
ATTR_PATH = c.CACHE_DIR / "d2_attributions.pkl"
FAMILIES = ["xgboost", "logreg"]


def union_index() -> np.ndarray:
    s = c.explanation_sets()
    return np.unique(np.concatenate([s["E-rand"], s["E-top"], s["E-kev"]]))


def attributions() -> dict:
    if ATTR_PATH.exists():
        return pd.read_pickle(ATTR_PATH)
    idx = union_index()
    X = c.data()["X_eval"].iloc[idx]
    out = {"index": idx}
    for f in FAMILIES:
        t = time.time()
        out[f"shap_{f}"] = c.shap_values(f, X)
        logger.info("D2 SHAP %s on %d rows: %.0f s", f, len(idx), time.time() - t)
        t = time.time()
        out[f"lime_{f}"], out[f"lime_r2_{f}"] = lime_local.explain(f, X)
        logger.info("D2 LIME %s on %d rows: %.0f s", f, len(idx), time.time() - t)
        pd.to_pickle(out, ATTR_PATH)
    return out


def month_ranks() -> pd.DataFrame:
    """Seed-mean C1-R XGBoost score and its rank within the scoring month (all test rows)."""
    d = c.data()
    pred = pd.read_pickle(c.CACHE_DIR / "c1r_test_predictions.pkl")
    s = np.mean([pred[f"xgboost_tuned__s{k}"] for k in SEEDS], axis=0)
    f = pd.DataFrame({"cve_id": d["cve_eval"], "month": d["group_eval"], "score": s, "y": d["y_eval"]})
    f["rank_in_month"] = f.groupby("month")["score"].rank(ascending=False, method="first").astype(int)
    f["month_size"] = f.groupby("month")["score"].transform("size")
    return f


def case_studies() -> pd.DataFrame:
    sets, r = c.explanation_sets(), month_ranks()
    top = r.iloc[sets["E-top"]].sort_values("score", ascending=False)
    kev = r.iloc[sets["E-kev"]].sort_values("rank_in_month", ascending=False)
    picks = pd.concat([top[top.y == 1].head(3).assign(case="top-ranked KEV hit"),
                       top[top.y == 0].head(3).assign(case="top-ranked, not in KEV"),
                       kev.head(3).assign(case="KEV, ranked lowest in its month")])
    return picks


def _bar(ax, values: np.ndarray, title: str, k: int = 8):
    names = c.feature_names()
    order = np.argsort(-np.abs(values))[:k][::-1]
    colors = ["#eb6834" if values[j] > 0 else "#2a78d6" for j in order]
    ax.barh(range(len(order)), values[order], color=colors, height=0.6)
    ax.set_yticks(range(len(order)), [c.label(names[j]) for j in order], fontsize=7)
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_title(title, fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))


def case_figure(cases: pd.DataFrame, attr: dict, path):
    pos = {i: k for k, i in enumerate(attr["index"])}
    fig, axes = plt.subplots(len(cases), 2, figsize=(13, 2.3 * len(cases)))
    for row, (i, case) in zip(axes, cases.iterrows()):
        k = pos[i]
        head = f"{case.cve_id} · {case.case} · rank {case.rank_in_month}/{case.month_size} ({case.month})"
        _bar(row[0], attr["shap_xgboost"][k], f"TreeSHAP (log-odds) — {head}")
        _bar(row[1], attr["lime_xgboost"][k], f"LIME (P(KEV), R² {attr['lime_r2_xgboost'][k]:.2f})")
    fig.suptitle("D2 case studies, M-XGB: orange raises the KEV score, blue lowers it", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def run() -> dict:
    from src.tracking import log_experiment

    attr = attributions()
    sets = c.explanation_sets()
    pos = {i: k for k, i in enumerate(attr["index"])}
    r2 = []
    for f in FAMILIES:
        for name in ("E-rand", "E-top", "E-kev"):
            v = attr[f"lime_r2_{f}"][[pos[i] for i in sets[name]]]
            r2.append({"model": f, "set": name, "n": len(v), "lime_r2_median": float(np.median(v)),
                       "lime_r2_q25": float(np.quantile(v, 0.25)), "lime_r2_q75": float(np.quantile(v, 0.75))})
    r2 = pd.DataFrame(r2)
    r2.to_csv(OUT / "d2_lime_r2.csv", index=False)

    cases = case_studies()
    names = c.feature_names()
    rows = []
    for i, case in cases.iterrows():
        k = pos[i]
        sh, li = attr["shap_xgboost"][k], attr["lime_xgboost"][k]
        top_s = [names[j] for j in np.argsort(-np.abs(sh))[:3]]
        top_l = [names[j] for j in np.argsort(-np.abs(li))[:3]]
        rows.append({"case": case.case, "cve_id": case.cve_id, "month": case.month, "score": case.score,
                     "rank_in_month": case.rank_in_month, "kev": case.y,
                     "shap_top3": "; ".join(f"{c.label(n)} {sh[names.index(n)]:+.2f}" for n in top_s),
                     "lime_top3": "; ".join(f"{c.label(n)} {li[names.index(n)]:+.4f}" for n in top_l),
                     "top3_overlap": len(set(top_s) & set(top_l))})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "d2_case_studies.csv", index=False)
    case_figure(cases, attr, OUT / "d2_case_studies.png")

    log_experiment("D2", run_name="D2_local_xai",
                   params={"lime": "continuous, sample_around_instance, 5000 samples, all features",
                           "n_explained": len(attr["index"])},
                   metrics={f"lime_r2_median_{r.model}_{r.set}".replace("-", "_"): r.lime_r2_median
                            for r in r2.itertuples()},
                   artifacts=sorted(OUT.glob("d2_*")), tags={"study": "D"})
    return {"r2": r2, "cases": table}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    pd.set_option("display.width", 250, "display.max_colwidth", 120)
    out = run()
    print(out["r2"].round(3).to_string())
    print(out["cases"].to_string())
