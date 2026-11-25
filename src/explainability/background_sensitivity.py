"""EXPLORATORY (added 2026-11-25 after the D2 case studies; NOT pre-registered).

D2 showed `cvss_version` with ~+2.7 log-odds in every case study. Test CVEs are
90% CVSS v3.1 while the pre-registered SHAP background (training rows) is 34%
v2.0, so the attribution may mostly encode "this is a recent CVE" relative to
the background rather than anything that separates test CVEs from each other.
This re-runs SHAP on E-rand with a test-era background (1,000 random test rows,
no labels used) and compares. The pre-registered D1/D2/D5 results stand; this
is reported beside them.

Usage:  python -m src.explainability.background_sensitivity   (after D2)
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from src.explainability import common as c
from src.explainability.shap_local import attributions
from src.explainability.stability import median_ci, tau_rows

logger = logging.getLogger(__name__)
OUT = c.OUTPUT_DIR


def run() -> dict:
    from src.tracking import log_experiment

    d = c.data()
    sets = c.explanation_sets()
    e_rand = sets["E-rand"]
    rng = np.random.default_rng([0, 2])
    bg_test = d["X_eval"].iloc[np.sort(rng.choice(len(d["y_eval"]), c.N_BACKGROUND, replace=False))]
    X = d["X_eval"].iloc[e_rand]
    attr = attributions()
    pos = {i: k for k, i in enumerate(attr["index"])}
    k = [pos[i] for i in e_rand]
    names = c.feature_names()
    rows, summary = [], {}
    for fam in ("xgboost", "logreg"):
        m = c.model(fam)
        train_bg = attr[f"shap_{fam}"][k]
        test_bg = c.lr_shap(m, X, bg_test) if fam == "logreg" else c.xgb_shap(m, X, bg_test)
        imp_tr, imp_te = np.abs(train_bg).mean(0), np.abs(test_bg).mean(0)
        for j, n in enumerate(names):
            rows.append({"model": fam, "feature": n, "mean_abs_shap_train_bg": imp_tr[j],
                         "mean_abs_shap_test_bg": imp_te[j]})
        tau_bg = tau_rows(train_bg, test_bg)
        lime = attr[f"lime_{fam}"][k]
        summary[fam] = {
            "global_rank_tau_train_vs_test_bg": float(kendalltau(imp_tr, imp_te).statistic),
            "per_instance_tau_train_vs_test_bg_median": float(np.median(tau_bg)),
            "cvss_version_rank_train_bg": int((imp_tr > imp_tr[names.index("cvss_version")]).sum() + 1),
            "cvss_version_rank_test_bg": int((imp_te > imp_te[names.index("cvss_version")]).sum() + 1),
            "cvss_version_share_train_bg": float(imp_tr[names.index("cvss_version")] / imp_tr.sum()),
            "cvss_version_share_test_bg": float(imp_te[names.index("cvss_version")] / imp_te.sum()),
            **dict(zip(["shap_lime_tau_median_test_bg", "ci_low", "ci_high"], median_ci(tau_rows(test_bg, lime)))),
        }
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "dx_background_sensitivity_features.csv", index=False)
    summ = pd.DataFrame(summary).T
    summ.to_csv(OUT / "dx_background_sensitivity_summary.csv")
    log_experiment("D-exploratory", run_name="DX_shap_background_sensitivity",
                   params={"status": "EXPLORATORY, not pre-registered", "background": "1,000 random test rows"},
                   metrics={f"{f}_{k2}": float(v) for f, s in summary.items() for k2, v in s.items()},
                   artifacts=sorted(OUT.glob("dx_*")), tags={"study": "D", "exploratory": "true"})
    return {"summary": summ, "table": table}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    pd.set_option("display.width", 250)
    out = run()
    print(out["summary"].round(4).to_string())
    t = out["table"]
    for f in ("xgboost", "logreg"):
        print(t[t.model == f].sort_values("mean_abs_shap_test_bg", ascending=False).head(8).round(4).to_string())
