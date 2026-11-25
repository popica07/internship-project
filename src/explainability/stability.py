"""D5 - stability and agreement; decides H6. Design: Study D addendum (c0a3ee1).

H6 (SHAP and LIME disagree materially): per E-rand CVE, Kendall tau-b between
|TreeSHAP| and |LIME| over all 69 features (M-XGB, seed 0). Median over
E-rand, 95% CI by bootstrap over instances (2,000). Supported if the CI upper
bound < 0.5; not supported if the lower bound >= 0.5; inconclusive otherwise.

Secondary: E-top, E-kev, M-LR; top-5 feature and sign agreement; tau over the
three group attributions. Noise ceilings: LIME seeds 0-9 on the first 200
E-rand CVEs; TreeSHAP across XGBoost model seeds 0-9 on E-rand; SHAP-LIME vs
LIME-LIME tau by Wilcoxon.

Usage:  python -m src.explainability.stability   (after D2)
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, wilcoxon

from src.explainability import common as c, lime_local
from src.explainability.shap_local import attributions
from src.models.baselines import SEEDS

logger = logging.getLogger(__name__)
OUT = c.OUTPUT_DIR
N_BOOT, THRESHOLD, TOP_K, N_LIME_STABILITY = 2_000, 0.5, 5, 200


def tau_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([kendalltau(np.abs(x), np.abs(y)).statistic for x, y in zip(a, b)])


def top_k_agreement(a: np.ndarray, b: np.ndarray, k: int = TOP_K) -> tuple[np.ndarray, np.ndarray]:
    feat, sign = [], []
    for x, y in zip(a, b):
        tx, ty = set(np.argsort(-np.abs(x))[:k]), set(np.argsort(-np.abs(y))[:k])
        both = tx & ty
        feat.append(len(both) / k)
        sign.append(sum(np.sign(x[j]) == np.sign(y[j]) for j in both) / k)
    return np.array(feat), np.array(sign)


def group_sum(a: np.ndarray) -> np.ndarray:
    names = c.feature_names()
    return np.stack([a[:, [names.index(f) for f in cols]].sum(axis=1) for cols in c.groups().values()], axis=1)


def median_ci(v: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    boots = np.array([np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(N_BOOT)])
    return float(np.median(v)), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def verdict(low: float, high: float) -> str:
    if high < THRESHOLD:
        return "SUPPORTED (material disagreement)"
    if low >= THRESHOLD:
        return "NOT SUPPORTED"
    return "INCONCLUSIVE"


def lime_seed_stability(X: pd.DataFrame) -> np.ndarray:
    path = c.CACHE_DIR / "d5_lime_seeds.pkl"
    if path.exists():
        return pd.read_pickle(path)
    W = np.stack([lime_local.explain("xgboost", X, lime_seed=s)[0] for s in SEEDS])
    pd.to_pickle(W, path)
    return W


def shap_seed_stability(X: pd.DataFrame) -> np.ndarray:
    path = c.CACHE_DIR / "d5_shap_seeds.pkl"
    if path.exists():
        return pd.read_pickle(path)
    P = np.stack([c.shap_values("xgboost", X, seed=s) for s in SEEDS])
    pd.to_pickle(P, path)
    return P


def run() -> dict:
    from src.tracking import log_experiment

    attr = attributions()
    sets = c.explanation_sets()
    pos = {i: k for k, i in enumerate(attr["index"])}
    rows, per_instance = [], {}
    for fam in ("xgboost", "logreg"):
        for name in ("E-rand", "E-top", "E-kev"):
            k = [pos[i] for i in sets[name]]
            S, L = attr[f"shap_{fam}"][k], attr[f"lime_{fam}"][k]
            tau = tau_rows(S, L)
            feat, sign = top_k_agreement(S, L)
            gtau = tau_rows(group_sum(S), group_sum(L))
            med, lo, hi = median_ci(tau)
            per_instance[(fam, name)] = tau
            rows.append({"model": fam, "set": name, "n": len(k), "tau_median": med, "tau_ci_low": lo,
                         "tau_ci_high": hi, "tau_mean": float(tau.mean()),
                         "top5_feature_agreement_mean": float(feat.mean()),
                         "top5_sign_agreement_mean": float(sign.mean()),
                         "group_tau_median": float(np.median(gtau)),
                         "primary": fam == "xgboost" and name == "E-rand"})
    agreement = pd.DataFrame(rows)
    agreement.to_csv(OUT / "d5_shap_vs_lime.csv", index=False)
    h6 = agreement[agreement.primary].iloc[0]
    h6_verdict = verdict(h6.tau_ci_low, h6.tau_ci_high)

    d = c.data()
    e_rand = sets["E-rand"]
    sub = e_rand[:N_LIME_STABILITY]
    W = lime_seed_stability(d["X_eval"].iloc[sub])
    lime_lime = np.mean([tau_rows(W[0], W[s]) for s in range(1, len(SEEDS))], axis=0)
    shap_lime_sub = per_instance[("xgboost", "E-rand")][:N_LIME_STABILITY]
    P = shap_seed_stability(d["X_eval"].iloc[e_rand])
    shap_shap = np.mean([tau_rows(P[0], P[s]) for s in range(1, len(SEEDS))], axis=0)
    if not np.allclose(P[0], attr["shap_xgboost"][[pos[i] for i in e_rand]], atol=1e-8):
        raise ValueError("seed-0 TreeSHAP differs between D2 and D5")
    w = wilcoxon(shap_lime_sub, lime_lime)
    ceilings = pd.DataFrame([
        {"comparison": "LIME vs LIME (seeds 1-9 vs 0), M-XGB", "n": len(sub), **dict(zip(
            ["tau_median", "tau_ci_low", "tau_ci_high"], median_ci(lime_lime)))},
        {"comparison": "TreeSHAP vs TreeSHAP (model seeds 1-9 vs 0)", "n": len(e_rand), **dict(zip(
            ["tau_median", "tau_ci_low", "tau_ci_high"], median_ci(shap_shap)))},
        {"comparison": "SHAP vs LIME on the same 200", "n": len(sub), **dict(zip(
            ["tau_median", "tau_ci_low", "tau_ci_high"], median_ci(shap_lime_sub)))},
    ])
    ceilings["wilcoxon_shaplime_vs_limelime_p"] = [float(w.pvalue), np.nan, float(w.pvalue)]
    ceilings.to_csv(OUT / "d5_noise_ceilings.csv", index=False)
    # top-5 feature agreement within each explainer (stability of what an analyst reads first)
    lime_top = np.mean([top_k_agreement(W[0], W[s])[0] for s in range(1, len(SEEDS))])
    shap_top = np.mean([top_k_agreement(P[0], P[s])[0] for s in range(1, len(SEEDS))])
    pd.DataFrame({"tau_shap_lime": per_instance[("xgboost", "E-rand")], "tau_shap_seeds": shap_shap},
                 index=d["cve_eval"][e_rand]).to_csv(OUT / "d5_per_instance_tau_erand.csv")

    h6_row = {"statistic": "median Kendall tau-b |TreeSHAP| vs |LIME|, M-XGB, E-rand",
              "median": h6.tau_median, "ci_low": h6.tau_ci_low, "ci_high": h6.tau_ci_high,
              "threshold": THRESHOLD, "verdict": h6_verdict, "lime_top5_self_agreement": lime_top,
              "shap_top5_seed_agreement": shap_top}
    pd.DataFrame([h6_row]).to_csv(OUT / "d5_h6.csv", index=False)
    log_experiment("D5", run_name="D5_stability_agreement_H6",
                   params={"h6_rule": "CI upper < 0.5 supported; lower >= 0.5 not supported",
                           "n_boot": N_BOOT, "h6_verdict": h6_verdict},
                   metrics={"h6_tau_median": h6.tau_median, "h6_ci_low": h6.tau_ci_low, "h6_ci_high": h6.tau_ci_high,
                            "lime_lime_tau_median": float(np.median(lime_lime)),
                            "shap_seed_tau_median": float(np.median(shap_shap)),
                            "wilcoxon_p": float(w.pvalue), "lime_top5_self": lime_top, "shap_top5_seed": shap_top},
                   artifacts=sorted(OUT.glob("d5_*")), tags={"study": "D", "hypothesis": "H6"})
    return {"agreement": agreement, "h6": h6_row, "ceilings": ceilings}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    pd.set_option("display.width", 250)
    out = run()
    print(out["agreement"].round(3).to_string())
    print(out["ceilings"].round(4).to_string())
    print(out["h6"])
