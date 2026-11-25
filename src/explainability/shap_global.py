"""D1 - global explainability (descriptive). Design: Study D addendum (c0a3ee1).

(1) SHAP on G-sample (M-XGB interventional TreeSHAP, M-LR exact linear SHAP),
    mean |phi| per feature and group, beeswarm of the top 15;
(2) permutation importance on the full test set (PR-AUC, 10 repeats), per
    feature and per group;
(3) first-order ALE (20 quantile bins) on a 20,000-row test sample;
(4) Kendall tau-b between the 69-feature rankings;
(5) L3 explanation-correctness check (SIMULATED).

Usage:  python -m src.explainability.shap_global
"""

import logging

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.evaluation.metrics import average_precision  # noqa: E402
from src.explainability import common as c  # noqa: E402
from src.models import classical  # noqa: E402
from src.models.baselines import SEEDS  # noqa: E402

logger = logging.getLogger(__name__)
OUT = c.OUTPUT_DIR
MODELS = {"logreg": "M-LR", "xgboost": "M-XGB"}
N_ALE, ALE_BINS = 20_000, 20


def _score(family, m, X):
    """Log-odds for both models (ALE); ranking is unchanged by the transform (permutation)."""
    return m.decision_function(X) if family == "logreg" else m.predict(X, output_margin=True)


# ---------------------------------------------------------------- (1) SHAP
def global_shap() -> dict:
    path = c.CACHE_DIR / "d1_shap_gsample.pkl"
    if path.exists():
        return pd.read_pickle(path)
    X = c.data()["X_eval"].iloc[c.explanation_sets()["G-sample"]]
    out = {f: c.shap_values(f, X) for f in MODELS}
    pd.to_pickle(out, path)
    return out


def beeswarm(phi: np.ndarray, X: pd.DataFrame, title: str, path):
    import shap

    exp = shap.Explanation(values=phi, data=X.to_numpy(), feature_names=[c.label(f) for f in X.columns])
    plt.figure()
    shap.plots.beeswarm(exp, max_display=15, show=False, plot_size=(10, 6.5))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close("all")


# ---------------------------------------------------------------- (2) permutation
def permutation_importance(family: str) -> pd.DataFrame:
    d = c.data()
    m = c.model(family)
    X, y = d["X_eval"], d["y_eval"]
    base = average_precision(y, classical.predict(m, X))
    units = {f: [f] for f in X.columns} | {f"GROUP:{g}": cols for g, cols in c.groups().items()}
    rows = []
    for name, cols in units.items():
        drops = []
        for r in SEEDS:
            perm = np.random.default_rng([r, 7]).permutation(len(X))
            Xp = X.copy()
            Xp[cols] = X[cols].to_numpy()[perm]  # one permutation per unit: groups move jointly
            drops.append(base - average_precision(y, classical.predict(m, Xp)))
        rows.append({"unit": name, "pr_auc_drop_mean": float(np.mean(drops)), "pr_auc_drop_std": float(np.std(drops))})
    logger.info("D1 permutation %s done (base PR-AUC %.4f)", family, base)
    return pd.DataFrame(rows).assign(model=MODELS[family], base_pr_auc=base)


# ---------------------------------------------------------------- (3) ALE
def ale(family: str, feature: str, X: pd.DataFrame, bins: int = ALE_BINS) -> pd.DataFrame:
    m = c.model(family)
    Xf = X[X[feature].notna()]
    edges = np.unique(np.quantile(Xf[feature], np.linspace(0, 1, bins + 1)))
    k = np.clip(np.searchsorted(edges, Xf[feature], side="left"), 1, len(edges) - 1)
    lo, hi = Xf.copy(), Xf.copy()
    lo[feature], hi[feature] = edges[k - 1], edges[k]
    diff = _score(family, m, hi) - _score(family, m, lo)
    effect = pd.Series(diff).groupby(k).mean().reindex(range(1, len(edges)), fill_value=0.0)
    counts = pd.Series(k).value_counts().reindex(range(1, len(edges)), fill_value=0)
    curve = np.concatenate([[0.0], effect.cumsum().to_numpy()])
    centres = (curve[1:] + curve[:-1]) / 2
    curve -= np.sum(centres * counts.to_numpy()) / counts.sum()
    return pd.DataFrame({"model": MODELS[family], "feature": feature, "x": edges, "ale_log_odds": curve})


def ale_figure(frame: pd.DataFrame, path):
    feats = list(dict.fromkeys(frame.feature))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, f in zip(axes.flat, feats):
        for (model, part), color in zip(frame[frame.feature == f].groupby("model"), ["#2a78d6", "#eb6834"]):
            ax.plot(part.x, part.ale_log_odds, "o-", ms=3, lw=2, color=color, label=model)
        if f.startswith("epss_pit") and f != "epss_pit_pct":
            ax.set_xscale("log")
        ax.axhline(0, color="#999", lw=0.8)
        ax.set(title=c.label(f), xlabel=f, ylabel="ALE (log-odds)")
        ax.legend(fontsize=8)
    fig.suptitle("D1 accumulated local effects on test (20 quantile bins)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- (5) L3
def l3_check() -> dict:
    import shap

    X, df, y = c.l3_frame()
    _, te = c.l3_split(df)
    m = c.l3_model()
    Xte = X.iloc[te]
    # categorical splits: SHAP 0.52 supports only path-dependent TreeSHAP for them
    phi = shap.TreeExplainer(m, feature_perturbation="tree_path_dependent").shap_values(Xte)
    imp = pd.Series(np.abs(phi).mean(axis=0), index=X.columns).sort_values(ascending=False)
    in_formula = imp.index.isin(c.L3_FORMULA_FEATURES)
    terms = c.l3_formula_terms(df.iloc[te])
    truth = (terms - terms.mean()).abs().mean()
    truth = truth.reindex(imp.index).fillna(0.0)  # features the formula does not use: true importance 0
    comparable = imp.index != "source_layer"  # gates which real component applies; no own term
    tau = kendalltau(imp[comparable], truth[comparable]).statistic
    table = pd.DataFrame({"mean_abs_shap": imp, "in_formula": in_formula, "formula_importance": truth})
    table.to_csv(OUT / "d1_l3_shap_vs_formula.csv")
    return {"share_on_formula_features": float(imp[in_formula].sum() / imp.sum()),
            "kendall_tau_shap_vs_formula": float(tau),
            "test_pr_auc": average_precision(y[te], m.predict_proba(Xte)[:, 1]), "table": table}


# ---------------------------------------------------------------- run
def run() -> dict:
    from src.tracking import log_experiment

    d = c.data()
    sets = c.explanation_sets()
    Xg = d["X_eval"].iloc[sets["G-sample"]]
    shap_g = global_shap()
    feats = c.feature_names()
    imp = pd.DataFrame({MODELS[f]: np.abs(v).mean(axis=0) for f, v in shap_g.items()}, index=feats)
    imp.index.name = "feature"
    imp.assign(label=[c.label(f, short=False) for f in feats]).to_csv(OUT / "d1_shap_mean_abs.csv")
    grp = imp.groupby([c.group_of(f) for f in feats]).sum()
    grp_share = grp / grp.sum()
    grp_share.to_csv(OUT / "d1_shap_group_share.csv")
    for f, v in shap_g.items():
        beeswarm(v, Xg, f"D1 SHAP, {MODELS[f]} (log-odds, 10,000 test CVEs)", OUT / f"d1_beeswarm_{f}.png")

    perm = pd.concat([permutation_importance(f) for f in MODELS])
    perm.to_csv(OUT / "d1_permutation_importance.csv", index=False)

    top_text = imp["M-XGB"].filter(like="text_").sort_values(ascending=False).index[:3].tolist()
    ale_feats = ["epss_pit", "epss_pit_pct", "cvss_score"] + top_text
    Xa = d["X_eval"].sample(N_ALE, random_state=0)
    ale_frame = pd.concat([ale(f, feat, Xa) for feat in ale_feats for f in MODELS])
    ale_frame.to_csv(OUT / "d1_ale.csv", index=False)
    ale_figure(ale_frame, OUT / "d1_ale.png")

    pi = perm[~perm.unit.str.startswith("GROUP:")].pivot(index="unit", columns="model", values="pr_auc_drop_mean")
    rankings = {"SHAP M-XGB": imp["M-XGB"], "SHAP M-LR": imp["M-LR"],
                "PermImp M-XGB": pi["M-XGB"].reindex(feats), "PermImp M-LR": pi["M-LR"].reindex(feats)}
    names = list(rankings)
    agree = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            agree.loc[a, b] = kendalltau(rankings[a], rankings[b]).statistic
    agree.to_csv(OUT / "d1_global_ranking_agreement.csv")

    l3 = l3_check()
    log_experiment("D1", run_name="D1_global_xai",
                   params={"shap": "interventional TreeSHAP (XGB) / exact linear (LR), 1,000-row train background",
                           "g_sample": len(Xg), "ale_features": ",".join(ale_feats)},
                   metrics={**{f"group_share_{g}_{m}".replace("-", "_"): float(v)
                               for g, row in grp_share.iterrows() for m, v in row.items()},
                            **{f"tau_{a}_vs_{b}".replace(" ", "_").replace("-", "_"): float(agree.loc[a, b])
                               for i, a in enumerate(names) for b in names[i + 1:]},
                            "l3_share_on_formula_features": l3["share_on_formula_features"],
                            "l3_tau_shap_vs_formula": l3["kendall_tau_shap_vs_formula"]},
                   artifacts=sorted(OUT.glob("d1_*")) + [OUT / "d_text_component_terms.csv"],
                   tags={"study": "D", "l3_part": "SIMULATED"})
    return {"importance": imp, "groups": grp_share, "perm": perm, "agreement": agree, "l3": l3,
            "top_text": top_text}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    pd.set_option("display.width", 250)
    out = run()
    print(out["groups"].round(3).to_string())
    print(out["importance"].sort_values("M-XGB", ascending=False).head(12).round(4).to_string())
    g = out["perm"][out["perm"].unit.str.startswith("GROUP:")]
    print(g.round(4).to_string())
    print(out["agreement"].round(3).to_string())
    print({k: v for k, v in out["l3"].items() if k != "table"})
