"""D3 - counterfactual explanations (descriptive). Design: Study D addendum (c0a3ee1).

L3 (SIMULATED, primary for D3): DiCE "random", 4 CFs for each of 100 held-out
findings predicted High/Critical, varying only remediation levers; validity
under the model and under the true (documented) label formula.

L1: minimum point-in-time EPSS at which a CVE ranked outside its month's
top-100 would reach tau_100 under M-XGB (deterministic 1-D search, 200-point
grid; percentile from that scoring date's empirical EPSS->percentile map).

Usage:  python -m src.explainability.counterfactuals
"""

import logging

import numpy as np
import pandas as pd

from src.explainability import common as c
from src.models import classical

logger = logging.getLogger(__name__)
OUT = c.OUTPUT_DIR
LEVERS = ["internet_facing", "has_admin_path", "admin_principal_count", "principal_count"]
IAM_LEVERS = ["has_admin_path", "admin_principal_count", "principal_count"]
N_L3, N_CF, GRID = 100, 4, 200


# ---------------------------------------------------------------- L3
MISSING = -999.0  # DiCE rejects NaN; non-lever numeric NaNs travel as this sentinel and are restored


class _Wrapped:
    """DiCE hands over object-typed frames; restore the training dtypes (and NaNs) before predicting."""

    def __init__(self, m, dtypes):
        self.m, self.dtypes = m, dtypes

    def _cast(self, X):
        X = X[list(self.dtypes.index)].copy()
        for col, t in self.dtypes.items():
            if isinstance(t, pd.CategoricalDtype):
                X[col] = X[col].astype(str).astype(t)  # "nan" is not a category -> NaN again
            else:
                X[col] = pd.to_numeric(X[col]).mask(lambda v: v == MISSING)
        return X

    def predict_proba(self, X):
        return self.m.predict_proba(self._cast(X))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def l3_counterfactuals() -> dict:
    import dice_ml

    X, df, y = c.l3_frame()
    tr, te = c.l3_split(df)
    m = c.l3_model()
    p_te = m.predict_proba(X.iloc[te])[:, 1]
    flagged = te[p_te >= 0.5]
    chosen = np.sort(np.random.default_rng(0).choice(flagged, min(N_L3, len(flagged)), replace=False))
    cat = [col for col in X.columns if isinstance(X[col].dtype, pd.CategoricalDtype)]
    cont = [col for col in X.columns if col not in cat]
    def dice_frame(frame):
        frame = frame.copy()
        for col in cat:
            frame[col] = frame[col].astype(str)
        frame[cont] = frame[cont].fillna(MISSING)
        return frame

    if X[LEVERS].isna().any().any():
        raise ValueError("levers must have no missing values")
    train = dice_frame(X.iloc[tr])
    train["y"] = y[tr]
    ranges = {f: [float(X.iloc[tr][f].min()), float(X.iloc[tr][f].max())] for f in LEVERS}
    data = dice_ml.Data(dataframe=train, continuous_features=cont, outcome_name="y")
    model = dice_ml.Model(model=_Wrapped(m, X.dtypes), backend="sklearn")
    explainer = dice_ml.Dice(data, model, method="random")

    rows, per_finding = [], []
    for i in chosen:
        q = dice_frame(X.iloc[[i]])
        try:
            res = explainer.generate_counterfactuals(q, total_CFs=N_CF, desired_class=0, features_to_vary=LEVERS,
                                                     permitted_range=ranges, random_seed=0, verbose=False)
            cfs = res.cf_examples_list[0].final_cfs_df
        except Exception as e:  # DiCE raises when it finds none; recorded, not dropped
            logger.info("D3 L3 finding %s: no CF (%s)", df.finding_id.iloc[i], str(e)[:80])
            cfs = None
        n = 0 if cfs is None else len(cfs)
        per_finding.append({"finding_id": df.finding_id.iloc[i], "p_model": float(m.predict_proba(X.iloc[[i]])[0, 1]),
                            "formula_label": int(c.l3_formula_label(df.iloc[[i]])[0]), "n_cf": n})
        if not n:
            continue
        cfs = cfs.drop(columns="y")
        p_cf = _Wrapped(m, X.dtypes).predict_proba(cfs)[:, 1]
        raw = pd.concat([df.iloc[[i]]] * n, ignore_index=True)
        for f in LEVERS:
            raw[f] = pd.to_numeric(cfs[f]).to_numpy()
        true_label = c.l3_formula_label(raw)
        lever_vals = cfs[LEVERS].apply(pd.to_numeric).to_numpy()
        changed = lever_vals != X.iloc[i][LEVERS].to_numpy(float)
        span = np.array([r[1] - r[0] or 1.0 for r in ranges.values()])
        # EXPLORATORY (added after the 5-finding check, not pre-registered): would flipping
        # internet_facing alone have sufficed? Measures superfluous changes made by DiCE "random".
        only_if = dice_frame(X.iloc[[i]])
        only_if["internet_facing"] = 1 - int(X.iloc[i]["internet_facing"])
        if_alone_valid = bool(_Wrapped(m, X.dtypes).predict_proba(only_if)[0, 1] < 0.5)
        for k in range(n):
            ch = [LEVERS[j] for j in np.flatnonzero(changed[k])]
            rows.append({"finding_id": df.finding_id.iloc[i], "cf": k, "p_cf": float(p_cf[k]),
                         "valid_model": bool(p_cf[k] < 0.5), "valid_formula": bool(true_label[k] == 0),
                         "n_changed": len(ch), "changed": ",".join(ch),
                         "iam_only": bool(ch) and set(ch) <= set(IAM_LEVERS),
                         "uses_internet_facing": "internet_facing" in ch,
                         "internet_facing_alone_valid": if_alone_valid,
                         "proximity_l1": float(np.abs((lever_vals[k] - X.iloc[i][LEVERS].to_numpy(float)) / span).sum())})
        if n > 1:
            d = [np.abs((lever_vals[a] - lever_vals[b]) / span).sum() for a in range(n) for b in range(a + 1, n)]
            per_finding[-1]["diversity"] = float(np.mean(d))
    cf, pf = pd.DataFrame(rows), pd.DataFrame(per_finding)
    cf.to_csv(OUT / "d3_l3_counterfactuals.csv", index=False)
    pf.to_csv(OUT / "d3_l3_per_finding.csv", index=False)
    valid = cf[cf.valid_model]
    summary = {
        "n_findings": len(pf), "share_with_any_cf": float((pf.n_cf > 0).mean()),
        "share_with_model_valid_cf": float(pf.finding_id.isin(valid.finding_id).mean()),
        "n_cf": len(cf), "share_cf_valid_model": float(cf.valid_model.mean()) if len(cf) else np.nan,
        "share_model_valid_also_formula_valid": float(valid.valid_formula.mean()) if len(valid) else np.nan,
        "share_model_valid_iam_only": float(valid.iam_only.mean()) if len(valid) else np.nan,
        "share_iam_only_formula_valid": float(valid[valid.iam_only].valid_formula.mean())
        if valid.iam_only.any() else np.nan,
        "share_model_valid_using_internet_facing": float(valid.uses_internet_facing.mean()) if len(valid) else np.nan,
        "mean_n_changed_valid": float(valid.n_changed.mean()) if len(valid) else np.nan,
        "exploratory_share_valid_with_superfluous_change": float(
            (valid.uses_internet_facing & valid.internet_facing_alone_valid & (valid.n_changed > 1)).mean())
        if len(valid) else np.nan,
        "mean_diversity": float(pf["diversity"].mean()) if "diversity" in pf else np.nan,
        "share_formula_positive_among_chosen": float(pf.formula_label.mean()),
        "share_flagged_where_internet_facing_could_help": float(
            (X.iloc[chosen]["internet_facing"] == 1).mean()),
    }
    return {"summary": summary, "cf": cf, "per_finding": pf}


# ---------------------------------------------------------------- L1
def epss_triggers(indices: np.ndarray, only_outside_top100: bool = True) -> tuple[pd.DataFrame, float]:
    d = c.data()
    m = c.model("xgboost", 0)
    X = d["X_eval"]
    p = classical.predict(m, X)
    month = pd.Series(d["group_eval"])
    tau100 = float(pd.Series(p).groupby(month).apply(lambda s: np.sort(s.to_numpy())[::-1][min(99, len(s) - 1)]).median())
    rank = pd.Series(p).groupby(month).rank(ascending=False, method="first").to_numpy()
    targets = indices[rank[indices] > 100] if only_outside_top100 else indices
    rows = []
    for i in targets:
        same = (d["scoring_date_eval"] == d["scoring_date_eval"][i]) & X["epss_pit"].notna().to_numpy()
        pairs = X.loc[same, ["epss_pit", "epss_pit_pct"]].sort_values("epss_pit")
        e0 = float(X["epss_pit"].iloc[i])
        grid = np.geomspace(max(e0, 1e-5), 1.0, GRID)
        Q = pd.concat([X.iloc[[i]]] * GRID, ignore_index=True)
        Q["epss_pit"] = grid
        Q["epss_pit_pct"] = np.interp(grid, pairs.epss_pit.to_numpy(), pairs.epss_pit_pct.to_numpy())
        pg = classical.predict(m, Q)
        hit = np.flatnonzero(pg >= tau100)
        rows.append({"cve_id": d["cve_eval"][i], "month": d["group_eval"][i], "kev": int(d["y_eval"][i]),
                     "rank_in_month": int(rank[i]), "score": float(p[i]), "epss_now": e0,
                     "epss_needed": float(grid[hit[0]]) if len(hit) else np.nan,
                     "pct_needed": float(Q["epss_pit_pct"].iloc[hit[0]]) if len(hit) else np.nan,
                     "max_score_any_epss": float(pg.max()), "reachable": bool(len(hit))})
    return pd.DataFrame(rows), tau100


def l1_epss_triggers() -> dict:
    t, tau100 = epss_triggers(c.explanation_sets()["E-rand"])
    t["uplift_ratio"] = t.epss_needed / t.epss_now
    t.to_csv(OUT / "d3_l1_epss_triggers.csv", index=False)
    r = t[t.reachable]
    summary = {"tau100": tau100, "n_cves": len(t), "share_reachable": float(t.reachable.mean()),
               "epss_now_median": float(t.epss_now.median()), "epss_needed_median": float(r.epss_needed.median()),
               "epss_needed_q25": float(r.epss_needed.quantile(0.25)), "epss_needed_q75": float(r.epss_needed.quantile(0.75)),
               "pct_needed_median": float(r.pct_needed.median()), "uplift_ratio_median": float(r.uplift_ratio.median())}
    return {"summary": summary, "table": t}


def run() -> dict:
    from src.tracking import log_experiment

    l3 = l3_counterfactuals()
    l1 = l1_epss_triggers()
    pd.DataFrame([{"part": "L3 (SIMULATED) DiCE", **l3["summary"]}]).to_csv(OUT / "d3_l3_summary.csv", index=False)
    pd.DataFrame([{"part": "L1 EPSS trigger", **l1["summary"]}]).to_csv(OUT / "d3_l1_summary.csv", index=False)
    log_experiment("D3", run_name="D3_counterfactuals",
                   params={"l3_levers": ",".join(LEVERS), "dice": "random, 4 CFs, desired class 0",
                           "l1": "min EPSS reaching tau_100, 200-point log grid"},
                   metrics={**{f"l3_{k}": float(v) for k, v in l3["summary"].items() if pd.notna(v)},
                            **{f"l1_{k}": float(v) for k, v in l1["summary"].items() if pd.notna(v)}},
                   artifacts=sorted(OUT.glob("d3_*")), tags={"study": "D", "l3_part": "SIMULATED"})
    return {"l3": l3["summary"], "l1": l1["summary"]}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out = run()
    print("L3 (SIMULATED):", out["l3"])
    print("L1:", out["l1"])
