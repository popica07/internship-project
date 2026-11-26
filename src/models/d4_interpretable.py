"""D4 - interpretable models: the price of interpretability (descriptive).

Design fixed in the Study D addendum (commit c0a3ee1), before any D4 score:
- EBM (interpret 0.7.8), RuleFit (imodels 3.0.4), sparse optimal tree
  (GOSDT 1.0.4, balance=True, threshold-guess binarisation with 200 depth-2
  stumps, leaf score = training KEV rate of the leaf);
- RuleFit/GOSDT: median imputation + missing indicators fitted on training rows;
  EBM: missing values natively;
- search on the v2 fold (seed 0), selected by validation PR-AUC; final config on
  the full training set, 10 seeds, test once;
- PR-AUC / P@100 / NDCG@100 / Brier, paired bootstrap vs C1-R LR, C1-R XGBoost,
  EPSS-PIT; model size.

Usage:  python -m src.models.d4_interpretable [--stage search|final|all]
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.evaluation.metrics import average_precision, evaluate
from src.models import b2_kev, classical, study_c_data
from src.models.baselines import SEEDS

logger = logging.getLogger(__name__)
OUTPUT_DIR = Path("outputs")
CACHE_DIR = Path("data/processed/cache")
GOSDT_TIME_LIMIT = 900
REPORT = ["pr_auc", "precision_at_100", "ndcg_at_100", "roc_auc", "brier", "ece"]

GRID = {
    "ebm": [{"interactions": i, "learning_rate": lr} for i in (0, 10) for lr in (0.015, 0.04)],
    "rulefit": [{"max_rules": k} for k in (10, 30, 60)],
    "gosdt": [{"depth_budget": d, "regularization": r} for d in (3, 4) for r in (0.001, 0.005)],
}
FALLBACK_GRID = [{"max_depth": d} for d in (3, 4)]


# ---------------------------------------------------------------- models
class Imputed:
    """Median imputation + missing indicators (fitted on training rows) in front of a model."""

    def fit_imputer(self, X: pd.DataFrame) -> pd.DataFrame:
        self.imputer = SimpleImputer(strategy="median", add_indicator=True).set_output(transform="pandas")
        return self.imputer.fit_transform(X)

    def impute(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.imputer.transform(X)


class EBM:
    def __init__(self, config: dict, seed: int):
        from interpret.glassbox import ExplainableBoostingClassifier

        self.m = ExplainableBoostingClassifier(random_state=seed, n_jobs=classical.N_JOBS, **config)

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict_proba(X)[:, 1]

    def size(self) -> dict:
        names = self.m.term_names_
        return {"n_terms": len(names), "n_interactions": sum(" & " in n for n in names)}


class RuleFit(Imputed):
    def __init__(self, config: dict, seed: int):
        from imodels import RuleFitClassifier

        self.m = RuleFitClassifier(random_state=seed, **config)

    def fit(self, X, y):
        Xi = self.fit_imputer(X)
        self.m.fit(Xi.to_numpy(), y, feature_names=list(Xi.columns))
        return self

    def predict(self, X):
        return self.m.predict_proba(self.impute(X).to_numpy())[:, 1]

    def size(self) -> dict:
        rules = self.m._get_rules()
        active = rules[rules.coef != 0]
        return {"n_terms": int(len(active)), "n_rules": int((active.type == "rule").sum()),
                "n_linear": int((active.type == "linear").sum())}


class SparseTree(Imputed):
    """GOSDT on threshold-guess binarised features; leaves scored by training KEV rate."""

    def __init__(self, config: dict, seed: int, binarizer=None):
        self.config, self.seed, self.binarizer = config, seed, binarizer

    def binarize_fit(self, X, y):
        from gosdt import ThresholdGuessBinarizer

        Xi = self.fit_imputer(X)
        self.binarizer = ThresholdGuessBinarizer(n_estimators=200, max_depth=2, random_state=self.seed)
        self.binarizer.set_output(transform="pandas")
        return self.binarizer.fit_transform(Xi, y)

    def fit(self, X, y, Xb: pd.DataFrame | None = None, imputer=None):
        from gosdt import GOSDTClassifier

        if Xb is None:
            Xb = self.binarize_fit(X, y)
        else:
            self.imputer = imputer
        self.columns = list(Xb.columns)
        self.m = GOSDTClassifier(balance=True, time_limit=GOSDT_TIME_LIMIT, verbose=False, **self.config).fit(Xb, y)
        self.tree = self.m.trees_[0].tree
        leaves = self._leaves(Xb)
        self._check_leaves(Xb, leaves)
        rate = pd.Series(y).groupby(leaves).mean()
        self.leaf_score = rate.to_dict()
        self.global_rate = float(np.mean(y))
        return self

    def _leaves(self, Xb: pd.DataFrame) -> np.ndarray:
        A = Xb[self.columns].to_numpy().astype(bool)
        out = np.empty(len(A), dtype=object)

        def walk(node, rows, path):
            if not hasattr(node, "feature"):
                out[rows] = path or "root"
                return
            f = node.feature
            true = A[rows, f]
            walk(node.left_child, rows[true], path + f"{f}T")  # left = feature true (checked below)
            walk(node.right_child, rows[~true], path + f"{f}F")

        walk(self.tree, np.arange(len(A)), "")
        return out

    def _check_leaves(self, Xb, leaves):
        """Every leaf must be pure in GOSDT's own prediction, or the left/right convention is wrong."""
        pred = self.m.predict(Xb)
        bad = pd.Series(pred).groupby(leaves).nunique().gt(1).any()
        if bad:
            raise ValueError("GOSDT leaf traversal disagrees with GOSDT predictions")

    def binarize(self, X):
        return self.binarizer.transform(self.impute(X))

    def predict(self, X):
        leaves = self._leaves(self.binarize(X))
        return np.array([self.leaf_score.get(l, self.global_rate) for l in leaves], float)

    def size(self) -> dict:
        return {"n_terms": len(self.leaf_score), "n_leaves": len(self.leaf_score)}


class WeightedCART(Imputed):
    """Registered fallback if GOSDT returns no tree with more than one leaf."""

    def __init__(self, config: dict, seed: int):
        from sklearn.tree import DecisionTreeClassifier

        self.m = DecisionTreeClassifier(class_weight="balanced", random_state=seed, **config)

    def fit(self, X, y):
        self.m.fit(self.fit_imputer(X), y)
        return self

    def predict(self, X):
        # balanced weights distort leaf probabilities; score by training KEV rate per leaf instead
        return self.leaf_rate[self.m.apply(self.impute(X))]

    def finish(self, X, y):
        leaves = self.m.apply(self.impute(X))
        rate = np.zeros(self.m.tree_.node_count)
        s = pd.Series(y).groupby(leaves).mean()
        rate[s.index] = s.values
        self.leaf_rate = rate
        return self

    def size(self) -> dict:
        return {"n_terms": int(self.m.get_n_leaves()), "n_leaves": int(self.m.get_n_leaves())}


def make(kind: str, config: dict, seed: int):
    return {"ebm": EBM, "rulefit": RuleFit, "gosdt": SparseTree, "cart_fallback": WeightedCART}[kind](config, seed)


def fit_predict(kind, config, seed, X_tr, y_tr, X_ev, **kw):
    m = make(kind, config, seed)
    t = time.time()
    if kind == "gosdt" and "Xb" in kw:
        m.fit(X_tr, y_tr, Xb=kw["Xb"], imputer=kw["imputer"])
    else:
        m.fit(X_tr, y_tr)
    if kind == "cart_fallback":
        m.finish(X_tr, y_tr)
    return m, m.predict(X_ev), time.time() - t


# ---------------------------------------------------------------- stages
def _gosdt_child(bin_path: str, cfg: dict, out_path: str):
    """Runs in a separate process: GOSDT's native code can die without a Python exception
    (depth 4 was killed silently in the first D4 run), which must be recorded, not crash D4."""
    Xb, y, imputer, binarizer, X_eval = pd.read_pickle(bin_path)
    m = SparseTree(cfg, 0, binarizer=binarizer)
    t = time.time()
    m.fit(None, y, Xb=Xb, imputer=imputer)
    pd.to_pickle((m.predict(X_eval), time.time() - t, m.size()), out_path)


def _gosdt_isolated(bin_path, cfg) -> tuple[np.ndarray, float, dict]:
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    out = CACHE_DIR / "d4_gosdt_child_out.pkl"
    out.unlink(missing_ok=True)
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as ex:
        ex.submit(_gosdt_child, str(bin_path), cfg, str(out)).result()  # BrokenProcessPool if killed
    return pd.read_pickle(out)


def search() -> pd.DataFrame:
    from src.tracking import log_experiment

    fold = study_c_data.load("v2fold")
    path = OUTPUT_DIR / "d4_search.csv"
    done = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["kind", "config_id"])
    # Failed configs are kept as failures and not retried: GOSDT depth 4 exhausted memory, swap filled
    # the disk ("No space left on device") and took the whole first run down with it.
    rows = done.to_dict("records")
    have = {(r["kind"], int(r["config_id"])) for r in rows}
    bin_path = CACHE_DIR / "d4_gosdt_v2fold_binarised.pkl"
    for kind, grid in GRID.items():
        for cid, cfg in enumerate(grid):
            if (kind, cid) in have:
                continue  # resumed: deterministic seed-0 fit already recorded
            logger.info("D4 search %s %s", kind, cfg)
            try:
                if kind == "gosdt":
                    if not bin_path.exists():
                        t = time.time()
                        st = SparseTree(cfg, 0)
                        Xb = st.binarize_fit(fold["X_train"], fold["y_train"])
                        pd.to_pickle((Xb, fold["y_train"], st.imputer, st.binarizer, fold["X_eval"]), bin_path)
                        logger.info("D4 GOSDT binarisation %.0f s, %d binary features", time.time() - t, Xb.shape[1])
                    p, sec, size = _gosdt_isolated(bin_path, cfg)
                else:
                    m, p, sec = fit_predict(kind, cfg, 0, fold["X_train"], fold["y_train"], fold["X_eval"])
                    size = m.size()
                rows.append({"kind": kind, "config_id": cid, "config": json.dumps(cfg),
                             "val_pr_auc": average_precision(fold["y_eval"], p), "fit_seconds": sec, **size})
            except Exception as e:  # recorded, never silently dropped
                logger.warning("D4 search %s %s failed: %r", kind, cfg, e)
                rows.append({"kind": kind, "config_id": cid, "config": json.dumps(cfg), "val_pr_auc": np.nan,
                             "error": repr(e)[:200]})
            pd.DataFrame(rows).to_csv(path, index=False)
    res = pd.DataFrame(rows)
    g = res[res.kind == "gosdt"]
    if not (g.get("n_leaves", pd.Series(dtype=float)) > 1).any():
        logger.warning("D4: no GOSDT tree with > 1 leaf - registered fallback (class-weighted CART)")
        for cid, cfg in enumerate(FALLBACK_GRID):
            m, p, sec = fit_predict("cart_fallback", cfg, 0, fold["X_train"], fold["y_train"], fold["X_eval"])
            rows.append({"kind": "cart_fallback", "config_id": cid, "config": json.dumps(cfg),
                         "val_pr_auc": average_precision(fold["y_eval"], p), "fit_seconds": sec, **m.size()})
        res = pd.DataFrame(rows)
    res.to_csv(OUTPUT_DIR / "d4_search.csv", index=False)
    ok = res[res.val_pr_auc.notna()]
    if "n_leaves" in ok:  # a one-leaf tree is a constant ranker, never "selected"
        ok = ok[~((ok.kind == "gosdt") & (ok.n_leaves <= 1))]
    sel = ok.loc[ok.groupby("kind").val_pr_auc.idxmax()].set_index("kind")
    sel.to_csv(OUTPUT_DIR / "d4_selected_configs.csv")
    log_experiment("D4", run_name="D4_search_v2fold",
                   params={"grid": json.dumps(GRID), "selection": "v2-fold validation PR-AUC, seed 0"},
                   metrics={f"val_pr_auc_{k}_{r.config_id}": float(r.val_pr_auc) for k, r in
                            res.set_index("kind").iterrows() if pd.notna(r.val_pr_auc)},
                   artifacts=[OUTPUT_DIR / "d4_search.csv", OUTPUT_DIR / "d4_selected_configs.csv"],
                   tags={"protocol": "v2", "study": "D"})
    return sel


def final() -> dict:
    from src.tracking import log_experiment

    sel = pd.read_csv(OUTPUT_DIR / "d4_selected_configs.csv", index_col=0)
    data = study_c_data.load("full")
    y = data["y_eval"]
    pred_path = CACHE_DIR / "d4_test_predictions.pkl"
    preds = pd.read_pickle(pred_path).to_dict("series") if pred_path.exists() else \
        {"cve_id": data["cve_eval"], "y": y}
    rows, sizes, importances = [], [], {}
    for kind, r in sel.iterrows():
        cfg = json.loads(r.config)
        for seed in SEEDS:
            col = f"{kind}__s{seed}"
            logger.info("D4 final %s seed %d", kind, seed)
            if col in preds:
                p = np.asarray(preds[col])
            else:
                m, p, sec = fit_predict(kind, cfg, seed, data["X_train"], data["y_train"], data["X_eval"])
                preds[col] = p
                sizes.append({"kind": kind, "seed": seed, "fit_seconds": sec, **m.size()})
                if kind == "ebm" and seed == 0:
                    imp = pd.Series(m.m.term_importances(), index=m.m.term_names_)
                    imp.to_csv(OUTPUT_DIR / "d4_ebm_term_importance.csv", header=["importance"])
                    importances["ebm"] = imp
                    pd.to_pickle(m, CACHE_DIR / "d4_ebm_s0.pkl")
                if seed == 0:
                    pd.to_pickle(m, CACHE_DIR / f"d4_{kind}_s0.pkl")
                pd.DataFrame(preds).to_pickle(pred_path)  # checkpoint per seed
            rows.append({"kind": kind, "seed": seed, **evaluate(y, p, is_probability=True)})
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "d4_results_by_seed.csv", index=False)
    if sizes:
        pd.DataFrame(sizes).to_csv(OUTPUT_DIR / "d4_model_size.csv", index=False)
    g = results.groupby("kind")[REPORT]
    summary = g.mean().round(4).astype(str) + " ± " + g.std(ddof=0).round(4).astype(str)
    summary.to_csv(OUTPUT_DIR / "d4_summary.csv")

    c1r = pd.read_pickle(CACHE_DIR / "c1r_test_predictions.pkl")
    refs = {"C1R_logreg": np.mean([c1r[f"logreg_tuned__s{s}"] for s in SEEDS], axis=0),
            "C1R_xgboost": np.mean([c1r[f"xgboost_tuned__s{s}"] for s in SEEDS], axis=0),
            "epss_pit": study_c_data.epss_pit_test_scores()}
    tests = {}
    for kind in sel.index:
        mean = np.mean([preds[f"{kind}__s{s}"] for s in SEEDS], axis=0)
        for ref, s in refs.items():
            tests[f"{kind}_vs_{ref}"] = b2_kev.paired_bootstrap(y, mean, s)
    tests = pd.DataFrame(tests).T
    tests.to_csv(OUTPUT_DIR / "d4_tests.csv")
    log_experiment("D4", run_name="D4_interpretable_models",
                   params={"selected": json.dumps({k: r.config for k, r in sel.iterrows()})},
                   metrics={f"pr_auc_{k}": float(v) for k, v in g.mean()["pr_auc"].items()},
                   artifacts=sorted(OUTPUT_DIR.glob("d4_*")), tags={"protocol": "v2", "study": "D"})
    return {"summary": summary, "tests": tests}


def load_model(path):
    """Load a cached D4 model. The final stage ran as `python -m`, so its pickles reference
    `__main__.EBM` etc.; expose the classes there before unpickling."""
    import __main__
    for cls in (EBM, RuleFit, SparseTree, WeightedCART):
        if not hasattr(__main__, cls.__name__):
            setattr(__main__, cls.__name__, cls)
    return pd.read_pickle(path)


def agreement() -> pd.DataFrame:
    """EBM global term importance joins D1's Kendall-tau ranking-agreement table (pre-registered)."""
    from scipy.stats import kendalltau

    d1 = pd.read_csv(OUTPUT_DIR / "d1_shap_mean_abs.csv", index_col="feature")
    perm = pd.read_csv(OUTPUT_DIR / "d1_permutation_importance.csv")
    perm = perm[~perm.unit.str.startswith("GROUP:")].pivot(index="unit", columns="model",
                                                             values="pr_auc_drop_mean")
    ebm = pd.read_csv(OUTPUT_DIR / "d4_ebm_term_importance.csv", index_col=0)["importance"]
    feats = d1.index
    rankings = {"SHAP M-XGB": d1["M-XGB"], "SHAP M-LR": d1["M-LR"],
                "PermImp M-XGB": perm["M-XGB"].reindex(feats), "PermImp M-LR": perm["M-LR"].reindex(feats),
                "EBM term importance": ebm.reindex(feats)}
    assert not rankings["EBM term importance"].isna().any(), "EBM terms must match D1 features"
    names = list(rankings)
    agree = pd.DataFrame({a: [kendalltau(rankings[a], rankings[b]).statistic for b in names] for a in names},
                         index=names)
    agree.to_csv(OUTPUT_DIR / "d4_global_ranking_agreement.csv")
    return agree


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["search", "final", "agreement", "all"], default="all")
    stage = parser.parse_args().stage
    pd.set_option("display.width", 250)
    if stage in ("search", "all"):
        print(search().to_string())
    if stage in ("final", "all"):
        out = final()
        print(out["summary"].to_string())
        print(out["tests"].round(4).to_string())
    if stage in ("agreement", "all"):
        print(agreement().round(3).to_string())
