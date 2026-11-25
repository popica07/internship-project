"""Shared pieces for Study D (design: Study D addendum, commit c0a3ee1).

- the two explained models, refitted exactly as C1-R and checked against the
  cached C1-R test predictions (M-LR, M-XGB seeds 0-9);
- the pre-registered explanation sets (E-rand, E-top, E-kev, G-sample) and the
  shared SHAP background;
- SHAP in log-odds for both models (exact linear SHAP for LR, interventional
  TreeSHAP for XGBoost);
- text-component labels (top TF-IDF terms) and feature groups.
"""

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.models import classical, study_c_data
from src.models.baselines import SEEDS
from src.models.c4_ablation import c1r_configs

logger = logging.getLogger(__name__)
OUTPUT_DIR = Path("outputs")
CACHE_DIR = Path("data/processed/cache")
N_RAND, N_TOP_PER_MONTH, N_GLOBAL, N_BACKGROUND = 1_000, 10, 10_000, 1_000


def data() -> dict:
    return study_c_data.load("full")


def feature_names() -> list[str]:
    return list(data()["X_train"].columns)


def group_of(feature: str) -> str:
    return feature.split("_")[0].replace("epss", "EPSS").replace("cvss", "CVSS")


def groups() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for f in feature_names():
        out.setdefault(group_of(f), []).append(f)
    return out


# ---------------------------------------------------------------- models
def _check(name: str, pred: np.ndarray, column: str):
    cached = pd.read_pickle(CACHE_DIR / "c1r_test_predictions.pkl")[column].to_numpy()
    if not np.allclose(pred, cached, rtol=1e-5, atol=1e-9):
        raise ValueError(f"{name} refit does not reproduce C1-R ({column}): max diff {np.abs(pred - cached).max()}")


@lru_cache(maxsize=None)
def model(family: str, seed: int = 0):
    """C1-R selected configuration refitted on the full training set (cached on disk)."""
    path = CACHE_DIR / f"d_model_{family}_s{seed}.pkl"
    d = data()
    if path.exists():
        m = pd.read_pickle(path)
    else:
        logger.info("D: refitting %s seed %d", family, seed)
        m = classical.make_model(family, c1r_configs()[family], seed).fit(d["X_train"], d["y_train"])
        pd.to_pickle(m, path)
    _check(f"{family} s{seed}", classical.predict(m, d["X_eval"]), f"{family}_tuned__s{seed}")
    return m


def xgb_seeds() -> list:
    return [model("xgboost", s) for s in SEEDS]


# ---------------------------------------------------------------- sets
def complete_case(X: pd.DataFrame) -> np.ndarray:
    return (X["cvss_score"].notna() & X["epss_pit"].notna()).to_numpy()


@lru_cache(maxsize=None)
def explanation_sets() -> dict[str, np.ndarray]:
    """Row indices into the test set, drawn once with seed 0 (pre-registered)."""
    d = data()
    rng = np.random.default_rng(0)
    cc = np.flatnonzero(complete_case(d["X_eval"]))
    e_rand = np.sort(rng.choice(cc, N_RAND, replace=False))
    g_sample = np.sort(rng.choice(len(d["y_eval"]), N_GLOBAL, replace=False))
    xgb_mean = np.mean([pd.read_pickle(CACHE_DIR / "c1r_test_predictions.pkl")[f"xgboost_tuned__s{s}"]
                        for s in SEEDS], axis=0)
    frame = pd.DataFrame({"i": np.arange(len(xgb_mean)), "month": d["group_eval"], "s": xgb_mean})
    frame = frame[complete_case(d["X_eval"])]
    e_top = np.sort(frame.sort_values("s", ascending=False).groupby("month").head(N_TOP_PER_MONTH)["i"].to_numpy())
    e_kev = np.sort(np.intersect1d(cc, np.flatnonzero(d["y_eval"] == 1)))
    background = np.sort(np.random.default_rng([0, 1]).choice(len(d["y_train"]), N_BACKGROUND, replace=False))
    return {"E-rand": e_rand, "E-top": e_top, "E-kev": e_kev, "G-sample": g_sample, "background": background}


def background() -> pd.DataFrame:
    return data()["X_train"].iloc[explanation_sets()["background"]]


# ---------------------------------------------------------------- SHAP (log-odds)
def lr_shap(pipe, X: pd.DataFrame, bg: pd.DataFrame | None = None) -> np.ndarray:
    """Exact SHAP of a linear model in its imputed, scaled input space; indicator
    columns folded into their source feature. Returns (n, 69) in the raw feature order."""
    bg = background() if bg is None else bg
    pre = pipe[:-1]
    coef = pipe[-1].coef_.ravel()
    z, zbg = pre.transform(X), pre.transform(bg)
    phi = coef * (z - zbg.mean(axis=0))
    names = pipe[0].get_feature_names_out()
    out = pd.DataFrame(0.0, index=range(len(X)), columns=feature_names())
    for j, n in enumerate(names):
        out[n.removeprefix("missingindicator_")] += phi[:, j]
    return out.to_numpy()


def xgb_shap(m, X: pd.DataFrame, bg: pd.DataFrame | None = None) -> np.ndarray:
    import shap

    bg = background() if bg is None else bg
    # The Booster, not the sklearn wrapper: XGBoost 3 sets enable_categorical=True by default and
    # SHAP 0.52 then refuses interventional TreeSHAP although every feature here is float.
    # The masker keeps all 1,000 background rows (SHAP would otherwise subsample to 100).
    explainer = shap.TreeExplainer(m.get_booster(), data=shap.maskers.Independent(bg, max_samples=len(bg)),
                                   feature_perturbation="interventional", model_output="raw")
    return explainer.shap_values(X, check_additivity=False)


def shap_values(family: str, X: pd.DataFrame, seed: int = 0) -> np.ndarray:
    m = model(family, seed)
    return lr_shap(m, X) if family == "logreg" else xgb_shap(m, X)


# ---------------------------------------------------------------- text labels
def text_terms(n_terms: int = 10) -> pd.DataFrame:
    path = OUTPUT_DIR / "d_text_component_terms.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0)
    from src.features import FeatureBuilder, load_l1, temporal_task

    task = temporal_task(load_l1(), classical.SPLIT_DATE)
    b = FeatureBuilder(use_epss=False).fit(task["train"])
    check = b.transform(task["train"].iloc[:2000]).filter(like="text_").to_numpy()
    if not np.allclose(check, data()["X_train"].filter(like="text_").to_numpy()[:2000]):
        raise ValueError("text model refit differs from cached features")
    terms = np.array(b.tfidf.get_feature_names_out())
    rows = {}
    for i, comp in enumerate(b.svd.components_):
        top = np.argsort(-np.abs(comp))[:n_terms]
        rows[f"text_{i:02d}"] = {"terms": ", ".join(f"{terms[j]}{'(-)' if comp[j] < 0 else ''}" for j in top)}
    out = pd.DataFrame(rows).T
    OUTPUT_DIR.mkdir(exist_ok=True)
    out.to_csv(path)
    return out


def label(feature: str, short: bool = True) -> str:
    if not feature.startswith("text_"):
        return feature
    terms = text_terms().loc[feature, "terms"].split(", ")
    return f"{feature} [{', '.join(terms[:3] if short else terms)}]"


# ---------------------------------------------------------------- L3 (SIMULATED)
L3_FORMULA_FEATURES = ["source_layer", "base_severity", "epss_percentile", "in_kev", "public_exploit",
                       "criticality_tier", "internet_facing", "regulatory_scope", "pci_scope", "dora_scope",
                       "sla_breached", "recurrence_count"]


def l3_frame() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """C4's L3 matrix (finding + context features), raw rows, High/Critical target."""
    from src.models.c4_ablation import CONTEXT, FINDING, L3_PATH

    df = pd.read_csv(L3_PATH, low_memory=False)
    X = df[FINDING + CONTEXT].copy()
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype("category")
        elif X[c].dtype == bool:
            X[c] = X[c].astype(int)
    y = df["priority_label"].isin(["High", "Critical"]).astype(int).to_numpy()
    return X, df, y


def l3_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import GroupShuffleSplit

    return next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(df, groups=df["asset_id"]))


@lru_cache(maxsize=None)
def l3_model():
    """XGBoost library defaults, grouped-by-asset split seed 0 (as C4's L3 ablation, 'all')."""
    from xgboost import XGBClassifier

    X, df, y = l3_frame()
    tr, _ = l3_split(df)
    return XGBClassifier(tree_method="hist", enable_categorical=True, n_jobs=classical.N_JOBS,
                         random_state=0).fit(X.iloc[tr], y[tr])


def l3_formula_terms(rows: pd.DataFrame) -> pd.DataFrame:
    """The documented L3 label formula, term by term, in priority-score units, from a row's own
    columns (so a counterfactual row can be re-scored). Checked against `priority_score`."""
    from src.simulator.config import SimulatorConfig

    c = SimulatorConfig()
    vuln = (rows["source_layer"].astype(str) == "L1").to_numpy()
    sev = dict(c.misconfig_severity_map)
    rw, cw = c.real_component_weight, c.context_component_weight
    f = lambda col: rows[col].astype(float).fillna(0.0).to_numpy()  # noqa: E731
    t = pd.DataFrame(index=rows.index)
    t["epss_percentile"] = np.where(vuln, rw * c.w_epss * f("epss_percentile"), 0.0)
    t["in_kev"] = np.where(vuln, rw * c.w_in_kev * f("in_kev"), 0.0)
    t["public_exploit"] = np.where(vuln, rw * c.w_public_exploit * f("public_exploit"), 0.0)
    t["base_severity"] = np.where(vuln, 0.0, rw * c.misconfig_scale *
                                  rows["base_severity"].astype(str).str.lower().map(sev).fillna(0.0).to_numpy())
    t["criticality_tier"] = cw * c.w_tier * np.array(c.tier_weights)[rows["criticality_tier"].astype(int).to_numpy()]
    t["internet_facing"] = cw * c.w_internet_facing * f("internet_facing")
    t["regulatory_scope"] = cw * c.w_regulatory * f("regulatory_scope")
    t["sla_breached"] = cw * c.w_sla_breached * f("sla_breached")
    t["recurrence_count"] = cw * c.w_recurrence * np.minimum(f("recurrence_count") / c.recurrence_cap, 1.0)
    return t


def l3_formula_label(rows: pd.DataFrame) -> np.ndarray:
    from src.simulator.config import SimulatorConfig

    return (l3_formula_terms(rows).sum(axis=1).to_numpy() >= SimulatorConfig().threshold_high).astype(int)
