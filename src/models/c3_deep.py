"""C3 - deep tabular models: MLP, TabNet, FT-Transformer (H4).

Design fixed in Study C addendum 2 (commit 59bd4b1), before any C3 model:
- inputs: median imputation + missing indicators + standard scaling, fitted
  on the training rows; unweighted BCE; no early stopping; epochs searched;
- search on the v2 fold, one fit each (seed 0): MLP 20 configs, TabNet 10,
  FT-Transformer 8 (budget set by a speed-only benchmark); select on
  validation PR-AUC; refit with 10 seeds, test once;
- H4 (deep models do NOT beat GBMs): best-validation deep model vs the C1-R
  best-validation GBM (XGBoost, cached test scores), paired bootstrap of
  PR-AUC; supported if the CI of deep - GBM is not entirely above 0.

Usage:  python -m src.models.c3_deep --stage tune|final
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
# torch MUST be imported before lightgbm/xgboost (pulled in via src.models):
# the other order loads two OpenMP runtimes and segfaults on macOS.
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import evaluate
from src.models import b2_kev, classical, study_c_data
from src.models.baselines import SEEDS

logger = logging.getLogger(__name__)
OUTPUT_DIR = Path("outputs")
CACHE_DIR = Path("data/processed/cache")
MODELS = ["mlp", "tabnet", "ft_transformer"]
N_CONFIGS = {"mlp": 20, "tabnet": 10, "ft_transformer": 8}
SEARCH_SEED = 0
torch.set_num_threads(classical.N_JOBS)


# ---------------------------------------------------------------- search spaces
def _log_uniform(rng, low, high):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def sample_config(model: str, rng: np.random.Generator) -> dict:
    if model == "mlp":
        return {"layers": int(rng.integers(1, 4)), "width": int(rng.choice([64, 128, 256])),
                "dropout": float(rng.uniform(0, 0.3)), "lr": _log_uniform(rng, 1e-4, 3e-3),
                "weight_decay": _log_uniform(rng, 1e-6, 1e-3), "epochs": int(rng.choice([5, 10, 20]))}
    if model == "tabnet":
        n = int(rng.choice([8, 16, 32]))
        return {"n_d": n, "n_a": n, "n_steps": int(rng.integers(3, 7)), "gamma": float(rng.uniform(1.0, 2.0)),
                "lambda_sparse": _log_uniform(rng, 1e-6, 1e-3), "lr": _log_uniform(rng, 5e-3, 3e-2),
                "epochs": int(rng.choice([5, 10, 20]))}
    if model == "ft_transformer":
        return {"d_token": int(rng.choice([32, 64])), "blocks": int(rng.choice([1, 2])),
                "attention_dropout": float(rng.uniform(0, 0.2)), "ffn_dropout": float(rng.uniform(0, 0.2)),
                "lr": _log_uniform(rng, 1e-4, 1e-3), "weight_decay": _log_uniform(rng, 1e-6, 1e-4),
                "epochs": int(rng.choice([2, 4, 6]))}
    raise ValueError(model)


# ---------------------------------------------------------------- networks
class FTTransformer(nn.Module):
    """Feature tokenizer + [CLS] + pre-norm Transformer (Gorishniy et al. 2021; GELU FFN)."""

    def __init__(self, n_features: int, d_token: int, blocks: int, attention_dropout: float, ffn_dropout: float):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.bias, a=5 ** 0.5)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        layer = nn.TransformerEncoderLayer(d_token, nhead=4, dim_feedforward=2 * d_token, dropout=ffn_dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        layer.self_attn.dropout = attention_dropout
        self.encoder = nn.TransformerEncoder(layer, blocks, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.ReLU(), nn.Linear(d_token, 1))

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.weight + self.bias
        tokens = torch.cat([self.cls.expand(len(x), -1, -1), tokens], dim=1)
        return self.head(self.encoder(tokens)[:, 0]).squeeze(-1)


def mlp(n_features: int, layers: int, width: int, dropout: float) -> nn.Module:
    blocks, d = [], n_features
    for _ in range(layers):
        blocks += [nn.Linear(d, width), nn.ReLU(), nn.Dropout(dropout)]
        d = width
    return nn.Sequential(*blocks, nn.Linear(d, 1), nn.Flatten(0))


def _train_torch(net: nn.Module, X: np.ndarray, y: np.ndarray, c: dict, seed: int, batch: int = 1024) -> nn.Module:
    gen = torch.Generator().manual_seed(seed)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))
    opt = torch.optim.AdamW(net.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    net.train()
    for _ in range(c["epochs"]):
        perm = torch.randperm(len(Xt), generator=gen)
        for i in range(0, len(Xt), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            nn.functional.binary_cross_entropy_with_logits(net(Xt[idx]), yt[idx]).backward()
            opt.step()
    return net


@torch.no_grad()
def _predict_torch(net: nn.Module, X: np.ndarray) -> np.ndarray:
    net.eval()
    Xt = torch.from_numpy(X)
    return np.concatenate([torch.sigmoid(net(Xt[i:i + 8192])).numpy() for i in range(0, len(Xt), 8192)])


def fit_predict(model: str, config: dict, seed: int, data: dict) -> np.ndarray:
    prep = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
    X_tr = prep.fit_transform(data["X_train"]).astype(np.float32)
    X_ev = prep.transform(data["X_eval"]).astype(np.float32)
    y_tr = data["y_train"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if model == "tabnet":
        from pytorch_tabnet.tab_model import TabNetClassifier

        clf = TabNetClassifier(n_d=config["n_d"], n_a=config["n_a"], n_steps=config["n_steps"],
                               gamma=config["gamma"], lambda_sparse=config["lambda_sparse"],
                               optimizer_params={"lr": config["lr"]}, seed=seed, verbose=0, device_name="cpu")
        clf.fit(X_tr, y_tr, max_epochs=config["epochs"], batch_size=4096, virtual_batch_size=256,
                drop_last=False)  # no eval_set -> no early stopping
        return clf.predict_proba(X_ev)[:, 1]
    if model == "mlp":
        net = mlp(X_tr.shape[1], config["layers"], config["width"], config["dropout"])
    elif model == "ft_transformer":
        net = FTTransformer(X_tr.shape[1], config["d_token"], config["blocks"], config["attention_dropout"],
                            config["ffn_dropout"])
    else:
        raise ValueError(model)
    return _predict_torch(_train_torch(net, X_tr, y_tr, config, seed), X_ev)


# ---------------------------------------------------------------- stages
def tune() -> pd.DataFrame:
    fold = study_c_data.load("v2fold")
    rows, val_preds = [], {}
    for model in MODELS:
        rng = np.random.default_rng([SEARCH_SEED, MODELS.index(model)])
        for i in range(N_CONFIGS[model]):
            config = sample_config(model, rng)
            start = time.time()
            p = fit_predict(model, config, 0, fold)
            val_preds[f"{model}__{i}"] = p
            rows.append({"model": model, "config_id": i, "config": json.dumps(config),
                         "val_pr_auc": average_precision_score(fold["y_eval"], p),
                         "seconds": round(time.time() - start, 1)})
            logger.info("C3 tune %s %d val PR-AUC %.4f (%.0fs)", model, i, rows[-1]["val_pr_auc"], rows[-1]["seconds"])
            pd.DataFrame(rows).to_csv(OUTPUT_DIR / "c3_tuning.csv", index=False)  # checkpoint
    pd.DataFrame(val_preds).assign(y=fold["y_eval"]).to_pickle(CACHE_DIR / "c3_val_predictions.pkl")
    return pd.DataFrame(rows)


def final() -> dict:
    from src.tracking import log_experiment

    tuning = pd.read_csv(OUTPUT_DIR / "c3_tuning.csv")
    chosen = tuning.loc[tuning.groupby("model")["val_pr_auc"].idxmax()].set_index("model")
    chosen.to_csv(OUTPUT_DIR / "c3_selected_configs.csv")
    best_deep = chosen["val_pr_auc"].idxmax()
    logger.info("Selected on validation: best deep model %s", best_deep)

    data = study_c_data.load("full")
    y = data["y_eval"]
    path = CACHE_DIR / "c3_test_predictions.pkl"
    preds = pd.read_pickle(path).drop(columns=["cve_id", "scoring_date", "y"]).to_dict("series") if path.exists() else {}
    for model in MODELS:
        config = json.loads(chosen.loc[model, "config"])
        for seed in SEEDS:
            key = f"{model}__s{seed}"
            if key in preds:  # checkpoint from an interrupted run
                continue
            start = time.time()
            preds[key] = fit_predict(model, config, seed, data)
            logger.info("C3 final %s seed %d (%.0fs)", model, seed, time.time() - start)
            pd.DataFrame({"cve_id": data["cve_eval"], "scoring_date": data["scoring_date_eval"], "y": y,
                          **preds}).to_pickle(path)
    rows = [{"model": m, "seed": s, **evaluate(y, np.asarray(preds[f"{m}__s{s}"]), is_probability=True)}
            for m in MODELS for s in SEEDS]
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "c3_results_by_seed.csv", index=False)
    metrics = ["pr_auc", "precision_at_10", "precision_at_100", "ndcg_at_100", "mrr", "roc_auc", "brier", "ece"]
    g = results.groupby("model")[metrics]
    summary = g.mean().round(4).astype(str) + " ± " + g.std(ddof=0).round(4).astype(str)
    summary = summary.loc[g.mean()["pr_auc"].sort_values(ascending=False).index]
    summary.to_csv(OUTPUT_DIR / "c3_summary.csv")

    c1r = pd.read_pickle(CACHE_DIR / "c1r_test_predictions.pkl")
    if not (c1r["cve_id"].to_numpy() == data["cve_eval"]).all():
        raise ValueError("C1-R predictions not aligned with the Study C test set")
    xgb = np.mean([c1r[f"xgboost_tuned__s{s}"] for s in SEEDS], axis=0)
    logreg = np.mean([c1r[f"logreg_tuned__s{s}"] for s in SEEDS], axis=0)
    mean_score = {m: np.mean([preds[f"{m}__s{s}"] for s in SEEDS], axis=0) for m in MODELS}
    tests = {f"H4_{best_deep}_vs_xgboost": b2_kev.paired_bootstrap(y, mean_score[best_deep], xgb)}
    for m in MODELS:
        if m != best_deep:
            tests[f"secondary_{m}_vs_xgboost"] = b2_kev.paired_bootstrap(y, mean_score[m], xgb)
        tests[f"secondary_{m}_vs_logreg"] = b2_kev.paired_bootstrap(y, mean_score[m], logreg)
        tests[f"secondary_{m}_vs_epss_pit"] = b2_kev.paired_bootstrap(y, mean_score[m], study_c_data.epss_pit_test_scores())
    tests_df = pd.DataFrame(tests).T
    tests_df.to_csv(OUTPUT_DIR / "c3_hypothesis_tests.csv")
    h4_row = tests[f"H4_{best_deep}_vs_xgboost"]
    h4 = {"best_deep": best_deep, "H4_supported": bool(h4_row["pr_auc_ci_low"] <= 0),
          "gbm_significantly_better": bool(h4_row["pr_auc_ci_high"] < 0)}

    means, stds = g.mean(), g.std(ddof=0)
    for m_name in MODELS:
        m = {f"{k}_mean": means.loc[m_name, k] for k in metrics}
        m.update({f"{k}_std": stds.loc[m_name, k] for k in metrics})
        m["val_pr_auc"] = float(chosen.loc[m_name, "val_pr_auc"])
        is_summary = m_name == best_deep
        if is_summary:
            m.update({f"{k}__{s}": v for k, d in tests.items() for s, v in d.items()})
            m.update({"H4_supported": float(h4["H4_supported"]),
                      "gbm_significantly_better": float(h4["gbm_significantly_better"])})
        log_experiment("C3", run_name=f"C3_{m_name}",
                       params={"model": m_name, "config": chosen.loc[m_name, "config"], "best_deep": best_deep,
                               "selection": "v2 fold, PR-AUC"},
                       metrics=m, artifacts=sorted(OUTPUT_DIR.glob("c3_*")) if is_summary else None,
                       tags={"protocol": "v2"})
    return {"summary": summary, "tests": tests_df, "h4": h4}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["tune", "final"], required=True)
    if parser.parse_args().stage == "tune":
        print(tune().groupby("model").val_pr_auc.describe().round(4))
    else:
        out = final()
        print(out["summary"].to_string())
        print(out["tests"].round(4).to_string())
        print(out["h4"])
