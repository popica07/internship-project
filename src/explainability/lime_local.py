"""LIME as pre-registered for D2/D5 (Study D addendum, c0a3ee1).

`LimeTabularExplainer`, continuous features not discretised,
`sample_around_instance=True`, 5,000 samples, all 69 features, default kernel
width, explaining P(KEV); training statistics from complete-case training rows.
The attribution is LIME's weight vector as returned by `explain_instance`.
"""

import numpy as np
import pandas as pd

from src.explainability import common as c

N_SAMPLES = 5_000


def explainer(seed: int):
    from lime.lime_tabular import LimeTabularExplainer

    X = c.data()["X_train"]
    X = X[c.complete_case(X) & X.notna().all(axis=1).to_numpy()]
    return LimeTabularExplainer(X.to_numpy(), mode="classification", feature_names=list(X.columns),
                                discretize_continuous=False, sample_around_instance=True, random_state=seed)


def predict_fn(family: str, seed: int = 0):
    m = c.model(family, seed)
    names = c.feature_names()
    return lambda a: m.predict_proba(pd.DataFrame(a, columns=names))


def explain(family: str, X: pd.DataFrame, lime_seed: int = 0, model_seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Returns (n, 69) LIME weights and the local R^2 per instance, in X's row order."""
    ex, fn = explainer(lime_seed), predict_fn(family, model_seed)
    k = X.shape[1]
    W, r2 = np.zeros((len(X), k)), np.zeros(len(X))
    for i, row in enumerate(X.to_numpy()):
        e = ex.explain_instance(row, fn, labels=(1,), num_features=k, num_samples=N_SAMPLES)
        for j, w in e.local_exp[1]:
            W[i, j] = w
        r2[i] = e.score
    return W, r2
