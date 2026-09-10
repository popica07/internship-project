"""Match equivalent checks across L2 tools.

The tools share no identifiers, so measuring cross-tool severity agreement
first requires pairing checks that describe the same control.

Why not join on CIS recommendation numbers: ScoutSuite ships a CIS 1.2.0
ruleset (`Recommendation 2.1`) while Prowler ships CIS 1.4/1.5/2.0/3.0, and
the sections were renumbered between those versions - CIS 1.2.0's 2.1
("CloudTrail enabled in all regions") is 3.1 from CIS 1.4 onward. Joining on
the raw numbers would silently produce wrong pairs, so it is not used.

Instead checks are paired by semantic similarity of their text, constrained
to the same cloud provider, using mutual-best-match so each check pairs at
most once. The similarity threshold is a free parameter, so A3 sweeps it
rather than trusting a single value.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"

_model_cache: dict[str, object] = {}


def _get_model(model_name: str = MODEL_NAME):
    """Load (and cache) the sentence-transformer model."""
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading sentence-transformer model %s", model_name)
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def check_text(df: pd.DataFrame) -> pd.Series:
    """Text used to represent a check for embedding."""
    title = df["title"].fillna("")
    description = df["description"].fillna("")
    return (title + ". " + description).str.strip().str.replace(r"\s+", " ", regex=True)


def embed_texts(texts: list[str], model_name: str = MODEL_NAME) -> np.ndarray:
    """L2-normalised embeddings, so a dot product is cosine similarity."""
    model = _get_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    # Pin the dtype so the similarity matmul is float32 whatever backend
    # produced the embeddings. Note: NumPy's BLAS backend on macOS emits
    # spurious overflow/invalid RuntimeWarnings from this matmul - verified
    # harmless (embedding norms are 1.0, no NaNs, max cosine 1.0).
    return np.asarray(embeddings, dtype=np.float32)


def embed_checks(df: pd.DataFrame, model_name: str = MODEL_NAME) -> np.ndarray:
    """Embed checks using their title + description."""
    return embed_texts(check_text(df).tolist(), model_name=model_name)


def match_tools(
    df: pd.DataFrame,
    tool_a: str,
    tool_b: str,
    threshold: float,
    require_same_service: bool = False,
    embeddings: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Pair checks from `tool_a` and `tool_b` by mutual-best semantic match.

    Candidate pairs must share a provider (and optionally a service). Each
    check appears in at most one pair, so the result can be treated as
    paired ratings for an agreement statistic.
    """
    a = df[df["tool"] == tool_a].reset_index(drop=True)
    b = df[df["tool"] == tool_b].reset_index(drop=True)
    if a.empty or b.empty:
        return pd.DataFrame()

    if embeddings and tool_a in embeddings and tool_b in embeddings:
        emb_a, emb_b = embeddings[tool_a], embeddings[tool_b]
    else:
        emb_a, emb_b = embed_checks(a), embed_checks(b)

    similarity = emb_a @ emb_b.T

    # Only compare checks describing the same provider (and service, if asked).
    same_provider = a["provider"].fillna("").values[:, None] == b["provider"].fillna("").values[None, :]
    mask = same_provider
    if require_same_service:
        mask = mask & (a["service"].fillna("").values[:, None] == b["service"].fillna("").values[None, :])
    similarity = np.where(mask, similarity, -1.0)

    best_for_a = similarity.argmax(axis=1)
    best_for_b = similarity.argmax(axis=0)

    rows = []
    for i, j in enumerate(best_for_a):
        # Mutual best match, above threshold.
        if best_for_b[j] != i or similarity[i, j] < threshold:
            continue
        rows.append(
            {
                "similarity": float(similarity[i, j]),
                "provider": a.at[i, "provider"],
                f"{tool_a}_catalog_id": a.at[i, "catalog_id"],
                f"{tool_a}_title": a.at[i, "title"],
                f"{tool_a}_service": a.at[i, "service"],
                f"{tool_a}_severity": a.at[i, "severity_normalised"],
                f"{tool_a}_severity_raw": a.at[i, "severity_raw"],
                f"{tool_b}_catalog_id": b.at[j, "catalog_id"],
                f"{tool_b}_title": b.at[j, "title"],
                f"{tool_b}_service": b.at[j, "service"],
                f"{tool_b}_severity": b.at[j, "severity_normalised"],
                f"{tool_b}_severity_raw": b.at[j, "severity_raw"],
            }
        )

    pairs = pd.DataFrame(rows).sort_values("similarity", ascending=False).reset_index(drop=True)
    logger.info(
        "Matched %d %s<->%s pairs at threshold %.2f%s",
        len(pairs),
        tool_a,
        tool_b,
        threshold,
        " (same service required)" if require_same_service else "",
    )
    return pairs
