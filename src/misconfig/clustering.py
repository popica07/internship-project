"""A4 - remediation family clustering (HDBSCAN over check embeddings).

Usage:
    python -m src.misconfig.clustering

The L2 catalog holds 3,870 individual checks, but an engineer does not
remediate 3,870 different ways: "enable encryption at rest" is one action
that recurs across dozens of services and all three tools. A4 recovers those
*remediation families* so later studies can prioritise at the level of the
action a team actually takes rather than the individual finding.

Method: embed each check's remediation-oriented text, reduce with UMAP, then
cluster with HDBSCAN (which finds variable-density clusters and, unlike
k-means, does not force every check into a family - genuinely unique checks
stay as noise).

Two things are swept rather than assumed, per CLAUDE.md rule 5:
  * `min_cluster_size` - the main structural knob;
  * the random seed - UMAP is stochastic, so cluster stability across seeds
    is measured with the Adjusted Rand Index and reported as mean +/- std.

Why UMAP is kept despite being the only stochastic step: clustering the
384-dimensional embeddings directly is perfectly reproducible (ARI 1.0 between
runs) but degenerate - it finds only 2-6 clusters with a silhouette of
0.02-0.05, the curse of dimensionality. UMAP first gives 138 families at
silhouette 0.66. The cost is that family *membership* is only moderately
reproducible across seeds (ARI ~0.53) even though the aggregate structure is
stable (138 +/- 3 families, 16.3% +/- 0.7% noise). Downstream work should treat
a family as a useful grouping, not as a hard object with a stable identity.

Caveat carried into A5/A6: only 46% of checks ship remediation text (Prowler
mostly), so for the rest the family is inferred from title and description
alone. Families are therefore "what the check is about" more than "the exact
fix" wherever remediation text is missing.
"""

import logging
from pathlib import Path

import hdbscan
import matplotlib
import numpy as np
import pandas as pd
import umap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.misconfig.matching import embed_texts  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

CATALOG_PATH = Path("data/processed/l2_misconfig_catalog.csv")
OUTPUT_DIR = Path("outputs")
ASSIGNMENTS_PATH = Path("data/processed/l2_remediation_families.csv")

SERIES_BLUE = "#2a78d6"
SERIES_ORANGE = "#eb6834"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"

MIN_CLUSTER_SIZES = [5, 10, 15, 20, 30]
PRIMARY_MIN_CLUSTER_SIZE = 10
SEEDS = list(range(10))  # rule 5: never a single run
UMAP_COMPONENTS = 10


def remediation_text(df: pd.DataFrame) -> pd.Series:
    """Text describing what the check is about and how it is fixed.

    Remediation text is included where the tool ships it (Prowler mostly);
    otherwise the family has to be inferred from title + description.
    """
    parts = [
        df["title"].fillna(""),
        df["description"].fillna(""),
        df["remediation_text"].fillna(""),
    ]
    combined = parts[0]
    for part in parts[1:]:
        combined = combined + ". " + part
    return combined.str.replace(r"\s+", " ", regex=True).str.strip(". ").str.strip()


def reduce_and_cluster(
    embeddings: np.ndarray, min_cluster_size: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """UMAP -> HDBSCAN. Returns (labels, 2D projection for plotting)."""
    reducer = umap.UMAP(
        n_components=UMAP_COMPONENTS,
        n_neighbors=15,
        min_dist=0.0,  # tight packing helps density clustering
        metric="cosine",
        random_state=seed,
    )
    reduced = reducer.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(reduced)
    return labels, reduced


def sweep_min_cluster_size(embeddings: np.ndarray, seed: int = 0) -> pd.DataFrame:
    """How the family structure responds to the main knob."""
    rows = []
    for size in MIN_CLUSTER_SIZES:
        labels, reduced = reduce_and_cluster(embeddings, size, seed)
        clustered = labels != -1
        n_clusters = len(set(labels[clustered]))
        silhouette = (
            silhouette_score(reduced[clustered], labels[clustered])
            if n_clusters > 1 and clustered.sum() > n_clusters
            else float("nan")
        )
        rows.append(
            {
                "min_cluster_size": size,
                "n_families": n_clusters,
                "noise_pct": round(100 * (~clustered).mean(), 1),
                "median_family_size": int(np.median(np.bincount(labels[clustered]))) if n_clusters else 0,
                "silhouette": round(silhouette, 3),
            }
        )
        logger.info("min_cluster_size=%d -> %d families, %.1f%% noise", size, n_clusters, rows[-1]["noise_pct"])
    return pd.DataFrame(rows)


def seed_stability(embeddings: np.ndarray, min_cluster_size: int) -> tuple[pd.DataFrame, dict]:
    """Re-cluster under several seeds and measure agreement between runs.

    UMAP is stochastic, so a single run's families could be an artefact.
    Pairwise Adjusted Rand Index across seeds says how reproducible they are.
    """
    runs = []
    for seed in SEEDS:
        labels, _ = reduce_and_cluster(embeddings, min_cluster_size, seed)
        clustered = labels != -1
        runs.append(
            {
                "seed": seed,
                "labels": labels,
                "n_families": len(set(labels[clustered])),
                "noise_pct": 100 * (~clustered).mean(),
            }
        )

    aris = [
        adjusted_rand_score(a["labels"], b["labels"])
        for i, a in enumerate(runs)
        for b in runs[i + 1 :]
    ]
    summary = pd.DataFrame([{k: v for k, v in r.items() if k != "labels"} for r in runs])
    stats = {
        "n_families_mean": summary["n_families"].mean(),
        "n_families_std": summary["n_families"].std(),
        "noise_pct_mean": summary["noise_pct"].mean(),
        "noise_pct_std": summary["noise_pct"].std(),
        "ari_mean": float(np.mean(aris)),
        "ari_std": float(np.std(aris)),
    }
    return summary, stats


def label_families(df: pd.DataFrame, texts: pd.Series, labels: np.ndarray, top_n: int = 5) -> pd.DataFrame:
    """Name each family by its most distinctive terms (mean TF-IDF)."""
    vectoriser = TfidfVectorizer(max_features=6000, stop_words="english", ngram_range=(1, 2))
    matrix = vectoriser.fit_transform(texts)
    vocabulary = np.array(vectoriser.get_feature_names_out())

    rows = []
    for family in sorted(set(labels[labels != -1])):
        members = labels == family
        mean_tfidf = np.asarray(matrix[members].mean(axis=0)).ravel()
        terms = vocabulary[mean_tfidf.argsort()[::-1][:top_n]]
        subset = df[members]
        rows.append(
            {
                "family": family,
                "size": int(members.sum()),
                "top_terms": ", ".join(terms),
                "tools": ", ".join(sorted(subset["tool"].unique())),
                "n_tools": subset["tool"].nunique(),
                "providers": ", ".join(sorted(subset["provider"].dropna().unique())[:5]),
                "n_providers": subset["provider"].nunique(),
                "example_check": subset["title"].iloc[0],
            }
        )
    return pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d5d4d0")
    ax.tick_params(colors=INK_SECONDARY, length=0)


def _titles(ax, title: str, subtitle: str) -> None:
    ax.set_title(title, color=INK_PRIMARY, fontsize=12, fontweight="bold", loc="left", pad=26)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_SECONDARY, fontsize=9.5, va="bottom")


def _truncate_terms(terms: str, limit: int = 50) -> str:
    """Trim a term list on a comma boundary, never mid-word."""
    if len(terms) <= limit:
        return terms
    kept = []
    length = 0
    for term in terms.split(", "):
        if length + len(term) + 2 > limit:
            break
        kept.append(term)
        length += len(term) + 2
    return ", ".join(kept or [terms.split(", ")[0]]) + "…"


def plot_family_sizes(families: pd.DataFrame, path: Path, top: int = 15) -> None:
    """Largest remediation families - magnitude, so bars, one hue."""
    data = families.head(top).iloc[::-1]
    labels = [_truncate_terms(t) for t in data["top_terms"]]

    fig, ax = plt.subplots(figsize=(9.5, 6.2), facecolor=SURFACE)
    ax.barh(range(len(data)), data["size"], color=SERIES_BLUE, height=0.72)
    ax.set_yticks(range(len(data)), labels, fontsize=9)

    for i, value in enumerate(data["size"]):
        ax.text(value + max(data["size"]) * 0.01, i, str(value), va="center", fontsize=9, color=INK_SECONDARY)

    _style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("checks in family", color=INK_SECONDARY, fontsize=10)
    _titles(ax, f"The {top} largest remediation families", "Families named by their most distinctive terms (TF-IDF)")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_sweep(sweep: pd.DataFrame, path: Path) -> None:
    """Two measures, two panels - never two y-axes on one plot."""
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.5, 5.6), sharex=True, facecolor=SURFACE, gridspec_kw={"hspace": 0.35}
    )

    top.plot(sweep["min_cluster_size"], sweep["n_families"], color=SERIES_BLUE, linewidth=2)
    top.scatter(sweep["min_cluster_size"], sweep["n_families"], s=64, color=SERIES_BLUE,
                edgecolor=SURFACE, linewidth=2, zorder=3)
    _style_axes(top)
    top.set_ylabel("families found", color=INK_SECONDARY, fontsize=10)
    _titles(top, "Family count falls fivefold, but the unclustered share barely moves",
            "HDBSCAN over UMAP-reduced check embeddings, seed 0")

    bottom.plot(sweep["min_cluster_size"], sweep["noise_pct"], color=SERIES_ORANGE, linewidth=2)
    bottom.scatter(sweep["min_cluster_size"], sweep["noise_pct"], s=64, color=SERIES_ORANGE,
                   edgecolor=SURFACE, linewidth=2, zorder=3)
    _style_axes(bottom)
    bottom.set_ylabel("unclustered (%)", color=INK_SECONDARY, fontsize=10)
    bottom.set_xlabel("min_cluster_size", color=INK_SECONDARY, fontsize=10)

    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def run() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CATALOG_PATH)
    texts = remediation_text(df)

    logger.info("Embedding %d checks", len(df))
    embeddings = embed_texts(texts.tolist())

    sweep = sweep_min_cluster_size(embeddings)
    sweep.to_csv(OUTPUT_DIR / "a4_min_cluster_size_sweep.csv", index=False)

    logger.info("Measuring seed stability over %d seeds", len(SEEDS))
    seed_summary, stability = seed_stability(embeddings, PRIMARY_MIN_CLUSTER_SIZE)
    seed_summary.to_csv(OUTPUT_DIR / "a4_seed_stability.csv", index=False)

    labels, _ = reduce_and_cluster(embeddings, PRIMARY_MIN_CLUSTER_SIZE, seed=SEEDS[0])
    families = label_families(df, texts, labels)
    families.to_csv(OUTPUT_DIR / "a4_remediation_families.csv", index=False)

    assignments = df[["catalog_id", "tool", "provider", "service", "title", "severity_normalised"]].copy()
    assignments["family"] = labels
    assignments["family_terms"] = assignments["family"].map(
        families.set_index("family")["top_terms"]
    ).fillna("(unclustered)")
    assignments.to_csv(ASSIGNMENTS_PATH, index=False)

    plot_family_sizes(families, OUTPUT_DIR / "a4_family_sizes.png")
    plot_sweep(sweep, OUTPUT_DIR / "a4_min_cluster_size_sweep.png")

    from src.tracking import log_experiment

    primary = sweep.set_index("min_cluster_size").loc[PRIMARY_MIN_CLUSTER_SIZE]
    log_experiment(
        "A4",
        params={"min_cluster_size": PRIMARY_MIN_CLUSTER_SIZE, "seeds": list(SEEDS)},
        metrics={
            "n_families": int((families["family"] >= 0).sum()),
            "silhouette": primary["silhouette"],
            "noise_pct": primary["noise_pct"],
            **{f"stability_{k}": v for k, v in stability.items()},
        },
        artifacts=sorted(OUTPUT_DIR.glob("a4_*")),
    )

    return {
        "df": df,
        "labels": labels,
        "sweep": sweep,
        "families": families,
        "assignments": assignments,
        "stability": stability,
        "seed_summary": seed_summary,
    }


def log_stats(results: dict) -> None:
    families = results["families"]
    stability = results["stability"]
    labels = results["labels"]
    assignments = results["assignments"]

    cross_tool = families[families["n_tools"] > 1]
    cross_provider = families[families["n_providers"] > 1]

    print()
    print("=" * 72)
    print("A4 remediation family clustering")
    print("=" * 72)
    print(f"Checks clustered:        {len(assignments):,}")
    print(f"Families found:          {len(families):,}  (min_cluster_size={PRIMARY_MIN_CLUSTER_SIZE})")
    print(f"Unclustered (noise):     {(labels == -1).sum():,} ({(labels == -1).mean():.1%})")
    print(f"Largest family:          {families['size'].max():,} checks")
    print(f"Median family size:      {families['size'].median():.0f} checks")
    print()
    print(f"Stability over {len(SEEDS)} seeds (UMAP is stochastic):")
    print(f"  families:   {stability['n_families_mean']:.1f} ± {stability['n_families_std']:.1f}")
    print(f"  noise:      {stability['noise_pct_mean']:.1f}% ± {stability['noise_pct_std']:.1f}%")
    print(f"  pairwise ARI: {stability['ari_mean']:.3f} ± {stability['ari_std']:.3f}")
    print()
    print(f"Families spanning >1 tool:     {len(cross_tool):,} of {len(families):,}")
    print(f"Families spanning >1 provider: {len(cross_provider):,} of {len(families):,}")
    print()
    print("Parameter sweep:")
    print(results["sweep"].to_string(index=False))
    print()
    print("10 largest families:")
    print(families.head(10)[["family", "size", "n_tools", "n_providers", "top_terms"]].to_string(index=False))
    print()
    print(f"Written to {ASSIGNMENTS_PATH} and {OUTPUT_DIR}/a4_*")
    print("=" * 72)


def main() -> None:
    results = run()
    log_stats(results)


if __name__ == "__main__":
    main()
