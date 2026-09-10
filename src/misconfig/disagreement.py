"""A3 - cross-tool severity disagreement analysis.

Usage:
    python -m src.misconfig.disagreement

Measures how far the L2 tools disagree about how severe the same control is.

Three constraints shape the design, all of them findings in their own right:

1. Checkov's open-source repo ships no severities at all, so it cannot take
   part in a severity agreement statistic. Agreement is therefore measured
   between Prowler and ScoutSuite only.
2. Prowler rates on five levels, ScoutSuite on two (`danger`/`warning`).
   Kappa over mismatched scales would measure the scale mapping as much as
   the tools, so the primary statistic collapses both onto a shared binary
   scale (high-risk vs lower). The full 5x2 cross-tab is also reported.
3. The tools share no identifiers, so pairs come from semantic matching
   (see matching.py) with a free similarity threshold. Kappa is reported
   across a sweep of thresholds rather than at one hand-picked value.
"""

import logging
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # headless: figures are written to outputs/, never shown
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from src.evaluation.statistical import interpret_kappa, kappa_with_ci  # noqa: E402
from src.misconfig.matching import embed_checks, match_tools  # noqa: E402

# Single-hue sequential ramp (light -> dark) and one categorical hue, from the
# project's validated palette. Sequential encoding is one hue by rule - never a
# rainbow colormap, which invents boundaries the data does not have.
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SERIES_BLUE = "#2a78d6"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

CATALOG_PATH = Path("data/processed/l2_misconfig_catalog.csv")
OUTPUT_DIR = Path("outputs")

TOOL_A = "prowler"
TOOL_B = "scoutsuite"

# Shared coarse scale. ScoutSuite cannot express critical/low, so this is the
# finest granularity on which the two tools can be compared honestly.
HIGH_RISK = {"critical", "high"}
LOWER = {"medium", "low", "informational"}

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
PRIMARY_THRESHOLD = 0.65


def collapse_severity(severity: str) -> str | None:
    if severity in HIGH_RISK:
        return "high_risk"
    if severity in LOWER:
        return "lower"
    return None


def severity_availability(df: pd.DataFrame) -> pd.DataFrame:
    """Per-tool severity coverage - why Checkov is excluded from kappa."""
    rows = []
    for tool, group in df.groupby("tool"):
        usable = (group["severity_normalised"] != "unknown").sum()
        rows.append(
            {
                "tool": tool,
                "checks": len(group),
                "with_severity": int(usable),
                "pct_with_severity": round(100 * usable / len(group), 1),
                "distinct_levels": group.loc[
                    group["severity_normalised"] != "unknown", "severity_raw"
                ].nunique(),
            }
        )
    return pd.DataFrame(rows).sort_values("checks", ascending=False).reset_index(drop=True)


def threshold_sweep(df: pd.DataFrame, embeddings: dict) -> tuple[pd.DataFrame, dict]:
    """Kappa (with bootstrap CI) at each similarity threshold."""
    rows = []
    pairs_by_threshold = {}

    for threshold in THRESHOLDS:
        pairs = match_tools(df, TOOL_A, TOOL_B, threshold, embeddings=embeddings)
        pairs_by_threshold[threshold] = pairs
        if pairs.empty:
            rows.append({"threshold": threshold, "n_pairs": 0, "kappa": float("nan")})
            continue

        rated = pairs.assign(
            a=pairs[f"{TOOL_A}_severity"].map(collapse_severity),
            b=pairs[f"{TOOL_B}_severity"].map(collapse_severity),
        ).dropna(subset=["a", "b"])

        result = kappa_with_ci(rated["a"], rated["b"])
        agreement = (rated["a"] == rated["b"]).mean() if len(rated) else float("nan")
        rows.append(
            {
                "threshold": threshold,
                "n_pairs": len(pairs),
                "n_rated": result["n"],
                "raw_agreement": round(agreement, 3),
                "kappa": round(result["kappa"], 3),
                "ci_low": round(result["ci_low"], 3),
                "ci_high": round(result["ci_high"], 3),
                "interpretation": interpret_kappa(result["kappa"]),
            }
        )

    return pd.DataFrame(rows), pairs_by_threshold


def _titles(ax, title: str, subtitle: str) -> None:
    """Left-aligned title + subtitle anchored to the axes, so they can't clip."""
    ax.set_title(title, color=INK_PRIMARY, fontsize=12, fontweight="bold", loc="left", pad=26)
    ax.text(
        0, 1.02, subtitle, transform=ax.transAxes,
        color=INK_SECONDARY, fontsize=9.5, va="bottom",
    )


def _ink_on(rgba) -> str:
    """Pick ink or white for a cell label from the cell's own luminance."""
    r, g, b = rgba[:3]
    channels = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    luminance = 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]
    return INK_PRIMARY if luminance > 0.45 else "#ffffff"


def _style_axes(ax) -> None:
    """Recessive grid and axes so the data carries the emphasis."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d5d4d0")
    ax.tick_params(colors=INK_SECONDARY, length=0)


def plot_kappa_sweep(sweep: pd.DataFrame, path: Path) -> None:
    """Kappa against the matching threshold, with its bootstrap interval.

    The threshold is a free parameter of the matcher, so the point of this
    figure is that the conclusion does not hinge on one chosen value.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)

    ax.fill_between(
        sweep["threshold"], sweep["ci_low"], sweep["ci_high"],
        color=SERIES_BLUE, alpha=0.15, linewidth=0,
    )
    ax.plot(sweep["threshold"], sweep["kappa"], color=SERIES_BLUE, linewidth=2, zorder=3)
    ax.scatter(
        sweep["threshold"], sweep["kappa"],
        s=64, color=SERIES_BLUE, edgecolor=SURFACE, linewidth=2, zorder=4,
    )
    ax.axhline(0, color="#b9b8b3", linewidth=1, linestyle=(0, (4, 3)), zorder=1)

    # Annotate n only at the ends - never a label on every point.
    for i in (0, len(sweep) - 1):
        row = sweep.iloc[i]
        ax.annotate(
            f"n={int(row['n_pairs'])}",
            (row["threshold"], row["kappa"]),
            textcoords="offset points", xytext=(0, 14),
            ha="center", fontsize=9, color=INK_SECONDARY,
        )

    _style_axes(ax)
    ax.set_xlabel("Semantic similarity threshold", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Cohen's κ", color=INK_SECONDARY, fontsize=10)
    _titles(
        ax,
        "Severity agreement stays slight at every threshold",
        "Prowler vs ScoutSuite, binary scale (high-risk vs lower); band is the 95% bootstrap CI",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_crosstab(crosstab: pd.DataFrame, path: Path) -> None:
    """Prowler's five levels against ScoutSuite's two, as a counts heatmap."""
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
    order = [s for s in ["critical", "high", "medium", "low", "informational"] if s in crosstab.index]
    data = crosstab.loc[order]

    fig, ax = plt.subplots(figsize=(6.2, 4.4), facecolor=SURFACE)
    im = ax.imshow(data.values, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(data.columns)), data.columns, color=INK_SECONDARY)
    ax.set_yticks(range(len(data.index)), data.index, color=INK_SECONDARY)

    # Direct labels: the cell counts are the point of the figure.
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data.values[i, j]
            ax.text(
                j, i, str(value), ha="center", va="center", fontsize=11,
                color=_ink_on(im.cmap(im.norm(value))),
            )

    ax.set_xlabel("ScoutSuite severity", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Prowler severity", color=INK_SECONDARY, fontsize=10)
    _titles(
        ax,
        "The tools' severity scales do not line up",
        "Matched check pairs at similarity ≥ 0.65",
    )
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    fig.colorbar(im, ax=ax, label="matched pairs").outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def run() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CATALOG_PATH)

    availability = severity_availability(df)
    availability.to_csv(OUTPUT_DIR / "a3_severity_availability.csv", index=False)

    # Embed once and reuse across the whole threshold sweep.
    embeddings = {
        tool: embed_checks(df[df["tool"] == tool].reset_index(drop=True))
        for tool in (TOOL_A, TOOL_B)
    }

    sweep, pairs_by_threshold = threshold_sweep(df, embeddings)
    sweep.to_csv(OUTPUT_DIR / "a3_kappa_sweep.csv", index=False)

    pairs = pairs_by_threshold[PRIMARY_THRESHOLD]
    pairs.to_csv(OUTPUT_DIR / "a3_matched_pairs.csv", index=False)

    # Full 5x2 cross-tab, before any collapsing.
    crosstab = pd.crosstab(
        pairs[f"{TOOL_A}_severity_raw"],
        pairs[f"{TOOL_B}_severity_raw"],
        rownames=[f"{TOOL_A} (5-level)"],
        colnames=[f"{TOOL_B} (2-level)"],
    )
    crosstab.to_csv(OUTPUT_DIR / "a3_severity_crosstab.csv")

    rated = pairs.assign(
        a=pairs[f"{TOOL_A}_severity"].map(collapse_severity),
        b=pairs[f"{TOOL_B}_severity"].map(collapse_severity),
    ).dropna(subset=["a", "b"])
    primary = kappa_with_ci(rated["a"], rated["b"])

    disagreements = rated[rated["a"] != rated["b"]]
    by_service = (
        rated.assign(disagree=(rated["a"] != rated["b"]).astype(int))
        .groupby(f"{TOOL_A}_service")
        .agg(pairs=("disagree", "size"), disagreements=("disagree", "sum"))
        .assign(disagreement_rate=lambda d: (d["disagreements"] / d["pairs"]).round(3))
        .sort_values(["pairs", "disagreement_rate"], ascending=False)
    )
    by_service.to_csv(OUTPUT_DIR / "a3_disagreement_by_service.csv")

    plot_kappa_sweep(sweep, OUTPUT_DIR / "a3_kappa_sweep.png")
    plot_crosstab(crosstab, OUTPUT_DIR / "a3_severity_crosstab.png")

    return {
        "availability": availability,
        "sweep": sweep,
        "pairs": pairs,
        "crosstab": crosstab,
        "primary": primary,
        "disagreements": disagreements,
        "by_service": by_service,
    }


def log_stats(results: dict) -> None:
    availability = results["availability"]
    sweep = results["sweep"]
    primary = results["primary"]
    pairs = results["pairs"]

    print()
    print("=" * 72)
    print("A3 cross-tool severity disagreement")
    print("=" * 72)
    print()
    print("Severity availability per tool:")
    print(availability.to_string(index=False))
    print()
    print(
        "Checkov is excluded from the agreement statistic: its open-source\n"
        "repo ships no severities. Agreement is Prowler vs ScoutSuite only."
    )
    print()
    print("Kappa across similarity thresholds (binary scale: high-risk vs lower):")
    print(sweep.to_string(index=False))
    print()
    print(f"Primary result (threshold {PRIMARY_THRESHOLD}):")
    print(f"  matched pairs:     {len(pairs):,}")
    print(f"  Cohen's kappa:     {primary['kappa']:.3f}  ({interpret_kappa(primary['kappa'])})")
    print(f"  95% bootstrap CI:  [{primary['ci_low']:.3f}, {primary['ci_high']:.3f}]")
    print(f"  disagreeing pairs: {len(results['disagreements']):,} of {primary['n']:,}")
    print()
    print("Raw severity cross-tab (before collapsing):")
    print(results["crosstab"].to_string())
    print()
    print("Most-compared services:")
    print(results["by_service"].head(10).to_string())
    print()
    print(f"Written to {OUTPUT_DIR}/a3_*.csv")
    print("=" * 72)


def main() -> None:
    results = run()
    log_stats(results)


if __name__ == "__main__":
    main()
