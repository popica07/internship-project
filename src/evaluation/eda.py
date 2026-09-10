"""A6 - exploratory data analysis across the three layers.

Reports L1, L2 and L3 separately throughout (CLAUDE.md rule 12). The headline
output is the EPSS leakage figure, which is what forces the point-in-time rule
in the locked evaluation protocol.
"""

import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.ingestion.epss import fetch_epss_asof  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

L1_PATH = Path("data/processed/l1_vulnerabilities.csv")
L2_PATH = Path("data/processed/l2_misconfig_catalog.csv")
L3_PATH = Path("data/simulator/l3_simulated_findings.csv")
OUTPUT_DIR = Path("outputs")

SPLIT_DATE = "2023-01-01"

SERIES_BLUE = "#2a78d6"
SERIES_ORANGE = "#eb6834"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"


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


def load_l1() -> pd.DataFrame:
    df = pd.read_csv(
        L1_PATH,
        usecols=["cve_id", "published_date", "cvss_score", "cvss_severity",
                 "epss_score", "in_kev", "kev_date_added", "public_exploit", "has_patch"],
    )
    df["published_date"] = pd.to_datetime(df["published_date"], format="ISO8601", utc=True, errors="coerce")
    df["kev_date_added"] = pd.to_datetime(df["kev_date_added"], utc=True, errors="coerce")
    return df


def epss_leakage_table(l1: pd.DataFrame, split_date: str = SPLIT_DATE) -> pd.DataFrame:
    """Compare current vs point-in-time EPSS for pre-split CVEs, by KEV timing.

    This is the evidence behind the protocol's EPSS rule: if CVEs whose
    exploitation only became known after the split already score high on
    *today's* EPSS, that feature carries the outcome.
    """
    boundary = pd.Timestamp(split_date, tz="UTC")
    asof = fetch_epss_asof(split_date)[["cve_id", "epss_score"]].rename(
        columns={"epss_score": "epss_asof"}
    )

    pre = l1[l1["published_date"] < boundary].merge(asof, on="cve_id", how="left")
    groups = {
        "never in KEV": pre["in_kev"] == 0,
        f"KEV before {split_date}": (pre["in_kev"] == 1) & (pre["kev_date_added"] < boundary),
        f"KEV after {split_date}": (pre["in_kev"] == 1) & (pre["kev_date_added"] >= boundary),
    }

    rows = []
    for name, mask in groups.items():
        subset = pre[mask]
        rows.append(
            {
                "group": name,
                "n": len(subset),
                "median_epss_current": subset["epss_score"].median(),
                "median_epss_asof_split": subset["epss_asof"].median(),
            }
        )
    return pd.DataFrame(rows)


def plot_epss_leakage(table: pd.DataFrame, path: Path) -> None:
    """The A6 headline: today's EPSS knows the future; point-in-time EPSS does not."""
    positions = np.arange(len(table))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9.0, 4.8), facecolor=SURFACE)
    ax.bar(positions - width / 2, table["median_epss_current"], width,
           color=SERIES_ORANGE, label="EPSS today (leaks)")
    ax.bar(positions + width / 2, table["median_epss_asof_split"], width,
           color=SERIES_BLUE, label=f"EPSS as of {SPLIT_DATE} (correct)")

    for x, value in zip(positions - width / 2, table["median_epss_current"]):
        ax.text(x, value + 0.012, f"{value:.3f}", ha="center", fontsize=9, color=INK_SECONDARY)
    for x, value in zip(positions + width / 2, table["median_epss_asof_split"]):
        ax.text(x, value + 0.012, f"{value:.3f}", ha="center", fontsize=9, color=INK_SECONDARY)

    _style_axes(ax)
    ax.grid(axis="x", visible=False)
    ax.set_xticks(positions, [g.replace(" ", "\n", 1) for g in table["group"]], fontsize=9.5)
    ax.set_ylabel("median EPSS", color=INK_SECONDARY, fontsize=10)
    _titles(
        ax,
        "Today's EPSS already knows which CVEs would later be exploited",
        "CVEs published before 2023-01-01 — i.e. rows that would be training data",
    )
    legend = ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_l1_over_time(l1: pd.DataFrame, path: Path) -> None:
    """Publication volume and KEV additions per year - two panels, not two axes."""
    published = l1["published_date"].dt.year.value_counts().sort_index()
    published = published[(published.index >= 1999) & (published.index <= 2026)]
    kev = l1.loc[l1["in_kev"] == 1, "kev_date_added"].dt.year.value_counts().sort_index()

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(8.5, 5.8), facecolor=SURFACE, gridspec_kw={"hspace": 0.42}
    )

    top.bar(published.index, published.values, color=SERIES_BLUE, width=0.75)
    _style_axes(top)
    top.grid(axis="x", visible=False)
    top.set_ylabel("CVEs published", color=INK_SECONDARY, fontsize=10)
    # Compute the caption from the data - a hardcoded count goes stale as soon
    # as CISA adds an entry (it did, mid-project: 1,695 -> 1,697).
    n_kev = int(l1["in_kev"].sum())
    _titles(top, "CVE volume keeps growing; KEV additions do not",
            f"Why the positive class stays rare: {n_kev:,} KEV entries against "
            f"{len(l1):,} CVEs (1:{len(l1) / n_kev - 1:.0f})")

    bottom.bar(kev.index, kev.values, color=SERIES_ORANGE, width=0.75)
    _style_axes(bottom)
    bottom.grid(axis="x", visible=False)
    bottom.set_ylabel("added to KEV", color=INK_SECONDARY, fontsize=10)
    bottom.set_xlabel("year", color=INK_SECONDARY, fontsize=10)

    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def run() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    l1 = load_l1()
    l2 = pd.read_csv(L2_PATH)
    l3 = pd.read_csv(L3_PATH)

    leakage = epss_leakage_table(l1)
    leakage.to_csv(OUTPUT_DIR / "a6_epss_leakage.csv", index=False)
    plot_epss_leakage(leakage, OUTPUT_DIR / "a6_epss_leakage.png")
    plot_l1_over_time(l1, OUTPUT_DIR / "a6_l1_over_time.png")

    summary = pd.DataFrame(
        [
            {"layer": "L1 (real)", "rows": len(l1), "unit": "CVEs",
             "positive_label": "CISA KEV", "positive_rate": round(l1["in_kev"].mean(), 5)},
            {"layer": "L2 (real)", "rows": len(l2), "unit": "checks",
             "positive_label": "n/a (no labels)", "positive_rate": np.nan},
            {"layer": "L3 (simulated)", "rows": len(l3), "unit": "findings",
             "positive_label": "formula (60/40)",
             "positive_rate": round((l3["priority_label"].isin(["High", "Critical"])).mean(), 5)},
        ]
    )
    summary.to_csv(OUTPUT_DIR / "a6_layer_summary.csv", index=False)

    return {"l1": l1, "l2": l2, "l3": l3, "leakage": leakage, "summary": summary}


def log_stats(results: dict) -> None:
    l1, l2, l3 = results["l1"], results["l2"], results["l3"]
    leakage = results["leakage"]

    print()
    print("=" * 72)
    print("A6 exploratory data analysis")
    print("=" * 72)
    print("Layers are reported separately (rule 12) - they do not share a label source.")
    print()
    print(results["summary"].to_string(index=False))
    print()
    print("L1 - class imbalance:")
    print(f"  KEV positives:    {l1['in_kev'].sum():,} of {len(l1):,}  (1:{(len(l1) / l1['in_kev'].sum() - 1):.0f})")
    print(f"  public_exploit:   {100 * l1['public_exploit'].mean():.1f}% of CVEs")
    print(f"  CVSS available:   {100 * l1['cvss_score'].notna().mean():.1f}% of CVEs")
    print()
    print("EPSS LEAKAGE (drives the protocol's point-in-time rule):")
    print(leakage.to_string(index=False))
    never = leakage.loc[leakage["group"] == "never in KEV"].iloc[0]
    after = leakage.iloc[2]
    print()
    print(f"  ratio with EPSS today:        {after['median_epss_current'] / never['median_epss_current']:.1f}x")
    print(f"  ratio with EPSS at split:     {after['median_epss_asof_split'] / never['median_epss_asof_split']:.1f}x")
    print("  -> today's EPSS encodes post-split outcomes; use fetch_epss_asof().")
    print()
    print("L2 - severity availability by tool:")
    for tool, group in l2.groupby("tool"):
        usable = 100 * (group["severity_normalised"] != "unknown").mean()
        print(f"  {tool:<12} {len(group):>5,} checks, {usable:5.1f}% with severity")
    print()
    print("L3 - label distribution (a formula, not ground truth):")
    for label, count in l3["priority_label"].value_counts().items():
        print(f"  {label:<9} {count:>7,} ({count / len(l3):.1%})")
    print()
    print(f"Written to {OUTPUT_DIR}/a6_*")
    print("=" * 72)


def main() -> None:
    log_stats(run())


if __name__ == "__main__":
    main()
