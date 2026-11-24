"""B1 - the baseline battery. Every Study B/C model has to beat these.

No baseline is trained, so none needs training labels: each is a fixed
ordering of the test CVEs, scored against KEV membership with the locked
`evaluate()`.

| baseline        | ranks by                                   | missing value     |
|-----------------|--------------------------------------------|-------------------|
| cvss_v3         | CVSS v3.x base score                       | ranked last       |
| cvss_v4         | CVSS v4.0 base score                       | ranked last       |
| severity_label  | CRITICAL > HIGH > MEDIUM > LOW > NONE      | ranked last       |
| recency         | newest publication first                   | -                 |
| epss_current    | today's EPSS                               | ranked last       |
| random          | nothing                                    | -                 |

TIES ARE THE CENTRAL METHODOLOGICAL ISSUE HERE. The test set has only ~84
distinct CVSS scores (one shared by 16k CVEs) and 5 severity labels, so the
top-k of a CVSS ordering is mostly decided by whatever order the rows happen to
be in. Every baseline therefore breaks ties uniformly at random, once per seed,
and is reported as mean +/- std over 10 seeds (CLAUDE.md rule 5). For
continuous scores the std is ~0; for CVSS and severity it is the honest width
of "sort by CVSS" as a policy.

`epss_current` is NOT a fair competitor: today's EPSS has seen the
exploitation activity KEV records (A6: 49.5x leakage ratio). It is reported as
a leaky reference ceiling, which the evaluation protocol permits for B1 only.
"""

import logging
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.labels import temporal_labels
from src.evaluation.metrics import brier_score_loss, evaluate, expected_calibration_error
from src.evaluation.splits import assert_no_temporal_leakage, temporal_split

logger = logging.getLogger(__name__)

L1_PATH = Path("data/processed/l1_vulnerabilities.csv")
V4_PATH = Path("data/processed/l1_cvss_v4.csv")
OUTPUT_DIR = Path("outputs")
SPLIT_DATE = "2023-01-01"
REPORT_METRICS = ["pr_auc", "precision_at_10", "precision_at_50", "precision_at_100",
                  "recall_at_100", "ndcg_at_100", "mrr", "roc_auc"]

SEVERITY_ORDER = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "NONE": 0}
SEEDS = list(range(10))
MISSING = -np.inf


def baseline_scores(test: pd.DataFrame) -> dict[str, np.ndarray]:
    """Primary ordering key per baseline (higher = more urgent), before tie-breaks."""
    v3 = test["cvss_score"].where(test["cvss_version"].astype(str).str.startswith("3"))
    published = pd.to_datetime(test["published_date"], format="ISO8601", utc=True)
    if published.isna().any():
        raise ValueError("unparsed published_date in test set")
    return {
        "cvss_v3": v3.fillna(MISSING).to_numpy(float),
        "cvss_v4": test["cvss4_score"].fillna(MISSING).to_numpy(float),
        "severity_label": test["cvss_severity"].map(SEVERITY_ORDER).fillna(MISSING).to_numpy(float),
        "recency": published.astype("int64").to_numpy(float),
        "epss_current": test["epss_score"].fillna(MISSING).to_numpy(float),
        "random": np.zeros(len(test)),
    }


def tie_broken_rank(primary: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A strict ordering: by `primary`, ties broken uniformly at random.

    Returned as a score in (0, 1] (1 = top), so evaluate() sees no ties at all
    and every metric reflects exactly one concrete queue an analyst could work.
    """
    jitter = rng.random(len(primary))
    order = np.lexsort((jitter, primary))  # ascending: last = most urgent
    score = np.empty(len(primary))
    score[order] = np.arange(1, len(primary) + 1) / len(primary)
    return score


def run_battery(test: pd.DataFrame, y_test: pd.Series, seeds: list[int] = SEEDS) -> pd.DataFrame:
    """One row per (baseline, seed) with the full locked metric set."""
    y = np.asarray(y_test)
    rows = []
    for name, primary in baseline_scores(test).items():
        for seed in seeds:
            # Seed offset per baseline so baselines do not share tie-break draws.
            # crc32, not hash(): str hashes are salted per process (not reproducible).
            rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
            metrics = evaluate(y, tie_broken_rank(primary, rng), is_probability=False)
            rows.append({"baseline": name, "seed": seed, **metrics})
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """mean +/- std per baseline, sorted by the primary metric (PR-AUC)."""
    grouped = results.groupby("baseline")[metrics]
    summary = grouped.mean().round(4).astype(str) + " ± " + grouped.std(ddof=1).round(4).astype(str)
    order = grouped.mean()["pr_auc"].sort_values(ascending=False).index
    return summary.loc[order]


def load_test() -> tuple[pd.DataFrame, pd.Series]:
    l1 = pd.read_csv(L1_PATH, low_memory=False)
    v4 = pd.read_csv(V4_PATH)
    if set(v4["cve_id"]) != set(l1["cve_id"]):
        raise ValueError("l1_cvss_v4.csv does not match L1 - rerun python -m src.ingestion.cvss_v4")
    l1 = l1.merge(v4, on="cve_id", how="left")
    train, test = temporal_split(l1, SPLIT_DATE)
    assert_no_temporal_leakage(train, test)
    _, y_test = temporal_labels(train, test, SPLIT_DATE)
    return test.reset_index(drop=True), y_test.reset_index(drop=True)


def plot_battery(results: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = [("pr_auc", "PR-AUC"), ("precision_at_100", "Precision@100")]
    order = results.groupby("baseline")["pr_auc"].mean().sort_values().index
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (metric, label) in zip(axes, metrics):
        stats = results.groupby("baseline")[metric].agg(["mean", "std"]).loc[order]
        colors = ["#b8b8b8" if b == "epss_current" else "#2a6f97" for b in stats.index]
        ax.barh(stats.index, stats["mean"], xerr=stats["std"], color=colors, capsize=3)
        ax.set_title(label, loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle(
        f"B1 baselines - KEV in {int(results['n_positive'].iloc[0])} of "
        f"{int(results['n'].iloc[0]):,} test CVEs (mean ± std, 10 tie-break seeds; "
        "grey = leaky reference)",
        x=0.01, ha="left", fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run() -> dict:
    from src.tracking import log_experiment

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    test, y_test = load_test()

    results = run_battery(test, y_test)
    results.to_csv(OUTPUT_DIR / "b1_baselines_by_seed.csv", index=False)
    summary = summarise(results, REPORT_METRICS)
    summary.to_csv(OUTPUT_DIR / "b1_baselines_summary.csv")

    # v3 vs v4 on equal footing: only CVEs that carry a v4 score.
    has_v4 = test["cvss4_score"].notna().to_numpy()
    subset = run_battery(test[has_v4].reset_index(drop=True), y_test[has_v4].reset_index(drop=True))
    subset = subset[subset["baseline"].isin(["cvss_v3", "cvss_v4", "random"])]
    subset_summary = summarise(subset, REPORT_METRICS)
    subset_summary.to_csv(OUTPUT_DIR / "b1_v3_vs_v4_on_v4_subset.csv")

    # EPSS is the only baseline that is a probability, so the only one whose
    # calibration means anything (protocol: every probabilistic score is checked).
    epss = test["epss_score"].notna().to_numpy()
    calibration = {
        "epss_current_brier": brier_score_loss(y_test[epss], test.loc[epss, "epss_score"]),
        "epss_current_ece": expected_calibration_error(y_test[epss], test.loc[epss, "epss_score"]),
        "epss_current_coverage": float(epss.mean()),
    }

    plot_battery(results, OUTPUT_DIR / "b1_baselines.png")

    means = results.groupby("baseline")[REPORT_METRICS].mean()
    stds = results.groupby("baseline")[REPORT_METRICS].std(ddof=1)
    for baseline in means.index:
        log_experiment(
            "B1",
            run_name=f"B1_{baseline}",
            params={"baseline": baseline, "split_date": SPLIT_DATE, "seeds": SEEDS,
                    "label_rule": "test = KEV as of snapshot", "tie_break": "uniform random per seed"},
            metrics={
                **{f"{m}_mean": means.loc[baseline, m] for m in REPORT_METRICS},
                **{f"{m}_std": stds.loc[baseline, m] for m in REPORT_METRICS},
                **(calibration if baseline == "epss_current" else {}),
            },
            artifacts=sorted(OUTPUT_DIR.glob("b1_*")) if baseline == "random" else None,
            tags={"leaky_reference": str(baseline == "epss_current")},
        )

    return {"test": test, "y_test": y_test, "results": results, "summary": summary,
            "subset_summary": subset_summary, "calibration": calibration,
            "v4_coverage": float(has_v4.mean())}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out = run()
    print(out["summary"].to_string())
    print(out["subset_summary"].to_string())
    print(out["calibration"])
