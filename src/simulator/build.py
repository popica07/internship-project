"""A5 - build the L3 simulated estate end to end.

Usage:
    python -m src.simulator.build

Writes:
    data/simulator/l3_simulated_findings.csv   one row per finding, labelled
    data/simulator/l3_assets.csv               the estate itself
    data/simulator/l3_simulator_parameters.csv every parameter used (rule 7)

The pipeline is: bank structure -> assets -> IAM graph -> inject real L1/L2
findings -> lifecycle (SLA, recurrence) -> label.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.simulator.assets import generate_assets
from src.simulator.config import DEFAULT_CONFIG, PRIORITY_ORDER, SimulatorConfig, as_records
from src.simulator.iam import asset_exposure_features, build_iam_graph
from src.simulator.injector import inject_findings
from src.simulator.labeller import label_findings
from src.simulator.temporal import apply_lifecycle

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

L1_PATH = Path("data/processed/l1_vulnerabilities.csv")
L2_PATH = Path("data/processed/l2_misconfig_catalog.csv")
OUTPUT_DIR = Path("data/simulator")

FINDINGS_PATH = OUTPUT_DIR / "l3_simulated_findings.csv"
ASSETS_PATH = OUTPUT_DIR / "l3_assets.csv"
PARAMETERS_PATH = OUTPUT_DIR / "l3_simulator_parameters.csv"

L1_COLUMNS = ["cve_id", "published_date", "cvss_score", "cvss_severity",
              "epss_score", "epss_percentile", "in_kev", "public_exploit"]


def build(config: SimulatorConfig = DEFAULT_CONFIG) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)

    logger.info("Loading real L1/L2 inputs")
    l1 = pd.read_csv(L1_PATH, usecols=L1_COLUMNS)
    l2 = pd.read_csv(L2_PATH)

    assets = generate_assets(config, rng)
    graph, principals = build_iam_graph(assets, config, rng)
    assets = assets.merge(asset_exposure_features(graph, assets, principals), on="asset_id")

    findings = inject_findings(assets, l1, l2, config, rng)
    findings = apply_lifecycle(findings, assets, config, rng)
    findings = label_findings(findings, assets, config, rng)

    # Carry the asset context onto each finding so downstream experiments have
    # one flat table, while asset_id stays available for grouped splits.
    context_columns = [
        "asset_id", "business_unit", "environment", "provider", "region", "asset_type",
        "criticality_tier", "internet_facing", "pci_scope", "dora_scope", "regulatory_scope",
        "principal_count", "admin_principal_count", "blast_radius", "has_admin_path",
    ]
    findings = findings.merge(assets[context_columns], on="asset_id", how="left")
    findings = findings.sort_values("first_seen", kind="stable").reset_index(drop=True)

    findings.to_csv(FINDINGS_PATH, index=False)
    assets.to_csv(ASSETS_PATH, index=False)
    pd.DataFrame(as_records(config)).to_csv(PARAMETERS_PATH, index=False)

    return {"findings": findings, "assets": assets, "principals": principals, "config": config}


def log_stats(results: dict) -> None:
    findings = results["findings"]
    assets = results["assets"]
    config = results["config"]

    print()
    print("=" * 72)
    print("A5 L3 estate simulator")
    print("=" * 72)
    print(f"Assets:                  {len(assets):,}")
    print(f"Findings:                {len(findings):,}")
    print(f"  from real CVEs (L1):   {(findings['source_layer'] == 'L1').sum():,}")
    print(f"  from real checks (L2): {(findings['source_layer'] == 'L2').sum():,}")
    print(f"Distinct real CVEs used: {findings.loc[findings.source_layer == 'L1', 'finding_ref'].nunique():,}")
    print(f"Distinct real checks:    {findings.loc[findings.source_layer == 'L2', 'finding_ref'].nunique():,}")
    print()
    print("Estate shape:")
    print(f"  internet-facing:       {100 * assets['internet_facing'].mean():.1f}% of assets")
    print(f"  in regulatory scope:   {100 * assets['regulatory_scope'].mean():.1f}% of assets")
    print(f"  median blast radius:   {assets['blast_radius'].median():.0f} assets")
    print(f"  on an admin path:      {100 * assets['has_admin_path'].mean():.1f}% of assets")
    print()
    print("Criticality tiers (0 = most critical):")
    for tier, count in assets["criticality_tier"].value_counts().sort_index().items():
        print(f"  tier {tier}: {count:>6,} ({count / len(assets):.1%})")
    print()
    print("Lifecycle:")
    print(f"  open:                  {100 * (findings['status'] == 'open').mean():.1f}%")
    print(f"  SLA breached:          {100 * findings['sla_breached'].mean():.1f}%")
    print(f"  mean recurrences:      {findings['recurrence_count'].mean():.2f}")
    print()
    print("Priority label distribution (60% real / 40% simulated context):")
    counts = findings["priority_label"].value_counts()
    for label in reversed(PRIORITY_ORDER):
        count = counts.get(label, 0)
        print(f"  {label:<9} {count:>7,} ({count / len(findings):.1%})")
    print()
    print(f"  priority_score: mean {findings['priority_score'].mean():.3f}, "
          f"p95 {findings['priority_score'].quantile(0.95):.3f}, "
          f"max {findings['priority_score'].max():.3f}")
    print(f"  label noise level: {config.noise_level}")
    print()
    print("REMINDER: this label is a documented formula, not ground truth. Report")
    print("L3 results separately from L1/L2 (CLAUDE.md rule 12).")
    print()
    print(f"Written to {FINDINGS_PATH}, {ASSETS_PATH}, {PARAMETERS_PATH}")
    print("=" * 72)


def main() -> None:
    results = build()
    log_stats(results)


if __name__ == "__main__":
    main()
