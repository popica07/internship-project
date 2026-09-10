"""Normalise Prowler + Checkov + ScoutSuite checks into one L2 catalog.

Usage:
    python -m src.misconfig.normaliser

Writes data/processed/l2_misconfig_catalog.csv.

SEVERITY NORMALISATION (documented, because A3 depends on it)
------------------------------------------------------------
The three tools do not share a severity scale, so the raw value is always
kept in `severity_raw` and the mapped value in `severity_normalised`:

  Prowler     critical / high / medium / low / informational  -> identity
  ScoutSuite  danger  -> high
              warning -> medium
  Checkov     (none shipped in OSS)                           -> unknown

ScoutSuite's two-level scale cannot express "critical" or "low", so mapping
it onto the five-level scale is lossy by construction. A3's cross-tool
agreement analysis must therefore either collapse both scales to a shared
coarse scale or report the mapping as a limitation - it cannot treat
`severity_normalised` as if all three tools measured the same thing.

PRIMARY KEY
-----------
`check_id` is NOT unique, legitimately: Checkov ships the same policy id for
several IaC frameworks (CKV2_AWS_33 for both Terraform and CloudFormation)
and Prowler reuses check ids across providers (`vpc_flow_logs_enabled` on
both AWS and Alibaba Cloud). There is also one genuine upstream collision in
Checkov, where CKV_NCP_22 is assigned to two unrelated checks. Use
`catalog_id` (unique by construction) as the row key downstream.
"""

import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.misconfig import checkov_extractor, prowler_extractor, scoutsuite_extractor
from src.misconfig.repos import ensure_repos, repo_commit

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_PATH = Path("data/processed/l2_misconfig_catalog.csv")

SEVERITY_MAP = {
    # Prowler (five-level, used as the canonical scale)
    "critical": "critical",
    "high": "high",
    "medium": "medium",
    "low": "low",
    "informational": "informational",
    # ScoutSuite (two-level)
    "danger": "high",
    "warning": "medium",
}

COLUMNS = [
    "catalog_id",
    "check_id",
    "tool",
    "provider",
    "service",
    "title",
    "description",
    "rationale_risk",
    "severity_raw",
    "severity_normalised",
    "categories",
    "resource_type",
    "framework",
    "check_type",
    "remediation_text",
    "remediation_code",
    "remediation_url",
    "references",
    "source_file",
    "source_commit",
]


def normalise_severity(raw: str | None) -> str:
    if not raw or not isinstance(raw, str):
        return "unknown"
    return SEVERITY_MAP.get(raw.strip().lower(), "unknown")


def build_l2_catalog() -> pd.DataFrame:
    stages = tqdm(total=5, desc="A2 pipeline", unit="stage")

    stages.set_description("Cloning/locating repos")
    repos = ensure_repos()
    commits = {name: repo_commit(path) for name, path in repos.items()}
    logger.info("Repo commits: %s", commits)
    stages.update(1)

    stages.set_description("Extracting Prowler")
    records = prowler_extractor.extract(repos["prowler"])
    for r in records:
        r["source_commit"] = commits["prowler"]
    stages.update(1)

    stages.set_description("Extracting Checkov")
    checkov_records = checkov_extractor.extract(repos["checkov"])
    for r in checkov_records:
        r["source_commit"] = commits["checkov"]
    records += checkov_records
    stages.update(1)

    stages.set_description("Extracting ScoutSuite")
    scoutsuite_records = scoutsuite_extractor.extract(repos["scoutsuite"])
    for r in scoutsuite_records:
        r["source_commit"] = commits["scoutsuite"]
    records += scoutsuite_records
    stages.update(1)

    stages.set_description("Normalising and writing CSV")
    df = pd.DataFrame(records)
    df["severity_normalised"] = df["severity_raw"].map(normalise_severity)

    # List-valued fields are pipe-joined so the CSV stays flat and diffable.
    for column in ("categories", "check_type", "references"):
        df[column] = df[column].apply(
            lambda values: "|".join(str(v) for v in values if v) if isinstance(values, list) else ""
        )

    df = df.sort_values(["tool", "provider", "check_id", "source_file"], kind="stable").reset_index(drop=True)

    # Readable, unique row key: "tool:check_id", suffixed only where a check id
    # legitimately repeats (see PRIMARY KEY in the module docstring).
    base = df["tool"] + ":" + df["check_id"].astype(str)
    occurrence = base.groupby(base).cumcount()
    df["catalog_id"] = base.where(occurrence == 0, base + "#" + (occurrence + 1).astype(str))

    df = df[COLUMNS]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    stages.update(1)
    stages.close()

    return df


def log_stats(df: pd.DataFrame) -> None:
    tqdm.write("")
    tqdm.write("=" * 68)
    tqdm.write("A2 extraction complete: data/processed/l2_misconfig_catalog.csv")
    tqdm.write("=" * 68)
    tqdm.write(f"Total checks:            {len(df):,}")
    tqdm.write("")
    tqdm.write("Per tool:")
    for tool, count in df["tool"].value_counts().items():
        with_sev = (df.loc[df["tool"] == tool, "severity_normalised"] != "unknown").mean() * 100
        tqdm.write(f"  {tool:<12} {count:>6,}   ({with_sev:.0f}% with a usable severity)")
    tqdm.write("")
    tqdm.write("Severity distribution (normalised):")
    for severity, count in df["severity_normalised"].value_counts().items():
        tqdm.write(f"  {severity:<15} {count:>6,} ({count / len(df):.1%})")
    tqdm.write("")
    tqdm.write("Top 10 providers:")
    for provider, count in df["provider"].value_counts().head(10).items():
        tqdm.write(f"  {provider or '(none)':<15} {count:>6,}")
    tqdm.write("")
    tqdm.write(f"Distinct services:       {df['service'].nunique():,}")
    tqdm.write(f"With remediation text:   {df['remediation_text'].notna().mean():.1%}")
    tqdm.write("")
    tqdm.write(
        "Note: Checkov OSS ships no severities (they live in the commercial\n"
        "platform), and ScoutSuite has only danger/warning. See the severity\n"
        "mapping in this module's docstring before running A3."
    )
    tqdm.write("=" * 68)


def main() -> None:
    df = build_l2_catalog()
    log_stats(df)


if __name__ == "__main__":
    main()
