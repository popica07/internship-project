"""Join NVD + EPSS + CISA KEV on CVE ID into data/processed/l1_vulnerabilities.csv.

Usage:
    python -m src.ingestion.join
"""

import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.ingestion.epss import fetch_epss_scores
from src.ingestion.kev import fetch_kev
from src.ingestion.nvd import fetch_all_cves

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_PATH = Path("data/processed/l1_vulnerabilities.csv")


def build_l1_dataset() -> pd.DataFrame:
    stages = tqdm(total=5, desc="A1 pipeline", unit="stage")

    stages.set_description("Downloading NVD (API 2.0)")
    nvd_df = pd.DataFrame(fetch_all_cves())
    stages.update(1)

    stages.set_description("Downloading EPSS")
    epss_df = fetch_epss_scores()
    stages.update(1)

    stages.set_description("Downloading CISA KEV")
    kev_df = fetch_kev()
    stages.update(1)

    stages.set_description("Joining on CVE ID")
    df = nvd_df.merge(epss_df, on="cve_id", how="left")
    df = df.merge(kev_df, on="cve_id", how="left")
    df["in_kev"] = df["kev_date_added"].notna().astype(int)
    df["published_date"] = pd.to_datetime(df["published_date"], format="ISO8601", utc=True, errors="coerce")
    df = df.sort_values("published_date", kind="stable").reset_index(drop=True)

    stages.update(1)

    stages.set_description("Writing CSV")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Write one uniform timestamp format. Pandas drops trailing zeros in
    # fractional seconds, which leaves a CSV with mixed representations; on
    # re-read pandas 2.x infers a single format and silently coerces every
    # other variant to NaT (this quietly destroyed 91% of the dates once).
    # Second precision is ample for a publication timestamp. Datetimes are
    # kept in the returned frame; only the file gets the string form.
    output = df.copy()
    for column in ("published_date", "last_modified_date"):
        parsed = pd.to_datetime(output[column], format="ISO8601", utc=True, errors="coerce")
        output[column] = parsed.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    output.to_csv(OUTPUT_PATH, index=False)
    stages.update(1)
    stages.close()

    return df


def log_stats(df: pd.DataFrame) -> None:
    total = len(df)
    kev_count = int(df["in_kev"].sum())
    non_kev_count = total - kev_count
    imbalance = f"1 : {non_kev_count / kev_count:.1f}" if kev_count else "N/A (no KEV matches)"
    date_min = df["published_date"].min()
    date_max = df["published_date"].max()
    epss_coverage = df["epss_score"].notna().mean() * 100

    tqdm.write("")
    tqdm.write("=" * 60)
    tqdm.write("A1 ingestion complete: data/processed/l1_vulnerabilities.csv")
    tqdm.write("=" * 60)
    tqdm.write(f"Total CVEs:              {total:,}")
    tqdm.write(f"KEV count:               {kev_count:,} ({kev_count / total:.2%})")
    tqdm.write(f"Class imbalance ratio:   {imbalance} (non-KEV : KEV)")
    tqdm.write(f"Date range:              {date_min.date()} to {date_max.date()}")
    tqdm.write(f"EPSS coverage:           {epss_coverage:.1f}% of CVEs have a score")
    tqdm.write(
        "Note: EPSS scores are a current snapshot, not historical-at-publication "
        "(see src/ingestion/epss.py docstring)."
    )
    tqdm.write("=" * 60)


def main() -> None:
    df = build_l1_dataset()
    log_stats(df)


if __name__ == "__main__":
    main()
