"""CVSS v4.0 scores, re-parsed from the cached NVD pages (no network).

A1's parser kept only the best of v3.1 / v3.0 / v2, so CVSS v4 - which the B1
baseline battery names explicitly - never reached l1_vulnerabilities.csv. This
re-reads the same raw pages A1 used, in the same order (base pages, then the
modified-since overlay, later versions winning), so the v4 columns describe
exactly the record versions in L1. L1 itself is not rebuilt: a rebuild on a new
day would also pull a new KEV snapshot and silently change the labels.

Usage:
    python -m src.ingestion.cvss_v4
"""

import json
import logging
from pathlib import Path

import pandas as pd

from src.ingestion.nvd import RAW_CACHE_DIR, UPDATES_CACHE_DIR

logger = logging.getLogger(__name__)

OUTPUT_PATH = Path("data/processed/l1_cvss_v4.csv")


def _page_index(path: Path) -> int:
    return int(path.stem.split("_")[1])


def _v4_fields(item: dict) -> dict:
    cve = item["cve"]
    entries = cve.get("metrics", {}).get("cvssMetricV40") or []
    # Same rule A1 applies to v3: the first listed entry.
    entry = entries[0] if entries else {}
    data = entry.get("cvssData", {})
    return {
        "cve_id": cve["id"],
        "cvss4_score": data.get("baseScore"),
        "cvss4_severity": data.get("baseSeverity"),
        "cvss4_source_type": entry.get("type"),  # Primary (NVD) or Secondary (CNA)
    }


def build_cvss_v4() -> pd.DataFrame:
    records: dict[str, dict] = {}
    base_pages = sorted(RAW_CACHE_DIR.glob("page_*.json"), key=_page_index)
    update_pages = [
        page
        for window in sorted(p for p in UPDATES_CACHE_DIR.glob("*") if p.is_dir())
        for page in sorted(window.glob("page_*.json"), key=_page_index)
    ]
    for page in base_pages + update_pages:
        for item in json.loads(page.read_text()).get("vulnerabilities", []):
            fields = _v4_fields(item)
            records[fields["cve_id"]] = fields

    df = pd.DataFrame(records.values())
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    logger.info(
        "CVSS v4 for %d of %d CVEs (%d base pages, %d update pages)",
        df["cvss4_score"].notna().sum(), len(df), len(base_pages), len(update_pages),
    )
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    build_cvss_v4()
