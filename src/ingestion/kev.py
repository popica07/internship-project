"""CISA Known Exploited Vulnerabilities (KEV) catalog download and parsing.

Per CLAUDE.md, KEV is the ONLY source of the "exploited in the wild" label
used anywhere in this project — never inferred or invented elsewhere.
"""

import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"


def fetch_kev(cache_dir: Path = Path("data/raw/kev")) -> pd.DataFrame:
    """Today's KEV catalog, kept as a dated raw copy.

    KEV changes weekly and CISA publishes no history, so an uncached download
    makes the L1 label set unreproducible: a rebuild silently relabels CVEs.
    Same-day reruns reuse the copy; older copies are the label provenance.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"known_exploited_vulnerabilities_{date.today():%Y-%m-%d}.json"
    if cache_path.exists():
        payload = json.loads(cache_path.read_text())
    else:
        response = requests.get(KEV_URL, timeout=60)
        response.raise_for_status()
        payload = response.json()
        cache_path.write_text(json.dumps(payload))

    vulnerabilities = payload.get("vulnerabilities", [])
    df = pd.DataFrame(
        [
            {
                "cve_id": v["cveID"],
                "kev_date_added": v.get("dateAdded"),
                "kev_vulnerability_name": v.get("vulnerabilityName"),
                "kev_required_action": v.get("requiredAction"),
                "kev_due_date": v.get("dueDate"),
            }
            for v in vulnerabilities
        ]
    )
    logger.info("Fetched %d KEV entries (catalog version %s)", len(df), payload.get("catalogVersion"))
    return df
