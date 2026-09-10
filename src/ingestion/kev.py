"""CISA Known Exploited Vulnerabilities (KEV) catalog download and parsing.

Per CLAUDE.md, KEV is the ONLY source of the "exploited in the wild" label
used anywhere in this project — never inferred or invented elsewhere.
"""

import logging

import pandas as pd
import requests

logger = logging.getLogger(__name__)

KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"


def fetch_kev() -> pd.DataFrame:
    response = requests.get(KEV_URL, timeout=60)
    response.raise_for_status()
    payload = response.json()

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
