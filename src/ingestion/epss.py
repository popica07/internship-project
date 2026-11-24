"""EPSS daily CSV download and parsing.

Note: this is a *current snapshot* of EPSS scores (today's score for every
CVE), not a historical score at each CVE's publication date. EPSS did not
exist before 2021 and its per-CVE score changes over time as the model is
retrained, so joining it onto decades of NVD history means older CVEs get
today's score rather than the score at (or near) their disclosure date.
This is a real limitation of the "current" endpoint CLAUDE.md specifies,
not a bug in this module.
"""

import gzip
import io
import logging
from datetime import date
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

EPSS_URL = "https://epss.cyentia.com/epss_scores-current.csv.gz"
# Daily point-in-time archives. Verified available back to at least 2021.
EPSS_ASOF_URL = "https://epss.empiricalsecurity.com/epss_scores-{date}.csv.gz"


def fetch_epss_scores(cache_dir: Path = Path("data/raw/epss")) -> pd.DataFrame:
    """Today's EPSS snapshot, kept as a dated raw copy (same reason as KEV)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"epss_scores-current-downloaded-{date.today():%Y-%m-%d}.csv.gz"
    if cache_path.exists():
        content = cache_path.read_bytes()
    else:
        response = requests.get(EPSS_URL, timeout=60)
        response.raise_for_status()
        content = response.content
        cache_path.write_bytes(content)

    with gzip.GzipFile(fileobj=io.BytesIO(content)) as gz:
        # First line is a metadata comment (#model_version:...,score_date:...)
        # before the real "cve,epss,percentile" header.
        df = pd.read_csv(gz, comment="#")

    df = df.rename(columns={"cve": "cve_id", "epss": "epss_score", "percentile": "epss_percentile"})
    logger.info("Fetched EPSS scores for %d CVEs", len(df))
    return df[["cve_id", "epss_score", "epss_percentile"]]


def fetch_epss_asof(date: str, cache_dir: Path = Path("data/raw/epss")) -> pd.DataFrame:
    """EPSS scores as they stood on `date` (YYYY-MM-DD), cached locally.

    Required for any temporally-split experiment. A6 showed that the *current*
    snapshot leaks: among CVEs published before 2023-01-01, those added to KEV
    only AFTER that date already carry a median present-day EPSS of 0.64 -
    45.7x the 0.014 of CVEs never added to KEV, and indistinguishable from
    those already in KEV before the cutoff. Today's EPSS is close to an oracle
    for future exploitation, so training on it under a temporal split leaks the
    outcome. Use this function with the split date instead.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"epss_scores-{date}.csv.gz"

    if not cache_path.exists():
        response = requests.get(EPSS_ASOF_URL.format(date=date), timeout=120)
        response.raise_for_status()
        cache_path.write_bytes(response.content)

    with gzip.GzipFile(cache_path) as gz:
        df = pd.read_csv(gz, comment="#")

    df = df.rename(columns={"cve": "cve_id", "epss": "epss_score", "percentile": "epss_percentile"})
    if "epss_percentile" not in df.columns:
        # EPSS v1 files (2021) shipped scores only, no percentile column.
        df["epss_percentile"] = float("nan")
    df["epss_asof_date"] = date
    logger.info("Fetched EPSS as of %s for %d CVEs", date, len(df))
    return df[["cve_id", "epss_score", "epss_percentile", "epss_asof_date"]]
