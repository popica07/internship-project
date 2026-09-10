"""NVD API 2.0 download and parsing.

The legacy bulk JSON feeds (nvd.nist.gov/feeds/json/cve/1.1/...) referenced
in older docs were retired by NIST in December 2023 and now return HTTP 403.
This module uses the live NVD API 2.0 instead, paginated.
"""

import json
import logging
import os
import signal
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

logger = logging.getLogger(__name__)

NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
RESULTS_PER_PAGE = 2000
RAW_CACHE_DIR = Path("data/raw/nvd")
MAX_RETRIES = 6
RETRYABLE_STATUS_CODES = {403, 429, 500, 502, 503, 504}
HARD_DEADLINE_SECONDS = 120


class _HardTimeout(Exception):
    """Raised when a single request exceeds a wall-clock deadline.

    requests' `timeout=` is a per-chunk inactivity timeout: a server that
    trickles a few bytes every ~80s never trips a 90s read timeout while
    still being effectively stalled. This alarm-based watchdog bounds the
    *total* time for one attempt regardless of how the server drips data.
    """


def _alarm_handler(signum, frame):
    raise _HardTimeout(f"request exceeded {HARD_DEADLINE_SECONDS}s hard deadline")


def _sleep_interval(api_key: str | None) -> float:
    # NVD's public rate limit is ~5 requests/30s; with a key, ~50/30s.
    return 0.6 if api_key else 6.0


def _fetch_page(start_index: int, api_key: str | None) -> dict:
    # A fresh connection per request (no persistent Session) avoids a stale
    # keep-alive connection from silently hanging past the configured
    # timeout after many requests over a long-running pull.
    headers = {"apiKey": api_key} if api_key else {}
    params = {"resultsPerPage": RESULTS_PER_PAGE, "startIndex": start_index}

    for attempt in range(1, MAX_RETRIES + 1):
        previous_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(HARD_DEADLINE_SECONDS)
        try:
            response = requests.get(NVD_API_URL, headers=headers, params=params, timeout=(15, 90))
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)
        except (requests.exceptions.RequestException, _HardTimeout) as exc:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)
            if attempt < MAX_RETRIES:
                backoff = min(2**attempt * _sleep_interval(api_key), 60)
                logger.warning(
                    "NVD request errored (%s) at startIndex=%d, attempt %d/%d, "
                    "backing off %.1fs",
                    exc.__class__.__name__,
                    start_index,
                    attempt,
                    MAX_RETRIES,
                    backoff,
                )
                time.sleep(backoff)
                continue
            raise RuntimeError(
                f"NVD API request failed after {MAX_RETRIES} attempts (startIndex={start_index})"
            ) from exc

        if response.status_code == 200:
            return response.json()

        if response.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES:
            backoff = min(2**attempt * _sleep_interval(api_key), 60)
            logger.warning(
                "NVD request failed (status %d) at startIndex=%d, attempt %d/%d, "
                "backing off %.1fs",
                response.status_code,
                start_index,
                attempt,
                MAX_RETRIES,
                backoff,
            )
            time.sleep(backoff)
            continue

        response.raise_for_status()

    raise RuntimeError(f"NVD API request failed after {MAX_RETRIES} attempts (startIndex={start_index})")


def _best_cvss(metrics: dict) -> tuple[float | None, str | None, str | None]:
    for version_key, version_label in (
        ("cvssMetricV31", "3.1"),
        ("cvssMetricV30", "3.0"),
        ("cvssMetricV2", "2.0"),
    ):
        entries = metrics.get(version_key)
        if entries:
            entry = entries[0]
            cvss_data = entry.get("cvssData", {})
            score = cvss_data.get("baseScore")
            severity = cvss_data.get("baseSeverity") or entry.get("baseSeverity")
            return score, severity, version_label
    return None, None, None


def _parse_cve_item(item: dict) -> dict:
    cve = item["cve"]
    descriptions = cve.get("descriptions", [])
    description = next((d["value"] for d in descriptions if d.get("lang") == "en"), None)
    if description is None and descriptions:
        description = descriptions[0].get("value")

    score, severity, version = _best_cvss(cve.get("metrics", {}))

    # NVD tags references it considers exploit material. This is the only
    # public-exploit signal available from NVD itself, and the L3 label formula
    # needs it - so it is taken from real data rather than invented.
    reference_tags = {tag for ref in cve.get("references", []) for tag in (ref.get("tags") or [])}

    return {
        "cve_id": cve["id"],
        "published_date": cve.get("published"),
        "last_modified_date": cve.get("lastModified"),
        "description": description,
        "cvss_score": score,
        "cvss_severity": severity,
        "cvss_version": version,
        "public_exploit": int("Exploit" in reference_tags),
        "has_patch": int("Patch" in reference_tags),
    }


def fetch_all_cves(api_key: str | None = None) -> list[dict]:
    """Paginate NVD API 2.0 for the full CVE catalog, caching raw pages to disk.

    A rerun after an interruption resumes from cached pages instead of
    re-fetching everything.
    """
    api_key = api_key or os.environ.get("NVD_API_KEY") or None
    RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    start_index = 0
    total_results = None
    pbar = None

    try:
        while total_results is None or start_index < total_results:
            cache_path = RAW_CACHE_DIR / f"page_{start_index}.json"

            if cache_path.exists():
                payload = json.loads(cache_path.read_text())
            else:
                payload = _fetch_page(start_index, api_key)
                cache_path.write_text(json.dumps(payload))
                time.sleep(_sleep_interval(api_key))

            if total_results is None:
                total_results = payload["totalResults"]
                pbar = tqdm(total=total_results, desc="NVD CVEs", unit="cve")

            vulnerabilities = payload.get("vulnerabilities", [])
            for item in vulnerabilities:
                records.append(_parse_cve_item(item))

            pbar.update(len(vulnerabilities))
            start_index += RESULTS_PER_PAGE
    finally:
        if pbar is not None:
            pbar.close()

    logger.info("Fetched %d CVE records from NVD API 2.0", len(records))
    return records
