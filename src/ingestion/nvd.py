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
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

logger = logging.getLogger(__name__)

NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
RESULTS_PER_PAGE = 2000
RAW_CACHE_DIR = Path("data/raw/nvd")
UPDATES_CACHE_DIR = Path("data/raw/nvd_updates")
# NVD rejects lastModStartDate/lastModEndDate ranges longer than 120 days.
MAX_LASTMOD_WINDOW = timedelta(days=120)
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
    return _fetch_page_params(start_index, RESULTS_PER_PAGE, api_key)


def _fetch_page_params(
    start_index: int,
    results_per_page: int,
    api_key: str | None,
    extra_params: dict | None = None,
) -> dict:
    # A fresh connection per request (no persistent Session) avoids a stale
    # keep-alive connection from silently hanging past the configured
    # timeout after many requests over a long-running pull.
    headers = {"apiKey": api_key} if api_key else {}
    params = {"resultsPerPage": results_per_page, "startIndex": start_index, **(extra_params or {})}

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


def fetch_total_results(api_key: str | None = None) -> int:
    """Live catalog size, deliberately NOT read from the page cache.

    Taking `totalResults` from cached page 0 pins the dataset to whatever size
    it had when that page was first written, so the pull can never grow and a
    "refresh" silently returns the old catalog. (This happened: a refresh in
    November still returned the September total of 387,494 against a live
    430,678.) One cheap request keeps the cache useful for resuming while
    letting the catalog grow.
    """
    payload = _fetch_page_params(0, 1, api_key)
    return int(payload["totalResults"])


def fetch_all_cves(api_key: str | None = None) -> list[dict]:
    """Paginate NVD API 2.0 for the full CVE catalog, caching raw pages to disk.

    A rerun after an interruption resumes from cached pages instead of
    re-fetching everything, and picks up CVEs added since the cache was built.
    """
    api_key = api_key or os.environ.get("NVD_API_KEY") or None
    RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    cached_mtimes: list[float] = []
    start_index = 0
    total_results = fetch_total_results(api_key)
    logger.info("NVD catalog currently holds %d CVEs", total_results)
    pbar = tqdm(total=total_results, desc="NVD CVEs", unit="cve")

    try:
        while start_index < total_results:
            cache_path = RAW_CACHE_DIR / f"page_{start_index}.json"

            payload = None
            if cache_path.exists():
                payload = json.loads(cache_path.read_text())
                # A page cached when it was the catalog's LAST page is short.
                # Once the catalog has grown past it, reusing it would silently
                # drop every CVE that has since filled the rest of that page
                # (this would have lost 506 CVEs at page_386000 in November).
                is_last_page = start_index + RESULTS_PER_PAGE >= total_results
                if len(payload.get("vulnerabilities", [])) < RESULTS_PER_PAGE and not is_last_page:
                    logger.info("Cached page %d is short; re-fetching", start_index)
                    payload = None
                else:
                    cached_mtimes.append(cache_path.stat().st_mtime)
            if payload is None:
                payload = _fetch_page(start_index, api_key)
                cache_path.write_text(json.dumps(payload))
                time.sleep(_sleep_interval(api_key))

            vulnerabilities = payload.get("vulnerabilities", [])
            for item in vulnerabilities:
                records.append(_parse_cve_item(item))

            pbar.update(len(vulnerabilities))
            start_index += RESULTS_PER_PAGE
    finally:
        if pbar is not None:
            pbar.close()

    # Cached pages hold CVE content as it was when each page was written. NVD
    # keeps editing old records (late CVSS scoring, rejections, new reference
    # tags), so overlay everything modified since the oldest cached page.
    if cached_mtimes:
        since = datetime.fromtimestamp(min(cached_mtimes), tz=timezone.utc) - timedelta(days=1)
        updates = fetch_modified_since(since, api_key=api_key)
        if updates:
            by_id = {record["cve_id"]: i for i, record in enumerate(records)}
            replaced = added = 0
            for record in updates:
                index = by_id.get(record["cve_id"])
                if index is None:
                    by_id[record["cve_id"]] = len(records)
                    records.append(record)
                    added += 1
                else:
                    records[index] = record
                    replaced += 1
            logger.info(
                "Applied NVD updates since %s: %d records refreshed, %d added",
                since.date(), replaced, added,
            )

    ids = [record["cve_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"NVD pull produced {len(ids) - len(set(ids))} duplicate CVE ids")

    logger.info("Fetched %d CVE records from NVD API 2.0", len(records))
    return records


def _nvd_timestamp(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000+00:00")


def fetch_modified_since(
    since: datetime, until: datetime | None = None, api_key: str | None = None
) -> list[dict]:
    """Every CVE NVD modified in [since, until], chunked into <=120-day windows.

    Pages are cached under data/raw/nvd_updates/<window>/ so the overlay is
    reproducible and resumable like the main pull.
    """
    api_key = api_key or os.environ.get("NVD_API_KEY") or None
    until = until or datetime.now(timezone.utc)
    records: list[dict] = []

    window_start = since
    while window_start < until:
        window_end = min(window_start + MAX_LASTMOD_WINDOW, until)
        # Reuse a window fetched earlier the same day instead of re-pulling it,
        # so a same-day rerun of A1 is free and reproduces the same dataset.
        same_day = sorted(
            UPDATES_CACHE_DIR.glob(f"{window_start:%Y%m%dT%H%M%S}_{window_end:%Y%m%d}T*")
        )
        if same_day:
            window_dir = same_day[-1]
            window_end = datetime.strptime(
                window_dir.name.split("_")[1], "%Y%m%dT%H%M%S"
            ).replace(tzinfo=timezone.utc)
            if window_end < window_start + MAX_LASTMOD_WINDOW:
                until = window_end  # the cached window was the final one: stop there
        else:
            window_dir = UPDATES_CACHE_DIR / (
                f"{window_start:%Y%m%dT%H%M%S}_{window_end:%Y%m%dT%H%M%S}"
            )
        window_dir.mkdir(parents=True, exist_ok=True)
        params = {
            "lastModStartDate": _nvd_timestamp(window_start),
            "lastModEndDate": _nvd_timestamp(window_end),
        }

        start_index, total = 0, None
        while total is None or start_index < total:
            cache_path = window_dir / f"page_{start_index}.json"
            if cache_path.exists():
                payload = json.loads(cache_path.read_text())
            else:
                payload = _fetch_page_params(start_index, RESULTS_PER_PAGE, api_key, params)
                cache_path.write_text(json.dumps(payload))
                time.sleep(_sleep_interval(api_key))
            total = int(payload["totalResults"])
            records.extend(_parse_cve_item(item) for item in payload.get("vulnerabilities", []))
            start_index += RESULTS_PER_PAGE

        logger.info(
            "NVD modified %s .. %s: %d CVEs", window_start.date(), window_end.date(), total
        )
        window_start = window_end

    # A CVE edited in two windows appears twice; keep the latest version.
    latest: dict[str, dict] = {}
    for record in records:
        latest[record["cve_id"]] = record
    return list(latest.values())
