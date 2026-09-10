"""Inject real L1 (CVE) and L2 (misconfiguration) findings onto simulated assets.

The findings themselves are never invented: every row points at a real CVE from
NVD/EPSS/KEV or a real check from Prowler/Checkov/ScoutSuite. What is simulated
is only *which asset a real finding landed on*.

Two documented simplifications:

1. **Misconfigurations** are drawn from checks matching the asset's cloud
   provider. Matching on service as well would be more faithful, but the L2
   service vocabulary (474 values, tool-specific spellings) does not align
   cleanly with the simulated asset types, so provider-level matching is used
   and the service is taken from the drawn check.
2. **Vulnerabilities** are drawn from a recent window with an exponential
   recency weight. Faithful CVE-to-asset mapping would need CPE matching
   against a real software inventory, which does not exist here - so CVE
   presence is a weighted draw rather than a claim about applicability.
"""

import logging

import numpy as np
import pandas as pd

from src.simulator.config import SimulatorConfig

logger = logging.getLogger(__name__)


def _recency_weights(published: pd.Series, config: SimulatorConfig) -> np.ndarray:
    """Exponential decay by age: newer CVEs are likelier to be live on an estate."""
    newest = published.max()
    age_days = (newest - published).dt.total_seconds().to_numpy() / 86_400
    return np.exp(-np.log(2) * age_days / config.vuln_recency_half_life_days)


def inject_findings(
    assets: pd.DataFrame,
    l1: pd.DataFrame,
    l2: pd.DataFrame,
    config: SimulatorConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    # ---- candidate vulnerability pool ------------------------------------
    l1 = l1.copy()
    # format="ISO8601" is required: without it pandas infers one format and
    # silently NaTs every row that differs (e.g. fractional seconds present
    # on some rows only).
    l1["published_date"] = pd.to_datetime(l1["published_date"], format="ISO8601", utc=True, errors="coerce")

    # Fail loudly rather than quietly drawing from a decimated pool: a silent
    # parse failure here once cut the candidate CVEs from ~100k to 715.
    unparsed = l1["published_date"].isna().mean()
    if unparsed > 0.01:
        raise ValueError(
            f"{unparsed:.1%} of L1 published_date values failed to parse - "
            "the timestamp format in l1_vulnerabilities.csv is not uniform."
        )
    l1 = l1.dropna(subset=["published_date"])
    cutoff = l1["published_date"].max() - pd.Timedelta(days=config.vuln_recency_window_days)
    vuln_pool = l1[l1["published_date"] >= cutoff].reset_index(drop=True)
    weights = _recency_weights(vuln_pool["published_date"], config)
    weights = weights / weights.sum()
    logger.info("Vulnerability pool: %d CVEs published since %s", len(vuln_pool), cutoff.date())

    # ---- candidate misconfiguration pool, indexed by provider ------------
    l2_by_provider = {
        provider: group.reset_index(drop=True)
        for provider, group in l2.groupby(l2["provider"].fillna(""))
    }
    generic_l2 = l2.reset_index(drop=True)  # fallback when a provider has no checks

    n_assets = len(assets)
    n_misconfig = rng.poisson(config.mean_misconfig_findings_per_asset, size=n_assets)
    n_vuln = rng.poisson(config.mean_vuln_findings_per_asset, size=n_assets)

    # ---- draw misconfiguration findings ----------------------------------
    misconfig_rows = []
    for provider, group in assets.groupby("provider"):
        pool = l2_by_provider.get(provider, generic_l2)
        counts = n_misconfig[group.index.to_numpy()]
        total = int(counts.sum())
        if total == 0:
            continue
        picks = rng.integers(0, len(pool), size=total)
        picked = pool.iloc[picks]
        misconfig_rows.append(
            pd.DataFrame(
                {
                    "asset_id": np.repeat(group["asset_id"].to_numpy(), counts),
                    "source_layer": "L2",
                    "finding_ref": picked["catalog_id"].to_numpy(),
                    "title": picked["title"].to_numpy(),
                    "tool": picked["tool"].to_numpy(),
                    "service": picked["service"].to_numpy(),
                    "base_severity": picked["severity_normalised"].to_numpy(),
                    "epss_score": np.nan,
                    "epss_percentile": np.nan,
                    "in_kev": 0,
                    "public_exploit": 0,
                    "cvss_score": np.nan,
                }
            )
        )

    # ---- draw vulnerability findings -------------------------------------
    total_vulns = int(n_vuln.sum())
    picks = rng.choice(len(vuln_pool), size=total_vulns, p=weights)
    picked = vuln_pool.iloc[picks]
    vuln_frame = pd.DataFrame(
        {
            "asset_id": np.repeat(assets["asset_id"].to_numpy(), n_vuln),
            "source_layer": "L1",
            "finding_ref": picked["cve_id"].to_numpy(),
            "title": picked["cve_id"].to_numpy(),
            "tool": "nvd",
            "service": "",
            "base_severity": picked["cvss_severity"].str.lower().to_numpy(),
            "epss_score": picked["epss_score"].to_numpy(),
            "epss_percentile": picked["epss_percentile"].to_numpy(),
            "in_kev": picked["in_kev"].to_numpy(),
            "public_exploit": picked["public_exploit"].to_numpy(),
            "cvss_score": picked["cvss_score"].to_numpy(),
        }
    )

    findings = pd.concat(misconfig_rows + [vuln_frame], ignore_index=True)
    findings["finding_id"] = [f"finding-{i:07d}" for i in range(len(findings))]
    findings["base_severity"] = findings["base_severity"].fillna("unknown")

    logger.info(
        "Injected %d findings (%d misconfiguration, %d vulnerability) across %d assets",
        len(findings),
        len(findings) - len(vuln_frame),
        len(vuln_frame),
        n_assets,
    )
    return findings
