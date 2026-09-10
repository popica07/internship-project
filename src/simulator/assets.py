"""Generate the simulated asset estate.

Each asset carries the context features the L3 label formula consumes:
criticality tier, internet exposure, and regulatory scope (PCI / DORA).
"""

import logging

import numpy as np
import pandas as pd

from src.simulator.bank import business_unit_index, draw_business_units, weighted_choice
from src.simulator.config import SimulatorConfig

logger = logging.getLogger(__name__)

ASSET_TYPES: tuple[tuple[str, float], ...] = (
    ("compute_instance", 0.24),
    ("container_workload", 0.18),
    ("managed_database", 0.13),
    ("object_storage", 0.12),
    ("serverless_function", 0.10),
    ("load_balancer", 0.07),
    ("message_queue", 0.06),
    ("secrets_store", 0.05),
    ("data_warehouse", 0.05),
)

# Asset types that can plausibly be exposed to the internet at all.
INTERNET_CAPABLE = {"compute_instance", "container_workload", "object_storage",
                    "serverless_function", "load_balancer"}


def generate_assets(config: SimulatorConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = config.n_assets
    bu_names = draw_business_units(rng, config, n)
    bu_index = business_unit_index(config)

    environments = weighted_choice(rng, config.environments, n)
    providers = weighted_choice(rng, config.providers, n)
    regions = weighted_choice(rng, config.regions, n)
    asset_types = weighted_choice(rng, ASSET_TYPES, n)

    criticality_bias = np.array([bu_index[name].criticality_bias for name in bu_names])

    # Tier 0 (most critical) .. Tier 4. A higher BU bias shifts mass toward 0.
    # Beta draw gives a smooth skew rather than hard buckets.
    tier_draw = rng.beta(1.0 + 3.0 * criticality_bias, 1.0 + 3.0 * (1 - criticality_bias))
    tiers = np.clip((4 * (1 - tier_draw)).round().astype(int), 0, 4)

    # Only production and DR can hold business-critical tiers.
    non_prod = ~np.isin(environments, ["production", "dr"])
    tiers[non_prod] = np.maximum(tiers[non_prod], config.non_prod_tier_floor)

    internet_p = dict(config.internet_facing_by_env)
    internet_facing = (
        (rng.random(n) < np.array([internet_p[e] for e in environments]))
        & np.isin(asset_types, list(INTERNET_CAPABLE))
    )

    pci_likelihood = np.array([bu_index[name].pci_likelihood for name in bu_names])
    dora_likelihood = np.array([bu_index[name].dora_likelihood for name in bu_names])
    # Regulatory scope applies to production/DR data paths, not dev sandboxes.
    prod_like = np.isin(environments, ["production", "dr"])
    pci_scope = (rng.random(n) < pci_likelihood) & prod_like
    dora_scope = (rng.random(n) < dora_likelihood) & prod_like

    assets = pd.DataFrame(
        {
            "asset_id": [f"asset-{i:06d}" for i in range(n)],
            "business_unit": bu_names,
            "environment": environments,
            "provider": providers,
            "region": regions,
            "asset_type": asset_types,
            "criticality_tier": tiers,
            "internet_facing": internet_facing,
            "pci_scope": pci_scope,
            "dora_scope": dora_scope,
        }
    )
    assets["regulatory_scope"] = assets["pci_scope"] | assets["dora_scope"]

    logger.info(
        "Generated %d assets (%.1f%% internet-facing, %.1f%% in regulatory scope)",
        len(assets),
        100 * assets["internet_facing"].mean(),
        100 * assets["regulatory_scope"].mean(),
    )
    return assets
