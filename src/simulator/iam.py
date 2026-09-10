"""IAM graph simulation and the blast-radius features derived from it.

Why a graph at all: an asset's real exposure is not just its own configuration
but who can reach it and what else those principals can reach. The graph gives
each asset a *blast radius* - how many other assets an attacker landing on it
could pivot to through shared principals - which is context no single-finding
severity captures.

Structure:
    principal --can_access--> asset
    principal --can_assume--> principal   (privilege escalation edges)

Blast radius of an asset = assets reachable from the principals that can
access it, following assume-role edges.
"""

import logging

import networkx as nx
import numpy as np
import pandas as pd

from src.simulator.config import SimulatorConfig

logger = logging.getLogger(__name__)


def build_iam_graph(
    assets: pd.DataFrame, config: SimulatorConfig, rng: np.random.Generator
) -> tuple[nx.DiGraph, pd.DataFrame]:
    """Build the principal/asset graph and return it with per-principal metadata."""
    n_principals = max(1, int(len(assets) * config.principals_per_100_assets / 100))

    business_units = assets["business_unit"].to_numpy()
    unique_bus = sorted(set(business_units))
    principal_bus = rng.choice(unique_bus, size=n_principals)
    is_admin = rng.random(n_principals) < config.admin_principal_share

    principals = pd.DataFrame(
        {
            "principal_id": [f"principal-{i:05d}" for i in range(n_principals)],
            "business_unit": principal_bus,
            "is_admin": is_admin,
            "principal_type": rng.choice(
                ["service_account", "human_user", "role"], size=n_principals, p=[0.55, 0.30, 0.15]
            ),
        }
    )

    graph = nx.DiGraph()
    graph.add_nodes_from(principals["principal_id"], kind="principal")
    graph.add_nodes_from(assets["asset_id"], kind="asset")

    # Index assets by BU so most grants stay inside a business unit.
    assets_by_bu = {bu: assets.index[assets["business_unit"] == bu].to_numpy() for bu in unique_bus}
    all_asset_positions = assets.index.to_numpy()
    asset_ids = assets["asset_id"].to_numpy()

    for principal_id, bu, admin in zip(
        principals["principal_id"], principals["business_unit"], principals["is_admin"]
    ):
        # Admins hold materially more grants - that is what makes them worth stealing.
        mean_grants = config.mean_assets_per_principal * (4.0 if admin else 1.0)
        n_grants = max(1, int(rng.poisson(mean_grants)))

        in_bu = assets_by_bu[bu]
        cross_bu = rng.random(n_grants) < config.cross_bu_edge_probability
        n_cross = int(cross_bu.sum())
        n_in_bu = n_grants - n_cross

        chosen = []
        if n_in_bu > 0 and len(in_bu) > 0:
            chosen.append(rng.choice(in_bu, size=min(n_in_bu, len(in_bu)), replace=False))
        if n_cross > 0:
            chosen.append(rng.choice(all_asset_positions, size=n_cross, replace=False))
        if not chosen:
            continue

        for position in np.concatenate(chosen):
            graph.add_edge(principal_id, asset_ids[position], relation="can_access")

    # Privilege-escalation edges between principals.
    principal_ids = principals["principal_id"].to_numpy()
    n_assume = int(len(principal_ids) * config.assume_role_edge_probability * 10)
    for _ in range(n_assume):
        source, target = rng.choice(principal_ids, size=2, replace=False)
        graph.add_edge(source, target, relation="can_assume")

    logger.info(
        "Built IAM graph: %d principals, %d assets, %d edges",
        n_principals,
        len(assets),
        graph.number_of_edges(),
    )
    return graph, principals


def asset_exposure_features(
    graph: nx.DiGraph, assets: pd.DataFrame, principals: pd.DataFrame
) -> pd.DataFrame:
    """Per-asset IAM features: who can reach it, and what that unlocks."""
    admin_principals = set(principals.loc[principals["is_admin"], "principal_id"])

    # Assets each principal can reach, following assume-role edges first.
    reachable_assets: dict[str, set[str]] = {}
    for principal_id in principals["principal_id"]:
        reachable_principals = {principal_id} | {
            node
            for node in nx.descendants(graph, principal_id)
            if graph.nodes[node].get("kind") == "principal"
        }
        assets_reached = set()
        for reached in reachable_principals:
            assets_reached.update(
                target
                for target in graph.successors(reached)
                if graph.nodes[target].get("kind") == "asset"
            )
        reachable_assets[principal_id] = assets_reached

    rows = []
    for asset_id in assets["asset_id"]:
        accessors = [
            source
            for source in graph.predecessors(asset_id)
            if graph.nodes[source].get("kind") == "principal"
        ]
        blast: set[str] = set()
        for principal_id in accessors:
            blast.update(reachable_assets.get(principal_id, ()))
        blast.discard(asset_id)

        rows.append(
            {
                "asset_id": asset_id,
                "principal_count": len(accessors),
                "admin_principal_count": sum(1 for p in accessors if p in admin_principals),
                "blast_radius": len(blast),
            }
        )

    features = pd.DataFrame(rows)
    features["has_admin_path"] = features["admin_principal_count"] > 0
    logger.info(
        "IAM features: median blast radius %.0f assets, %.1f%% of assets on an admin path",
        features["blast_radius"].median(),
        100 * features["has_admin_path"].mean(),
    )
    return features
