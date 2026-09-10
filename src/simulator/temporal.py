"""Finding lifecycle: discovery, SLA, remediation and recurrence.

This is where `sla_breached` and `recurrence_count` come from - two of the five
context terms in the L3 label formula. Both are simulated, both are documented
here and parameterised in config.py.

Remediation is modelled as: a team with a given discipline tries to fix within
the SLA; whether it succeeds depends on that discipline, tightened for critical
assets and relaxed for the long tail. Anything unfixed at the snapshot date is
still open, and anything open past its due date has breached SLA.
"""

import logging

import numpy as np
import pandas as pd

from src.simulator.bank import business_unit_index
from src.simulator.config import SimulatorConfig

logger = logging.getLogger(__name__)


def apply_lifecycle(
    findings: pd.DataFrame,
    assets: pd.DataFrame,
    config: SimulatorConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    bu_index = business_unit_index(config)
    asset_context = assets.set_index("asset_id")

    n = len(findings)
    context = asset_context.loc[findings["asset_id"]]

    # ---- discovery -------------------------------------------------------
    snapshot = pd.Timestamp.utcnow().normalize()
    age_days = rng.integers(0, config.simulation_window_days, size=n)
    first_seen = snapshot - pd.to_timedelta(age_days, unit="D")

    # ---- SLA -------------------------------------------------------------
    sla_map = config.sla_days()
    sla_days = findings["base_severity"].map(sla_map).fillna(sla_map["medium"]).to_numpy(dtype=float)
    internet_facing = context["internet_facing"].to_numpy()
    sla_days = np.where(internet_facing, sla_days * config.internet_facing_sla_multiplier, sla_days)
    sla_days = np.maximum(sla_days.round(), 1).astype(int)
    due_date = first_seen + pd.to_timedelta(sla_days, unit="D")

    # ---- remediation -----------------------------------------------------
    discipline = np.array(
        [bu_index[bu].remediation_discipline for bu in context["business_unit"]]
    )
    # Critical-tier assets get more attention; tier 4 gets less.
    tier_adjustment = 1.0 + 0.10 * (2 - context["criticality_tier"].to_numpy())
    fix_probability = np.clip(discipline * tier_adjustment, 0.02, 0.98)

    # Only findings whose SLA window has already elapsed can have been resolved
    # in time; the rest are simply still in flight.
    window_elapsed = age_days >= sla_days
    fixed_in_time = window_elapsed & (rng.random(n) < fix_probability)

    # Late fixes: some of the missed ones are eventually resolved anyway.
    late_fixed = window_elapsed & ~fixed_in_time & (rng.random(n) < 0.45)

    resolved = fixed_in_time | late_fixed
    days_to_fix = np.where(
        fixed_in_time,
        rng.integers(1, np.maximum(sla_days, 2)),
        sla_days + rng.integers(1, 90, size=n),
    )
    resolved_date = pd.Series(pd.NaT, index=range(n), dtype="datetime64[ns, UTC]")
    resolved_date[resolved] = (first_seen + pd.to_timedelta(days_to_fix, unit="D"))[resolved]
    # A fix cannot land in the future relative to the snapshot.
    still_future = resolved_date > snapshot
    resolved_date[still_future] = pd.NaT
    resolved = resolved & ~still_future

    status = np.where(resolved, "resolved", "open")
    # Breach = still open past due, or resolved after due.
    breached_open = (~resolved) & (snapshot > due_date)
    breached_late = resolved & (resolved_date > due_date)
    sla_breached = breached_open | breached_late

    # ---- recurrence ------------------------------------------------------
    # Only resolved findings can recur; each recurrence rolls again.
    recurrence_count = np.zeros(n, dtype=int)
    eligible = resolved.copy()
    for _ in range(config.max_recurrences):
        recurs = eligible & (rng.random(n) < config.recurrence_probability)
        recurrence_count += recurs.astype(int)
        eligible = recurs
        if not recurs.any():
            break

    lifecycle = findings.copy()
    lifecycle["first_seen"] = first_seen.values
    lifecycle["sla_days"] = sla_days
    lifecycle["due_date"] = due_date.values
    lifecycle["resolved_date"] = resolved_date.values
    lifecycle["status"] = status
    lifecycle["sla_breached"] = sla_breached
    lifecycle["recurrence_count"] = recurrence_count
    lifecycle["age_days"] = age_days
    lifecycle["snapshot_date"] = snapshot

    logger.info(
        "Lifecycle: %.1f%% open, %.1f%% SLA-breached, mean recurrence %.2f",
        100 * (lifecycle["status"] == "open").mean(),
        100 * lifecycle["sla_breached"].mean(),
        lifecycle["recurrence_count"].mean(),
    )
    return lifecycle
