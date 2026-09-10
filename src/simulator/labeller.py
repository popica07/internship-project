"""L3 priority label computation - the formula fixed in CLAUDE.md.

    priority_score = 0.60 * real_component + 0.40 * context_component

    real (vulnerability)    = 0.40*epss + 0.35*in_kev + 0.25*public_exploit
    real (misconfiguration) = severity_map[severity]
    context                 = 0.30*tier_weight + 0.20*internet_facing
                            + 0.20*(pci OR dora) + 0.15*sla_breached
                            + 0.15*min(recurrence/5, 1)

    >=0.80 Critical, >=0.60 High, >=0.40 Medium, else Low

THE SCIENTIFIC DEFENSIBILITY RULE (CLAUDE.md): this label is 60% real signal
and 40% simulated context, and it is a *documented formula*, not ground truth.
It must never be presented as an external label the way CISA KEV is, and L3
results must always be reported separately from L1/L2 results.

Consequence worth stating plainly for Study C: a model trained on this label
is partly learning a formula we wrote. High accuracy on L3 is therefore not
evidence that the approach works on real bank data - it is evidence the model
can recover a deterministic function of its own inputs. The informative L3
experiments are the ones about *robustness* (label noise, weight sweeps,
feature ablation), not raw predictive performance.
"""

import logging

import numpy as np
import pandas as pd

from src.simulator.config import PRIORITY_ORDER, SimulatorConfig

logger = logging.getLogger(__name__)


def compute_real_component(findings: pd.DataFrame, config: SimulatorConfig) -> np.ndarray:
    """External signal: EPSS/KEV/exploit for CVEs, tool severity for misconfigs."""
    is_vuln = findings["source_layer"].to_numpy() == "L1"

    # Rank, not raw probability - see `use_epss_percentile` in config.py.
    epss_column = "epss_percentile" if config.use_epss_percentile else "epss_score"
    epss = findings[epss_column].fillna(0.0).to_numpy(dtype=float)
    in_kev = findings["in_kev"].fillna(0).to_numpy(dtype=float)
    public_exploit = findings["public_exploit"].fillna(0).to_numpy(dtype=float)
    vuln_component = config.w_epss * epss + config.w_in_kev * in_kev + config.w_public_exploit * public_exploit

    # Scaled onto the vulnerability layer's range - an explicit, swept choice.
    misconfig_component = (
        findings["base_severity"].map(config.severity_score).to_numpy(dtype=float)
        * config.misconfig_scale
    )

    return np.where(is_vuln, vuln_component, misconfig_component)


def compute_context_component(
    findings: pd.DataFrame, assets: pd.DataFrame, config: SimulatorConfig
) -> np.ndarray:
    context = assets.set_index("asset_id").loc[findings["asset_id"]]

    tier_weights = np.array(config.tier_weights)
    tier_term = tier_weights[context["criticality_tier"].to_numpy()]
    internet_term = context["internet_facing"].to_numpy(dtype=float)
    regulatory_term = context["regulatory_scope"].to_numpy(dtype=float)
    sla_term = findings["sla_breached"].to_numpy(dtype=float)
    recurrence_term = np.minimum(
        findings["recurrence_count"].to_numpy(dtype=float) / config.recurrence_cap, 1.0
    )

    return (
        config.w_tier * tier_term
        + config.w_internet_facing * internet_term
        + config.w_regulatory * regulatory_term
        + config.w_sla_breached * sla_term
        + config.w_recurrence * recurrence_term
    )


def assign_labels(scores: np.ndarray, config: SimulatorConfig) -> np.ndarray:
    labels = np.full(len(scores), "Low", dtype=object)
    labels[scores >= config.threshold_medium] = "Medium"
    labels[scores >= config.threshold_high] = "High"
    labels[scores >= config.threshold_critical] = "Critical"
    return labels


def inject_label_noise(
    labels: np.ndarray, config: SimulatorConfig, rng: np.random.Generator
) -> np.ndarray:
    """Bump a share of labels to an adjacent class.

    Models the fact that a real analyst's priority call is not a deterministic
    function of the inputs. Off by default; swept in Study C.
    """
    if config.noise_level <= 0:
        return labels

    order = list(PRIORITY_ORDER)
    index = np.array([order.index(label) for label in labels])
    flip = rng.random(len(labels)) < config.noise_level
    direction = rng.choice([-1, 1], size=len(labels))
    index[flip] = np.clip(index[flip] + direction[flip], 0, len(order) - 1)
    return np.array([order[i] for i in index], dtype=object)


def label_findings(
    findings: pd.DataFrame,
    assets: pd.DataFrame,
    config: SimulatorConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    real = compute_real_component(findings, config)
    context = compute_context_component(findings, assets, config)
    score = config.real_component_weight * real + config.context_component_weight * context

    labelled = findings.copy()
    labelled["real_component"] = real
    labelled["context_component"] = context
    labelled["priority_score"] = score
    labelled["priority_label_clean"] = assign_labels(score, config)
    labelled["priority_label"] = inject_label_noise(
        labelled["priority_label_clean"].to_numpy(), config, rng
    )

    distribution = labelled["priority_label"].value_counts(normalize=True)
    logger.info(
        "Labels: %s",
        ", ".join(f"{label} {100 * distribution.get(label, 0):.1f}%" for label in PRIORITY_ORDER),
    )
    return labelled
