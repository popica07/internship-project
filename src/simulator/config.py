"""Every tunable parameter of the L3 estate simulator, in one place.

CLAUDE.md rule 7: nothing undocumented, everything sweepable. Each field below
carries the reasoning for its default. Nothing in the simulator may read a
magic number that is not defined here.

The estate is a *parameterised* financial institution, not a model of any real
bank - no Deutsche Bank data was used or available. Defaults are chosen to be
plausible for a large EU retail+investment bank and are all sweepable.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BusinessUnit:
    """One line of business, with the properties that drive asset criticality."""

    name: str
    weight: float  # share of the estate
    criticality_bias: float  # 0..1, pushes assets toward lower (more critical) tiers
    pci_likelihood: float  # P(asset is in PCI-DSS scope)
    dora_likelihood: float  # P(asset is in DORA scope) - EU financial-entity regulation
    remediation_discipline: float  # 0..1, P(fix within SLA); mature teams are higher


@dataclass(frozen=True)
class SimulatorConfig:
    # ---- reproducibility -------------------------------------------------
    seed: int = 42

    # ---- estate size -----------------------------------------------------
    # 5k assets / ~40k findings keeps a full run inside a couple of minutes
    # while staying large enough for grouped-by-asset splits to be meaningful.
    n_assets: int = 5_000

    # ---- bank structure --------------------------------------------------
    business_units: tuple[BusinessUnit, ...] = (
        # Payments and Retail carry card data -> high PCI exposure.
        BusinessUnit("Payments", 0.14, 0.85, 0.80, 0.95, 0.75),
        BusinessUnit("Retail Banking", 0.22, 0.65, 0.55, 0.90, 0.65),
        BusinessUnit("Investment Banking", 0.18, 0.75, 0.10, 0.90, 0.70),
        BusinessUnit("Wealth Management", 0.12, 0.55, 0.15, 0.85, 0.60),
        BusinessUnit("Risk & Compliance", 0.10, 0.70, 0.05, 0.95, 0.80),
        # Corporate IT is the long tail: big, less critical, weaker discipline.
        BusinessUnit("Corporate IT", 0.16, 0.25, 0.05, 0.40, 0.45),
        BusinessUnit("Digital Channels", 0.08, 0.60, 0.45, 0.75, 0.70),
    )

    # Environment mix. Non-prod dominates by count in most real estates.
    environments: tuple[tuple[str, float], ...] = (
        ("production", 0.35),
        ("staging", 0.20),
        ("development", 0.33),
        ("dr", 0.12),
    )
    # Only production/DR assets can be business-critical (tier 0/1).
    non_prod_tier_floor: int = 2

    # EU-centric footprint with US/APAC presence.
    regions: tuple[tuple[str, float], ...] = (
        ("eu-central-1", 0.34),
        ("eu-west-1", 0.26),
        ("us-east-1", 0.22),
        ("ap-southeast-1", 0.10),
        ("us-west-2", 0.08),
    )
    providers: tuple[tuple[str, float], ...] = (
        ("aws", 0.55),
        ("azure", 0.32),
        ("gcp", 0.13),
    )

    # ---- exposure --------------------------------------------------------
    # P(internet-facing), by environment. Production edge services dominate.
    internet_facing_by_env: tuple[tuple[str, float], ...] = (
        ("production", 0.22),
        ("staging", 0.10),
        ("development", 0.06),
        ("dr", 0.12),
    )

    # ---- IAM graph -------------------------------------------------------
    principals_per_100_assets: int = 22  # humans + service accounts
    admin_principal_share: float = 0.06  # share of principals with admin-ish power
    mean_assets_per_principal: float = 9.0
    cross_bu_edge_probability: float = 0.08  # principal granted outside its own BU
    assume_role_edge_probability: float = 0.04  # principal -> principal escalation

    # ---- findings injection ---------------------------------------------
    mean_misconfig_findings_per_asset: float = 5.5
    mean_vuln_findings_per_asset: float = 2.5
    # Vulnerabilities are sampled from a recent window; older CVEs are far less
    # likely to still be present on a live estate.
    vuln_recency_window_days: int = 1_460  # 4 years
    # Sampling weight decay: newer CVEs are more likely to be present.
    vuln_recency_half_life_days: int = 365

    # ---- temporal dynamics ----------------------------------------------
    simulation_window_days: int = 540  # findings first seen within ~18 months
    # SLA by finding severity, in days. Typical large-bank policy.
    sla_days_by_severity: tuple[tuple[str, int], ...] = (
        ("critical", 7),
        ("high", 30),
        ("medium", 90),
        ("low", 180),
        ("informational", 365),
    )
    # Internet-facing assets get a tightened SLA (common policy).
    internet_facing_sla_multiplier: float = 0.5
    # P(recurrence) after a fix - drift/redeploy reintroduces the finding.
    recurrence_probability: float = 0.18
    max_recurrences: int = 8

    # ---- label formula (fixed by CLAUDE.md; weights sweepable in C7) -----
    real_component_weight: float = 0.60
    context_component_weight: float = 0.40

    # Real component - vulnerabilities (weights sum to 1.0)
    w_epss: float = 0.40
    w_in_kev: float = 0.35
    w_public_exploit: float = 0.25

    # Use EPSS *percentile* rather than the raw probability in the L3 label.
    # A6/A5 rationale: the L3 label is an ordinal priority class, so it needs
    # discrimination, not calibration. Raw EPSS is ~0.003 for a typical CVE, so
    # 74.9% of vulnerabilities squashed below 0.1 and 99.4% labelled Low. The
    # percentile is the same real EPSS data expressed as a rank within the CVE
    # population, and restores within-layer variance. Raw EPSS is still used
    # unchanged in L1 experiments (B2/B3), where calibration is the point.
    use_epss_percentile: bool = True

    # Explicit cross-layer balance. Multiplies the misconfiguration severity
    # map so the two layers' real components live on a comparable scale.
    # This is a VALUE JUDGEMENT, not a fact in the data - "is a medium
    # misconfiguration more urgent than a 60th-percentile CVE?" is the bank's
    # call. It is therefore stated as a parameter and swept in C7 rather than
    # left to be decided accidentally by a units mismatch.
    misconfig_scale: float = 0.55

    # Real component - misconfigurations
    misconfig_severity_map: tuple[tuple[str, float], ...] = (
        ("critical", 1.00),
        ("high", 0.75),
        ("medium", 0.50),
        ("low", 0.25),
        ("informational", 0.10),
        ("unknown", 0.50),  # Checkov ships no severity - treated as medium
    )

    # Context component (weights sum to 1.0)
    w_tier: float = 0.30
    w_internet_facing: float = 0.20
    w_regulatory: float = 0.20
    w_sla_breached: float = 0.15
    w_recurrence: float = 0.15
    recurrence_cap: int = 5  # min(recurrence_count / 5, 1.0)

    # Tier 0 = most critical .. Tier 4 = least
    tier_weights: tuple[float, ...] = (1.00, 0.75, 0.50, 0.25, 0.00)

    # Label thresholds.
    # CLAUDE.md's original 0.80/0.60/0.40 were calibrated for the original
    # component scales. Changing `use_epss_percentile` and `misconfig_scale`
    # shifts the score distribution, so the thresholds had to move with them -
    # they are coupled, and changing one alone silently changes the class
    # sizes. These values are calibrated to reproduce the ORIGINAL operational
    # class sizes (~0.4% Critical, ~5.7% High, ~37% Medium) under the default
    # config. C7 sweeps `misconfig_scale`; any sweep must recalibrate these or
    # report the resulting class-size shift.
    threshold_critical: float = 0.545
    threshold_high: float = 0.420
    threshold_medium: float = 0.285

    # Label noise: P(a label is bumped to an adjacent class). 0.0 disables.
    noise_level: float = 0.0

    def sla_days(self) -> dict[str, int]:
        return dict(self.sla_days_by_severity)

    def severity_score(self, severity: str | None) -> float:
        return dict(self.misconfig_severity_map).get(severity or "unknown", 0.50)


DEFAULT_CONFIG = SimulatorConfig()

# Label ordering, low -> high, used by the noise injector and for reporting.
PRIORITY_ORDER: tuple[str, ...] = ("Low", "Medium", "High", "Critical")


def as_records(config: SimulatorConfig) -> list[dict]:
    """Flatten the config for provenance - written alongside the dataset."""
    records = []
    for key, value in vars(config).items():
        if key == "business_units":
            for bu in value:
                for bu_key, bu_value in vars(bu).items():
                    if bu_key != "name":
                        records.append({"parameter": f"business_unit.{bu.name}.{bu_key}", "value": bu_value})
        elif isinstance(value, tuple) and value and isinstance(value[0], tuple):
            for item_key, item_value in value:
                records.append({"parameter": f"{key}.{item_key}", "value": item_value})
        else:
            records.append({"parameter": key, "value": value})
    return records


_ = field  # re-exported for downstream configs that extend this one
