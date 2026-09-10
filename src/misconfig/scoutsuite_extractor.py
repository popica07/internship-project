"""Extract finding rules from a cloned ScoutSuite repository.

ScoutSuite splits a rule across two files: the finding itself lives in
`providers/<provider>/rules/findings/<name>.json` (description, rationale,
references, remediation) while its severity lives in a *ruleset*
(`rules/rulesets/default.json`) that maps the finding filename to a `level`.
Both are joined here.

ScoutSuite's severity scale has only two usable levels (`danger`, `warning`),
which is coarser than Prowler's five - relevant when A3 compares severities
across tools.
"""

import json
import logging
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)

# Rulesets are consulted in order; the first one defining a finding wins.
RULESET_PRIORITY = ("default.json", "detailed.json")


def _load_ruleset_levels(rules_dir: Path) -> dict[str, tuple[str, str]]:
    """Map finding filename -> (level, ruleset name) for one provider."""
    levels: dict[str, tuple[str, str]] = {}
    for ruleset_name in RULESET_PRIORITY:
        ruleset_path = rules_dir / "rulesets" / ruleset_name
        if not ruleset_path.exists():
            continue
        try:
            ruleset = json.loads(ruleset_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Skipping unparseable ScoutSuite ruleset: %s", ruleset_path)
            continue

        for filename, entries in (ruleset.get("rules") or {}).items():
            if filename in levels or not entries:
                continue
            level = entries[0].get("level") if isinstance(entries[0], dict) else None
            if level:
                levels[filename] = (level, ruleset_name)
    return levels


def extract(repo_path: Path) -> list[dict]:
    providers_dir = repo_path / "ScoutSuite" / "providers"
    records = []

    finding_files = sorted(providers_dir.glob("*/rules/findings/*.json"))
    levels_by_provider: dict[str, dict[str, tuple[str, str]]] = {}

    for path in tqdm(finding_files, desc="ScoutSuite findings", unit="rule"):
        provider = path.relative_to(providers_dir).parts[0]
        if provider not in levels_by_provider:
            levels_by_provider[provider] = _load_ruleset_levels(providers_dir / provider / "rules")

        try:
            finding = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Skipping unparseable ScoutSuite finding: %s", path)
            continue

        level, ruleset = levels_by_provider[provider].get(path.name, (None, None))

        # "acm.regions.id.certificates.id" -> service "acm"
        rule_path = finding.get("path") or ""
        service = rule_path.split(".")[0] if rule_path else path.stem.split("-")[0]

        remediation = finding.get("remediation")
        if isinstance(remediation, list):
            remediation = " ".join(str(r) for r in remediation)

        records.append(
            {
                "tool": "scoutsuite",
                "check_id": path.stem,
                "provider": provider,
                "service": service,
                "title": finding.get("description"),
                "description": finding.get("rationale"),
                "rationale_risk": finding.get("rationale"),
                "severity_raw": level,
                "categories": [finding.get("dashboard_name")] if finding.get("dashboard_name") else [],
                "resource_type": rule_path,
                "framework": "",
                "check_type": [ruleset] if ruleset else [],
                "remediation_text": remediation,
                "remediation_code": None,
                "remediation_url": None,
                "references": list(finding.get("references") or []),
                "source_file": str(path.relative_to(repo_path)),
            }
        )

    logger.info("Extracted %d ScoutSuite findings", len(records))
    return records
