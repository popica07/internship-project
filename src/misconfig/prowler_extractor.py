"""Extract check definitions from a cloned Prowler repository.

Prowler ships one `<check>.metadata.json` per check under
`prowler/providers/<provider>/services/<service>/<check>/`, with a rich,
already-structured schema (severity, description, risk, remediation).
Of the three L2 tools this is the only one that ships a full severity scale.
"""

import json
import logging
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)


def extract(repo_path: Path) -> list[dict]:
    metadata_files = sorted(repo_path.rglob("*.metadata.json"))
    records = []

    for path in tqdm(metadata_files, desc="Prowler checks", unit="check"):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning("Skipping unparseable Prowler metadata: %s", path)
            continue

        remediation = meta.get("Remediation") or {}
        recommendation = remediation.get("Recommendation") or {}
        code = remediation.get("Code") or {}

        references = [meta.get("RelatedUrl")] + list(meta.get("AdditionalURLs") or [])

        records.append(
            {
                "tool": "prowler",
                "check_id": meta.get("CheckID"),
                "provider": meta.get("Provider"),
                "service": meta.get("ServiceName"),
                "title": meta.get("CheckTitle"),
                "description": meta.get("Description"),
                "rationale_risk": meta.get("Risk"),
                "severity_raw": meta.get("Severity"),
                "categories": list(meta.get("Categories") or []),
                "resource_type": meta.get("ResourceType"),
                "framework": "",
                "check_type": list(meta.get("CheckType") or []),
                "remediation_text": recommendation.get("Text"),
                "remediation_code": code.get("Terraform") or code.get("CLI") or code.get("Other"),
                "remediation_url": recommendation.get("Url"),
                "references": [r for r in references if r],
                "source_file": str(path.relative_to(repo_path)),
            }
        )

    logger.info("Extracted %d Prowler checks", len(records))
    return records
