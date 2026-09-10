"""Extract check definitions from a cloned Checkov repository.

Unlike Prowler, Checkov has no metadata files: each check is a Python class
whose `__init__` sets `id`, `name`, `categories` and `supported_resources`,
so they are recovered by parsing the AST (more robust than regex against the
several call styles in the repo). Graph checks are YAML/JSON instead, with a
`metadata:` block.

Important for A3: Checkov's open-source repo ships essentially **no
severities** (they live in the commercial Bridgecrew platform). Severity is
therefore recorded as None here, and `categories` is the only usable
criticality-ish signal on the Checkov side.
"""

import ast
import json
import logging
from pathlib import Path

import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Terraform-style resource prefixes, longest first so `azurerm_` wins over `azure`.
_RESOURCE_PREFIXES = (
    "azurerm_",
    "alicloud_",
    "digitalocean_",
    "openstack_",
    "yandex_",
    "tencentcloud_",
    "google_",
    "linode_",
    "github_",
    "gitlab_",
    "okta_",
    "panos_",
    "aws_",
    "oci_",
    "ncp_",
)


def _resolve(node: ast.AST, local_vars: dict[str, ast.AST], depth: int = 0):
    """Best-effort static evaluation of an AST node into a Python value."""
    if depth > 5 or node is None:
        return None
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [_resolve(e, local_vars, depth + 1) for e in node.elts]
    if isinstance(node, ast.Attribute):
        # e.g. CheckCategories.NETWORKING -> "NETWORKING"
        return node.attr
    if isinstance(node, ast.Name):
        if node.id in local_vars:
            return _resolve(local_vars[node.id], local_vars, depth + 1)
        return None
    return None


def _extract_from_python(path: Path) -> dict | None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        logger.warning("Skipping unparseable Checkov check: %s", path)
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "__init__":
            continue

        local_vars: dict[str, ast.AST] = {}
        super_kwargs: dict[str, ast.AST] = {}

        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        local_vars[target.id] = stmt.value
            elif isinstance(stmt, ast.Call):
                func = stmt.func
                # super().__init__(...) or Base.__init__(self, ...)
                if isinstance(func, ast.Attribute) and func.attr == "__init__":
                    for kw in stmt.keywords:
                        if kw.arg:
                            super_kwargs[kw.arg] = kw.value

        def field(name):
            if name in super_kwargs:
                value = _resolve(super_kwargs[name], local_vars)
                if value is not None:
                    return value
            if name in local_vars:
                return _resolve(local_vars[name], local_vars)
            return None

        check_id = field("id")
        if not isinstance(check_id, str) or not check_id.startswith("CKV"):
            continue  # base class / registry / helper, not an actual check

        resources = field("supported_resources") or field("supported_entities") or []
        if isinstance(resources, str):
            resources = [resources]
        categories = field("categories") or []
        if isinstance(categories, str):
            categories = [categories]

        return {
            "check_id": check_id,
            "title": field("name"),
            "resources": [r for r in resources if isinstance(r, str)],
            "categories": [c for c in categories if isinstance(c, str)],
        }
    return None


def _extract_from_graph_check(path: Path) -> dict | None:
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        doc = json.loads(raw) if path.suffix == ".json" else yaml.safe_load(raw)
    except (yaml.YAMLError, json.JSONDecodeError):
        logger.warning("Skipping unparseable Checkov graph check: %s", path)
        return None
    if not isinstance(doc, dict):
        return None

    meta = doc.get("metadata") or {}
    check_id = meta.get("id")
    if not check_id:
        return None

    definition = doc.get("definition") or {}
    resources = definition.get("resource_types") if isinstance(definition, dict) else None
    category = meta.get("category")

    return {
        "check_id": check_id,
        "title": meta.get("name"),
        "resources": list(resources) if isinstance(resources, list) else [],
        "categories": [category] if category else [],
    }


def _service_from_resources(resources: list[str]) -> str:
    """Heuristic: aws_s3_bucket -> s3, google_compute_instance -> compute."""
    if not resources:
        return ""
    name = resources[0]
    for prefix in _RESOURCE_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name.split("_")[0]


def _path_context(path: Path, checkov_pkg: Path) -> tuple[str, str, str]:
    """Return (framework, entity_type, provider) inferred from the file path."""
    parts = path.relative_to(checkov_pkg).parts
    framework = parts[0] if parts else ""
    try:
        after_checks = parts[parts.index("checks") + 1 : -1]
    except ValueError:
        after_checks = ()
    entity_type = after_checks[0] if after_checks else ""
    provider = after_checks[1] if len(after_checks) > 1 else ""
    return framework, entity_type, provider


def extract(repo_path: Path) -> list[dict]:
    checkov_pkg = repo_path / "checkov"

    py_files = [
        p
        for p in checkov_pkg.rglob("*.py")
        if "checks" in p.parts and p.name != "__init__.py"
    ]
    graph_files = [
        p
        for p in checkov_pkg.rglob("*")
        if "graph_checks" in p.parts and p.suffix in {".yaml", ".yml", ".json"}
    ]

    records = []
    for path in tqdm(sorted(py_files) + sorted(graph_files), desc="Checkov checks", unit="file"):
        parsed = (
            _extract_from_graph_check(path)
            if path.suffix in {".yaml", ".yml", ".json"}
            else _extract_from_python(path)
        )
        if not parsed:
            continue

        framework, entity_type, provider = _path_context(path, checkov_pkg)
        records.append(
            {
                "tool": "checkov",
                "check_id": parsed["check_id"],
                "provider": provider,
                "service": _service_from_resources(parsed["resources"]),
                "title": parsed["title"],
                "description": None,
                "rationale_risk": None,
                # Checkov OSS ships no severities - see module docstring.
                "severity_raw": None,
                "categories": parsed["categories"],
                "resource_type": "|".join(parsed["resources"]),
                "framework": framework,
                "check_type": [entity_type] if entity_type else [],
                "remediation_text": None,
                "remediation_code": None,
                "remediation_url": None,
                "references": [],
                "source_file": str(path.relative_to(repo_path)),
            }
        )

    logger.info("Extracted %d Checkov checks", len(records))
    return records
