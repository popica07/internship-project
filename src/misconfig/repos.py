"""Shallow-clone the L2 source repositories (idempotent).

Cloning rather than scraping keeps A2 reproducible: the extractors read the
checked-out working tree, and the commit each catalog was built from is
recorded so a later rerun can be pinned to the same snapshot.
"""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

REPOS_DIR = Path("data/raw/repos")

REPOS = {
    "prowler": "https://github.com/prowler-cloud/prowler.git",
    "checkov": "https://github.com/bridgecrewio/checkov.git",
    "scoutsuite": "https://github.com/nccgroup/ScoutSuite.git",
}


def ensure_repos() -> dict[str, Path]:
    """Clone any missing repo (depth 1). Returns {name: local path}."""
    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, url in REPOS.items():
        target = REPOS_DIR / name
        if not target.exists():
            logger.info("Cloning %s -> %s", url, target)
            subprocess.run(
                ["git", "clone", "--depth", "1", url, str(target)],
                check=True,
                capture_output=True,
            )
        paths[name] = target
    return paths


def repo_commit(path: Path) -> str:
    """Short commit SHA of a cloned repo, for provenance."""
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()
