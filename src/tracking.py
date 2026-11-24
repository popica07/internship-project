"""MLflow logging shared by every experiment (CLAUDE.md rule 6).

One call per experiment run:

    log_experiment("A3", params={...}, metrics={...}, artifacts=[Path(...)])

Runs land in a local SQLite store under experiments/ (gitignored). Browse with:

    mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db

Each run is tagged with the git commit and whether the tree was dirty, so a
logged number can always be traced back to the code that produced it.
Logging failures are reported loudly but never discard a finished experiment's
outputs - the CSVs and figures are already on disk by the time this runs.
"""

import logging
import math
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

TRACKING_URI = "sqlite:///experiments/mlflow.db"
ARTIFACT_ROOT = Path("experiments/artifacts")
EXPERIMENT_NAMES = {
    "A": "study_a_data_foundation",
    "B": "study_b_exploitation_prediction",
    "C": "study_c_model_comparison",
    "D": "study_d_explainability",
}


def _git_tags() -> dict:
    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": str(bool(git("status", "--porcelain")))}


def log_experiment(
    experiment_id: str,
    params: dict | None = None,
    metrics: dict | None = None,
    artifacts: list[Path] | None = None,
    tags: dict | None = None,
    run_name: str | None = None,
) -> str | None:
    """Log one run under its study's MLflow experiment. Returns the run id."""
    import mlflow

    study = experiment_id[0].upper()
    experiment_name = EXPERIMENT_NAMES[study]

    Path("experiments").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(experiment_name) is None:
        ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(
            experiment_name, artifact_location=(ARTIFACT_ROOT / experiment_name).resolve().as_uri()
        )
    mlflow.set_experiment(experiment_name)

    # MLflow rejects NaN/inf metrics on some backends; drop them explicitly and
    # say so rather than let the whole run fail.
    clean_metrics = {}
    for key, value in (metrics or {}).items():
        value = float(value)
        if math.isfinite(value):
            clean_metrics[key] = value
        else:
            logger.warning("%s: metric %s is %s, not logged", experiment_id, key, value)

    with mlflow.start_run(run_name=run_name or experiment_id) as run:
        mlflow.set_tags({"experiment": experiment_id, **_git_tags(), **(tags or {})})
        if params:
            # MLflow keys allow only [A-Za-z0-9_\-. :/]; e.g. "Risk & Compliance" is rejected.
            mlflow.log_params({re.sub(r"[^\w\-. :/]", "_", k): str(v)[:500] for k, v in params.items()})
        if clean_metrics:
            mlflow.log_metrics(clean_metrics)
        for path in artifacts or []:
            if Path(path).exists():
                mlflow.log_artifact(str(path))
            else:
                logger.warning("%s: artifact %s missing, not logged", experiment_id, path)
        logger.info("Logged %s to MLflow (run %s)", experiment_id, run.info.run_id)
        return run.info.run_id
