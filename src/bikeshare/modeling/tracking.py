"""MLflow setup and the lineage tags that make each run reproducible."""
import os
import subprocess
from pathlib import Path

import mlflow
import yaml

DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT = "bikeshare-demand"
DVC_LOCK = Path("dvc.lock")
# Files the pipeline itself rewrites while `dvc repro` runs. Changes to them do not
# mean the code or configuration differ from the commit.
PIPELINE_WRITTEN = (":(exclude)dvc.lock", ":(exclude)reports")


def start_tracking() -> None:
    """Point MLflow at the tracking server and fail clearly if it is down."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    try:
        mlflow.set_experiment(EXPERIMENT)
    except Exception as error:
        raise RuntimeError(f"MLflow tracking server unavailable at {uri}") from error


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def lineage_tags() -> dict[str, str]:
    """Git commit, whether code or configuration had uncommitted changes, and the DVC data version.

    The data version is the hash DVC recorded for the feature output in dvc.lock.
    """
    lock = yaml.safe_load(DVC_LOCK.read_text())
    features = next(out for out in lock["stages"]["features"]["outs"]
                    if out["path"] == "data/processed/features")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": str(bool(_git("status", "--porcelain", "--untracked-files=no",
                                      "--", ".", *PIPELINE_WRITTEN))),
        "dvc_features_md5": features["md5"],
    }
