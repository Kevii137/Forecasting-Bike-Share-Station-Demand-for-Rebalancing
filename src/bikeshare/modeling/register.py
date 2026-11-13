"""Register the latest trained model in the MLflow Model Registry.

The serving layer loads whichever version holds the champion alias (Decision D7).
This script only assigns that alias when no champion exists yet, so the first model
becomes champion. Replacing a champion is the promotion gate's job (Decision D9).
"""
import json
import logging
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from bikeshare.modeling.tracking import start_tracking

log = logging.getLogger(__name__)

MODEL_NAME = "bikeshare-demand"
CHAMPION = "champion"
METRICS_PATH = Path("reports/metrics.json")
COPIED_TAGS = ("git_commit", "git_dirty", "dvc_features_md5", "split_test")


def register(run_id: str, client: MlflowClient) -> str:
    """Register the run's model, copy its lineage tags, and return the new version."""
    run = client.get_run(run_id)
    model_id = run.outputs.model_outputs[0].model_id
    version = mlflow.register_model(f"models:/{model_id}", MODEL_NAME).version
    run_tags = run.data.tags
    for key in COPIED_TAGS:
        if key in run_tags:
            client.set_model_version_tag(MODEL_NAME, version, key, run_tags[key])
    return str(version)


def champion_version(client: MlflowClient) -> str | None:
    try:
        return client.get_model_version_by_alias(MODEL_NAME, CHAMPION).version
    except mlflow.exceptions.MlflowException:
        return None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    start_tracking()
    client = MlflowClient()
    run_id = json.loads(METRICS_PATH.read_text())["lightgbm_run_id"]

    existing = [version for version in client.search_model_versions(f"name='{MODEL_NAME}'")
                if version.run_id == run_id]
    version = existing[0].version if existing else register(run_id, client)
    log.info("Run %s is %s version %s", run_id, MODEL_NAME, version)

    current = champion_version(client)
    if current is None:
        client.set_registered_model_alias(MODEL_NAME, CHAMPION, version)
        log.info("No champion existed, so version %s is now champion", version)
    else:
        log.info("Champion stays at version %s; promotion decides any change", current)


if __name__ == "__main__":
    main()
