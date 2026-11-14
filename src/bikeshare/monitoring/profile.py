"""Build the drift reference profile for the trained model and attach it to its MLflow run.

The reference is the training window: the inputs the model learned from and the
forecasts it makes on them. The profile is logged next to the model, so the service
always compares against the profile of the model it is serving.

    PYTHONPATH=src python -m bikeshare.monitoring.profile
"""
import json
import logging
from pathlib import Path

import mlflow
import mlflow.pyfunc  # explicit: mlflow exposes pyfunc through a lazy proxy
import pandas as pd
from mlflow import MlflowClient

from bikeshare.config import load_params
from bikeshare.modeling.dataset import FEATURE_COLUMNS, load_split
from bikeshare.modeling.tracking import start_tracking
from bikeshare.monitoring.drift import build_profile

log = logging.getLogger(__name__)

METRICS_PATH = Path("reports/metrics.json")
PROFILE_PATH = Path("reports/reference_profile.json")
ARTIFACT_DIR = "monitoring"
ARTIFACT_PATH = f"{ARTIFACT_DIR}/{PROFILE_PATH.name}"


def logged_model_uri(run_id: str, client: MlflowClient) -> str:
    return f"models:/{client.get_run(run_id).outputs.model_outputs[0].model_id}"


def with_forecasts(frame: pd.DataFrame, model) -> pd.DataFrame:
    predicted = model.predict(frame[FEATURE_COLUMNS].astype("float64"))
    return frame.assign(**{column: predicted[column].to_numpy() for column in predicted})


def reference_profile(model, params: dict) -> dict:
    monitoring = params["monitoring"]
    train = load_split(params["split"]["train"])
    sample = train.sample(n=min(monitoring["profile_sample_rows"], len(train)),
                          random_state=params["seed"])
    return build_profile(with_forecasts(sample, model), monitoring["drift_columns"],
                         monitoring["bins"], window="-".join(params["split"]["train"]))


def load_profile(run_id: str) -> dict | None:
    """The profile logged with a run, or None when the run has none."""
    try:
        path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=ARTIFACT_PATH)
    except Exception as error:  # noqa: BLE001 - MLflow raises several types for a missing file
        log.warning("No reference profile for run %s: %s", run_id, error)
        return None
    return json.loads(Path(path).read_text())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    params = load_params()
    start_tracking()
    client = MlflowClient()
    run_id = json.loads(METRICS_PATH.read_text())["lightgbm_run_id"]
    model = mlflow.pyfunc.load_model(logged_model_uri(run_id, client))

    profile = {**reference_profile(model, params), "run_id": run_id}
    PROFILE_PATH.write_text(json.dumps(profile, indent=2))
    client.log_artifact(run_id, str(PROFILE_PATH), ARTIFACT_DIR)
    log.info("Logged a %d-row reference profile of %d columns to run %s",
             profile["rows"], len(profile["columns"]), run_id)


if __name__ == "__main__":
    main()
