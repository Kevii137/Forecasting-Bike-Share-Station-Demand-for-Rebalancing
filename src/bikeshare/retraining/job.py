"""Retrain a challenger on the data observed up to a simulated time, gate it, promote it.

The trainer service runs this in a subprocess, so the memory a job uses is returned when
it ends and a crash cannot take the service down. It can also be run by hand:

    PYTHONPATH=src python -m bikeshare.retraining.job --as-of 2026-03-10T23:00
    PYTHONPATH=src python -m bikeshare.retraining.job --as-of 2026-03-10T23:00 --dry-run

Hyperparameters are the ones Ray Tune chose (reports/best_params.json). Milestone 2 found
all trials within 0.5% of each other, so a new search per retrain would cost far more
than it could gain. A dry run trains and gates but writes nothing to MLflow.
"""
import argparse
import json
import logging
import tempfile
import time
from pathlib import Path

import mlflow
import mlflow.pyfunc  # explicit: mlflow exposes pyfunc through a lazy proxy
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from bikeshare.config import load_params
from bikeshare.modeling.baselines import BASELINES, baseline_forecast
from bikeshare.modeling.boosting import fit
from bikeshare.modeling.dataset import FEATURE_COLUMNS, FEATURES_PATH, TARGETS
from bikeshare.modeling.evaluation import evaluate, flatten
from bikeshare.modeling.packaging import log_model
from bikeshare.modeling.register import MODEL_NAME, register
from bikeshare.modeling.tracking import lineage_tags, start_tracking
from bikeshare.monitoring.drift import build_profile
from bikeshare.monitoring.profile import ARTIFACT_DIR, PROFILE_PATH
from bikeshare.retraining import registry
from bikeshare.retraining.gate import compare
from bikeshare.retraining.windows import HOUR, Windows, load_rows, plan

log = logging.getLogger(__name__)

BEST_PARAMS_PATH = Path("reports/best_params.json")
CHALLENGER, CHAMPION = "challenger", "champion"
STAGE = "retrain"


def train_challenger(windows: Windows, params: dict, best: dict, features_path: str,
                     max_rounds: int | None = None) -> tuple[dict, dict, pd.DataFrame]:
    """Fit one booster per target. Returns the boosters, row counts and a profile sample."""
    model = params["model"]
    train = load_rows(windows.train_start, windows.stop_start - HOUR, path=features_path)
    stop = load_rows(windows.stop_start, windows.gate_start - HOUR, path=features_path)
    log.info("Fitting on %d rows, early stopping on %d", len(train), len(stop))
    boosters = {}
    for demand, target in TARGETS.items():
        boosters[demand] = fit(
            best[demand]["config"], (train[FEATURE_COLUMNS], train[target]),
            (stop[FEATURE_COLUMNS], stop[target]), threads=params["retraining"]["threads"],
            seed=params["seed"], max_rounds=max_rounds or model["max_rounds"],
            early_stopping_rounds=model["early_stopping_rounds"],
        )
        log.info("%s: stopped at round %d", demand, boosters[demand].best_iteration)
    rows = params["monitoring"]["profile_sample_rows"]
    sample = train.sample(n=min(rows, len(train)), random_state=params["seed"])
    return boosters, {"train_rows": len(train), "early_stopping_rows": len(stop)}, sample


def predict(boosters: dict, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {demand: np.clip(booster.predict(frame[FEATURE_COLUMNS],
                                            num_iteration=booster.best_iteration), 0, None)
            for demand, booster in boosters.items()}


def score_gate(gate: pd.DataFrame, boosters: dict, champion, evaluation: dict) -> dict:
    """Score the challenger, the champion and both baselines on the same gate rows."""
    needed = [f"{demand}_{column}" for demand in TARGETS for column in BASELINES.values()]
    scored = gate.dropna(subset=needed).reset_index(drop=True)
    for demand, values in predict(boosters, scored).items():
        scored[f"{demand}_{CHALLENGER}"] = values
    theirs = champion.predict(scored[FEATURE_COLUMNS].astype("float64"))
    for demand in TARGETS:
        scored[f"{demand}_{CHAMPION}"] = theirs[f"{demand}_forecast"].to_numpy()
        for baseline in BASELINES:
            scored[f"{demand}_{baseline}"] = baseline_forecast(scored, baseline, demand)
    scores = {name: evaluate(scored, {demand: f"{demand}_{name}" for demand in TARGETS},
                             evaluation)
              for name in (CHALLENGER, CHAMPION, *BASELINES)}
    return {"rows": len(scored), "scores": scores}


def profile_of(boosters: dict, sample: pd.DataFrame, params: dict, windows: Windows) -> dict:
    """The drift reference: the challenger's training rows and its forecasts on them."""
    frame = sample.assign(**{f"{demand}_forecast": values
                             for demand, values in predict(boosters, sample).items()})
    monitoring = params["monitoring"]
    return build_profile(frame, monitoring["drift_columns"], monitoring["bins"],
                         window=windows.describe()["train"])


def summary_of(gate: dict) -> dict:
    return {name: {f"{demand}_wape": round(scores[demand]["overall"]["wape"], 4)
                   for demand in TARGETS}
            for name, scores in gate["scores"].items()}


def run(as_of: pd.Timestamp, trigger: str = "manual", dry_run: bool = False,
        params: dict | None = None, features_path: str = FEATURES_PATH,
        best_params_path: Path = BEST_PARAMS_PATH, max_rounds: int | None = None,
        client: MlflowClient | None = None) -> dict:
    started = time.monotonic()
    params = params or load_params()
    promotion, retraining = params["promotion"], params["retraining"]
    windows = plan(as_of, params["split"]["train"][0], promotion["gate_days"],
                   retraining["early_stopping_days"])
    log.info("Retraining as of %s: %s", windows.as_of, windows.describe())
    client = client or MlflowClient()
    champion_version = registry.champion_version(client)
    if champion_version is None:
        raise registry.RegistryError("no champion to compare against")
    champion = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{champion_version}")

    best = json.loads(Path(best_params_path).read_text())
    boosters, rows, sample = train_challenger(windows, params, best, features_path, max_rounds)
    gate_rows = load_rows(windows.gate_start, windows.as_of, keys=True, path=features_path)
    gate = score_gate(gate_rows, boosters, champion, params["evaluation"])
    decision = compare(gate["scores"][CHALLENGER], gate["scores"][CHAMPION],
                       {name: gate["scores"][name] for name in BASELINES},
                       promotion["tier_tolerance"])
    result = {
        "as_of": f"{windows.as_of:%Y-%m-%dT%H:%M}", "trigger": trigger, "dry_run": dry_run,
        "windows": windows.describe(), **rows, "gate_rows": gate["rows"],
        "rounds": {demand: booster.best_iteration for demand, booster in boosters.items()},
        "champion_version": champion_version, "summary": summary_of(gate),
        "gate": decision, "run_id": None, "version": None,
    }
    log.info("Gate %s: %s", "passed" if decision["passed"] else "failed", result["summary"])
    if dry_run:
        result["outcome"] = "dry_run"
    else:
        result.update(record(result, boosters, gate, sample, params, windows, client))
    result["seconds"] = round(time.monotonic() - started, 1)
    return result


def record(result: dict, boosters: dict, gate: dict, sample: pd.DataFrame, params: dict,
           windows: Windows, client: MlflowClient) -> dict:
    """Log the challenger to MLflow, register it, and promote it if the gate passed."""
    passed = result["gate"]["passed"]
    tags = {**lineage_tags(), "stage": STAGE, "model_type": "lightgbm",
            "trigger": result["trigger"], "as_of": result["as_of"],
            "champion_version": result["champion_version"],
            "split_test": windows.describe()["gate"], "gate": "passed" if passed else "failed"}
    with mlflow.start_run(run_name=STAGE) as run, tempfile.TemporaryDirectory() as scratch:
        mlflow.set_tags(tags)
        mlflow.log_params({**{f"window_{name}": value for name, value in result["windows"].items()},
                           "train_rows": result["train_rows"],
                           "early_stopping_rows": result["early_stopping_rows"],
                           "gate_rows": result["gate_rows"],
                           **{f"{demand}_rounds": rounds
                              for demand, rounds in result["rounds"].items()}})
        for name, scores in gate["scores"].items():
            for demand in TARGETS:
                mlflow.log_metrics(flatten(f"gate_{name}_{demand}", scores[demand]))
        mlflow.log_dict(result["gate"], "gate.json")
        profile = {**profile_of(boosters, sample, params, windows), "run_id": run.info.run_id}
        profile_path = Path(scratch) / PROFILE_PATH.name
        profile_path.write_text(json.dumps(profile, indent=2))
        mlflow.log_artifact(str(profile_path), ARTIFACT_DIR)
        log_model(boosters, sample, Path(scratch) / "model")

    version = register(run.info.run_id, client)
    for key in ("trigger", "as_of", "gate", "champion_version"):
        client.set_model_version_tag(MODEL_NAME, version, key, tags[key])
    outcome = "rejected"
    if passed:
        try:
            registry.promote(client, version, result["champion_version"])
            outcome = "promoted"
        except registry.RegistryError as error:
            log.warning("Not promoted: %s", error)
            outcome = "stale"
    client.set_tag(run.info.run_id, "outcome", outcome)
    client.set_tag(run.info.run_id, "version", version)
    log.info("Version %s %s", version, outcome)
    return {"run_id": run.info.run_id, "version": version, "outcome": outcome}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--as-of", required=True, help="last observed hour, e.g. 2026-03-10T23:00")
    parser.add_argument("--trigger", default="manual")
    parser.add_argument("--dry-run", action="store_true", help="train and gate, write nothing")
    parser.add_argument("--result", type=Path, help="write the result as JSON to this file")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    start_tracking()
    result = run(pd.Timestamp(args.as_of), args.trigger, args.dry_run)
    text = json.dumps(result, indent=2, default=str)
    if args.result:
        args.result.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
