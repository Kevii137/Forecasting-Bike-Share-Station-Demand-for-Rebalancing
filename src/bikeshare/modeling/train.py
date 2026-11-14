"""Train the final models, score them against both baselines and log one MLflow model."""
import json
import logging
from pathlib import Path

import lightgbm as lgb
import matplotlib
import mlflow
import numpy as np
import pandas as pd

from bikeshare.config import load_params
from bikeshare.modeling.baselines import BASELINES, baseline_forecast
from bikeshare.modeling.boosting import fit
from bikeshare.modeling.dataset import FEATURE_COLUMNS, TARGETS, load_split
from bikeshare.modeling.evaluation import evaluate, flatten
from bikeshare.modeling.packaging import log_model
from bikeshare.modeling.tracking import lineage_tags, start_tracking

# The non-interactive backend must be selected before pyplot is imported.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

BEST_PARAMS_PATH = Path("reports/best_params.json")
MODEL_DIR = Path("models")
METRICS_PATH = Path("reports/metrics.json")
IMPORTANCE_PATH = Path("reports/importance.json")
IMPORTANCE_PLOT = Path("reports/importance.png")
SHAP_SAMPLE_ROWS = 50_000
MODEL_NAME = "lightgbm"


def importance(boosters: dict[str, lgb.Booster], sample: pd.DataFrame) -> dict:
    """Gain importance and mean absolute SHAP value per feature, for each target."""
    result = {}
    for demand, booster in boosters.items():
        contributions = booster.predict(sample[FEATURE_COLUMNS], pred_contrib=True)[:, :-1]
        result[demand] = {
            "gain": dict(zip(FEATURE_COLUMNS, booster.feature_importance("gain").tolist())),
            "mean_abs_shap": dict(zip(FEATURE_COLUMNS, np.abs(contributions).mean(axis=0).tolist())),
        }
    return result


def plot_importance(scores: dict) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=False)
    for axis, (demand, values) in zip(axes, scores.items()):
        shap = pd.Series(values["mean_abs_shap"]).sort_values().tail(12)
        axis.barh(shap.index, shap.values, color="#3b6ea5")
        axis.set_title(f"{demand.capitalize()} model")
        axis.set_xlabel("Mean |SHAP value| (log trips, linear scale)")
    figure.suptitle("Top 12 features by mean absolute SHAP value, February 2026 test rows")
    figure.tight_layout()
    figure.savefig(IMPORTANCE_PLOT, dpi=120)
    plt.close(figure)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    params = load_params()
    model, evaluation = params["model"], params["evaluation"]
    best = json.loads(BEST_PARAMS_PATH.read_text())
    start_tracking()

    train = load_split(params["split"]["train"])
    validation = load_split(params["split"]["validation"])
    test = load_split(params["split"]["test"])
    baseline_columns = [f"{demand}_{column}" for demand in TARGETS for column in BASELINES.values()]
    scored = test.dropna(subset=baseline_columns).reset_index(drop=True)
    log.info("Scoring %d of %d test rows that have every baseline", len(scored), len(test))

    tags = {**lineage_tags(), "split_test": "-".join(params["split"]["test"])}
    report = {"test_rows": len(scored), "test_rows_total": len(test), "models": {}}

    with mlflow.start_run(run_name=MODEL_NAME) as run:
        mlflow.set_tags({**tags, "stage": "train", "model_type": MODEL_NAME})
        mlflow.log_params({"train_rows": len(train), "horizons": params["forecast"]["horizons"],
                           "train_months": "-".join(params["split"]["train"]),
                           "feature_count": len(FEATURE_COLUMNS)})
        boosters = {}
        for demand, target in TARGETS.items():
            config = best[demand]["config"]
            booster = fit(
                config, (train[FEATURE_COLUMNS], train[target]),
                (validation[FEATURE_COLUMNS], validation[target]),
                threads=model["final_threads"], seed=params["seed"],
                max_rounds=model["max_rounds"], early_stopping_rounds=model["early_stopping_rounds"],
            )
            boosters[demand] = booster
            mlflow.log_params({f"{demand}_{key}": value for key, value in config.items()})
            mlflow.log_param(f"{demand}_rounds", booster.best_iteration)
            scored[f"{demand}_{MODEL_NAME}"] = np.clip(
                booster.predict(scored[FEATURE_COLUMNS], num_iteration=booster.best_iteration), 0, None
            )

        report["models"][MODEL_NAME] = evaluate(
            scored, {demand: f"{demand}_{MODEL_NAME}" for demand in TARGETS}, evaluation)
        for demand in TARGETS:
            mlflow.log_metrics(flatten(f"test_{demand}", report["models"][MODEL_NAME][demand]))
        mlflow.log_metrics({f"test_net_flow_{key}": value for key, value
                            in report["models"][MODEL_NAME]["net_flow"].items()})

        sample = scored.sample(n=min(SHAP_SAMPLE_ROWS, len(scored)), random_state=params["seed"])
        scores = importance(boosters, sample)
        IMPORTANCE_PATH.write_text(json.dumps(scores, indent=2))
        plot_importance(scores)
        mlflow.log_artifact(str(IMPORTANCE_PATH))
        mlflow.log_artifact(str(IMPORTANCE_PLOT))
        log_model(boosters, scored, MODEL_DIR)
        report["lightgbm_run_id"] = run.info.run_id

    for baseline in BASELINES:
        for demand in TARGETS:
            scored[f"{demand}_{baseline}"] = baseline_forecast(scored, baseline, demand)
        report["models"][baseline] = evaluate(
            scored, {demand: f"{demand}_{baseline}" for demand in TARGETS}, evaluation)
        with mlflow.start_run(run_name=baseline):
            mlflow.set_tags({**tags, "stage": "baseline", "model_type": baseline})
            for demand in TARGETS:
                mlflow.log_metrics(flatten(f"test_{demand}", report["models"][baseline][demand]))
            mlflow.log_metrics({f"test_net_flow_{key}": value for key, value
                                in report["models"][baseline]["net_flow"].items()})

    report["summary"] = {
        name: {**{f"{demand}_wape": round(scores[demand]["overall"]["wape"], 4) for demand in TARGETS},
               "net_flow_mae": round(scores["net_flow"]["mae"], 4)}
        for name, scores in report["models"].items()
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2))
    log.info("Summary: %s", report["summary"])


if __name__ == "__main__":
    main()
