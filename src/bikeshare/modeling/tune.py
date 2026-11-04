"""Parallel hyperparameter search with Ray Tune, one search per target.

Each trial trains LightGBM on a sample of the training months, stops early on the
validation month, and reports validation WAPE. Every trial is logged to MLflow.
"""
import json
import logging
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import ray
from ray import tune
from ray.tune.search.basic_variant import BasicVariantGenerator

from bikeshare.config import load_params
from bikeshare.modeling.dataset import FEATURE_COLUMNS, TARGETS, load_split
from bikeshare.modeling.metrics import poisson_deviance, wape
from bikeshare.modeling.tracking import lineage_tags, start_tracking

log = logging.getLogger(__name__)

OUTPUT_PATH = Path("reports/best_params.json")
RAY_RESULTS = Path("ray_results").resolve()
SEARCH_SPACE = {
    "learning_rate": tune.loguniform(0.02, 0.2),
    "num_leaves": tune.choice([31, 63, 127, 255]),
    "min_data_in_leaf": tune.choice([50, 200, 500, 1000]),
    "feature_fraction": tune.uniform(0.6, 1.0),
    "bagging_fraction": tune.uniform(0.6, 1.0),
    "lambda_l2": tune.loguniform(1e-3, 10.0),
}


def lightgbm_params(config: dict, threads: int, seed: int) -> dict:
    """Fixed settings plus the tuned ones. deterministic makes reruns repeatable."""
    return {
        "objective": "poisson", "metric": "poisson", "bagging_freq": 1,
        "num_threads": threads, "seed": seed, "deterministic": True,
        "force_row_wise": True, "verbosity": -1, **config,
    }


def fit(config: dict, train: tuple, validation: tuple, threads: int, seed: int,
        max_rounds: int, early_stopping_rounds: int) -> lgb.Booster:
    train_set = lgb.Dataset(train[0], train[1], feature_name=FEATURE_COLUMNS)
    valid_set = lgb.Dataset(validation[0], validation[1], reference=train_set)
    return lgb.train(
        lightgbm_params(config, threads, seed), train_set, num_boost_round=max_rounds,
        valid_sets=[valid_set], callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )


def trial(config: dict, train: tuple, validation: tuple, settings: dict) -> dict:
    booster = fit(config, train, validation, **settings)
    predicted = booster.predict(validation[0], num_iteration=booster.best_iteration)
    return {
        "val_wape": wape(validation[1], predicted),
        "val_poisson_deviance": poisson_deviance(validation[1], predicted),
        "best_iteration": booster.best_iteration,
    }


def arrays(frame, target: str) -> tuple[np.ndarray, np.ndarray]:
    return frame[FEATURE_COLUMNS].to_numpy("float32"), frame[target].to_numpy("float32")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    params = load_params()
    model = params["model"]
    start_tracking()

    train = load_split(params["split"]["train"])
    train = train.sample(n=min(model["tuning_sample_rows"], len(train)), random_state=params["seed"])
    validation = load_split(params["split"]["validation"])
    settings = {
        "threads": model["threads_per_trial"], "seed": params["seed"],
        "max_rounds": model["max_rounds"], "early_stopping_rounds": model["early_stopping_rounds"],
    }

    ray.init(
        num_cpus=model["threads_per_trial"] * model["max_concurrent_trials"],
        object_store_memory=model["ray_object_store_bytes"],
        include_dashboard=False,
    )
    best = {}
    for demand, target in TARGETS.items():
        trainable = tune.with_parameters(
            trial, train=arrays(train, target), validation=arrays(validation, target),
            settings=settings,
        )
        tuner = tune.Tuner(
            tune.with_resources(trainable, {"cpu": model["threads_per_trial"]}),
            param_space=SEARCH_SPACE,
            tune_config=tune.TuneConfig(
                metric="val_poisson_deviance", mode="min", num_samples=model["num_samples"],
                max_concurrent_trials=model["max_concurrent_trials"],
                search_alg=BasicVariantGenerator(random_state=params["seed"]),
            ),
            run_config=tune.RunConfig(name=f"tune-{demand}", storage_path=str(RAY_RESULTS), verbose=0),
        )
        results = tuner.fit()
        if results.errors:
            raise RuntimeError(f"{len(results.errors)} {demand} trials failed: {results.errors[0]}")

        with mlflow.start_run(run_name=f"tune-{demand}"):
            mlflow.set_tags({**lineage_tags(), "stage": "tune", "target": demand})
            mlflow.log_params({"tuning_rows": len(train), "num_samples": model["num_samples"],
                               "search_space": json.dumps(str(SEARCH_SPACE))[:500]})
            for index, result in enumerate(results):
                with mlflow.start_run(run_name=f"{demand}-trial-{index}", nested=True):
                    mlflow.set_tags({**lineage_tags(), "stage": "tune", "target": demand})
                    mlflow.log_params(result.config)
                    mlflow.log_metrics({key: result.metrics[key] for key in
                                        ("val_wape", "val_poisson_deviance", "best_iteration")})
            winner = results.get_best_result()
            mlflow.log_params({f"best_{key}": value for key, value in winner.config.items()})
            mlflow.log_metrics({f"best_{key}": winner.metrics[key]
                                for key in ("val_poisson_deviance", "val_wape")})

        best[demand] = {"config": winner.config,
                        **{key: winner.metrics[key] for key in
                           ("val_poisson_deviance", "val_wape", "best_iteration")}}
        log.info("Best %s: %s", demand, best[demand])

    ray.shutdown()
    OUTPUT_PATH.write_text(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
