"""Smoke tests proving the pinned stack works together on the paths this project depends on."""
import numpy as np
import pandas as pd


def test_spark_apply_in_pandas():
    """Decision D6 relies on running one pandas feature function through Spark."""
    from bikeshare.spark import get_spark

    spark = get_spark("smoke-test", master="local[2]", driver_memory="1g")
    frame = spark.createDataFrame(
        pd.DataFrame({"station": ["a", "a", "b"], "pickups": [1, 2, 5]})
    )

    def add_rolling_sum(group: pd.DataFrame) -> pd.DataFrame:
        return group.assign(total=group["pickups"].sum())

    result = (
        frame.groupBy("station")
        .applyInPandas(add_rolling_sum, "station string, pickups long, total long")
        .toPandas()
    )
    spark.stop()
    assert dict(zip(result["station"], result["total"])) == {"a": 3, "b": 5}


def test_spark_preserves_hourly_timestamps_through_pandas():
    """Hourly station data must cross the Spark-pandas boundary unchanged, including DST days."""
    from bikeshare.spark import get_spark

    spark = get_spark("smoke-test", master="local[2]", driver_memory="1g")
    hours = pd.to_datetime(["2026-03-08 01:00", "2026-03-08 03:00", "2026-11-01 01:00"])
    frame = spark.createDataFrame(
        pd.DataFrame({"station": ["a", "a", "a"], "hour": hours.astype("datetime64[ns]")})
    )

    def next_hour(group: pd.DataFrame) -> pd.DataFrame:
        return group.assign(hour=group["hour"] + pd.Timedelta(hours=1))

    result = (
        frame.groupBy("station")
        .applyInPandas(next_hour, "station string, hour timestamp")
        .toPandas()
        .sort_values("hour")
    )
    spark.stop()
    assert list(result["hour"].dt.strftime("%m-%d %H:%M")) == [
        "03-08 02:00",
        "03-08 04:00",
        "11-01 02:00",
    ]


def test_lightgbm_poisson_is_non_negative():
    import lightgbm as lgb

    rng = np.random.default_rng(0)
    features = rng.normal(size=(500, 3))
    target = rng.poisson(np.exp(features[:, 0]))
    model = lgb.LGBMRegressor(objective="poisson", n_estimators=10, verbosity=-1)
    model.fit(features, target)
    assert (model.predict(features) >= 0).all()


def test_ray_tune_runs_parallel_trials():
    import ray
    from ray import tune

    def objective(config):
        return {"score": config["x"] ** 2}

    ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True)
    tuner = tune.Tuner(
        objective,
        param_space={"x": tune.grid_search([1, 2])},
        tune_config=tune.TuneConfig(metric="score", mode="min"),
    )
    best = tuner.fit().get_best_result()
    ray.shutdown()
    assert best.config["x"] == 1


def test_mlflow_logs_to_sqlite(tmp_path):
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    with mlflow.start_run() as run:
        mlflow.log_metric("wape", 0.5)
    stored = mlflow.get_run(run.info.run_id)
    assert stored.data.metrics["wape"] == 0.5
