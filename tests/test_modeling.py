"""Tests for metrics, baselines and the MLflow model wrapper."""
import json
import subprocess

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import pytest

from bikeshare.modeling.baselines import baseline_forecast
from bikeshare.modeling.dataset import FEATURE_COLUMNS
from bikeshare.modeling.metrics import (
    net_flow_error,
    poisson_deviance,
    volume_tier,
    wape,
)
from bikeshare.modeling.pyfunc import DemandModel


def test_wape_is_total_error_over_total_demand():
    assert wape(np.array([2.0, 0.0, 2.0]), np.array([1.0, 1.0, 2.0])) == pytest.approx(0.5)


def test_poisson_deviance_is_zero_for_a_perfect_forecast():
    actual = np.array([0.0, 1.0, 4.0])
    assert poisson_deviance(actual, actual) == pytest.approx(0.0, abs=1e-6)
    assert poisson_deviance(actual, actual + 1) > 0


def test_poisson_deviance_handles_zero_forecasts_without_warnings():
    with np.errstate(all="raise"):
        assert np.isfinite(poisson_deviance(np.array([0.0, 3.0]), np.array([0.0, 0.0])))


def test_volume_tiers_are_half_open():
    tiers = volume_tier(pd.Series([5, 9.9, 10, 25]), [5, 10, 20])
    assert list(tiers) == ["5-10", "5-10", "10-20", "20+"]


def test_net_flow_sums_all_horizons_and_skips_incomplete_origins():
    frame = pd.DataFrame({
        "station_key": ["A"] * 4, "origin_hour": [1, 1, 1, 2], "horizon": [1, 2, 3, 1],
        "pickups_target": [1, 1, 1, 0], "dropoffs_target": [3, 3, 3, 0],
        "pickups_forecast": [1, 1, 1, 0], "dropoffs_forecast": [2, 2, 2, 0],
    })
    result = net_flow_error(frame, "pickups_forecast", "dropoffs_forecast")
    assert result == {"mae": 3.0, "rmse": 3.0, "origins": 1}


def test_baselines_read_the_seasonal_features():
    frame = pd.DataFrame({"pickups_same_hour_1w": [4.0], "pickups_same_hour_8w_mean": [2.5]})
    assert baseline_forecast(frame, "seasonal_naive", "pickups").iloc[0] == 4.0
    assert baseline_forecast(frame, "historical_average", "pickups").iloc[0] == 2.5


@pytest.fixture
def logged_model(tmp_path):
    rng = np.random.default_rng(0)
    features = pd.DataFrame(rng.random((200, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    artifacts = {}
    for demand in ("pickups", "dropoffs"):
        booster = lgb.train({"objective": "poisson", "verbosity": -1},
                            lgb.Dataset(features, rng.poisson(2, 200)), num_boost_round=5)
        path = tmp_path / f"{demand}.txt"
        booster.save_model(path)
        artifacts[demand] = str(path)
    columns = tmp_path / "feature_columns.json"
    columns.write_text(json.dumps(FEATURE_COLUMNS))
    artifacts["feature_columns"] = str(columns)

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    with mlflow.start_run():
        info = mlflow.pyfunc.log_model(name="model", python_model=DemandModel(),
                                       artifacts=artifacts, code_paths=["src/bikeshare"])
    return mlflow.pyfunc.load_model(info.model_uri), features


def test_pyfunc_returns_non_negative_forecasts_for_both_targets(logged_model):
    model, features = logged_model
    result = model.predict(features.head(10))
    assert list(result.columns) == ["pickups_forecast", "dropoffs_forecast"]
    assert (result >= 0).all().all()


def test_pyfunc_rejects_missing_feature_columns(logged_model):
    model, features = logged_model
    with pytest.raises(ValueError, match="missing feature columns"):
        model.predict(features.drop(columns=["capacity"]).head(2))


def test_lineage_ignores_files_the_pipeline_rewrites(tmp_path, monkeypatch):
    from bikeshare.modeling.tracking import lineage_tags

    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "metrics.json").write_text("{}")
    (tmp_path / "code.py").write_text("x = 1\n")
    (tmp_path / "dvc.lock").write_text(
        "stages:\n  features:\n    outs:\n    - path: data/processed/features\n      md5: abc\n")
    git("init", "-q")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    monkeypatch.chdir(tmp_path)

    (tmp_path / "reports" / "metrics.json").write_text('{"a": 1}')
    (tmp_path / "dvc.lock").write_text((tmp_path / "dvc.lock").read_text() + "# rerun\n")
    assert lineage_tags()["git_dirty"] == "False"
    (tmp_path / "code.py").write_text("x = 2\n")
    assert lineage_tags()["git_dirty"] == "True"
