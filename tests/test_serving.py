"""Serving tests on a small synthetic panel, plus online/offline parity on real data."""
from pathlib import Path

import mlflow.pyfunc
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from helpers import ORIGIN, RULES, START, ConstantModel, FakeClient

from bikeshare.features import build_features
from bikeshare.modeling.dataset import FEATURE_COLUMNS
from bikeshare.serving.app import Settings, create_app
from bikeshare.serving.forecast import (
    FALLBACK,
    MODEL,
    alerts,
    fallback_forecast,
    forecast_rows,
    station_summary,
)
from bikeshare.serving.model_store import ModelStore
from bikeshare.serving.online import FeatureStore


def test_online_features_equal_build_features_on_full_history(store, data_dir):
    online = store.features_at(ORIGIN)
    panel = pd.read_parquet(data_dir / "panel")
    history = panel[(panel["station_key"] == "busy") & (panel["hour"] <= ORIGIN)].drop(
        columns="month")
    expected = build_features(history, [1, 2, 3])
    expected = expected[expected["origin_hour"] == ORIGIN].set_index("horizon")
    got = online[online["station_key"] == "busy"].set_index("horizon")
    for column in FEATURE_COLUMNS[1:-3]:  # demand and calendar columns, before station details
        np.testing.assert_allclose(got[column], expected[column], rtol=1e-6)


def test_origin_must_be_on_the_hour_and_inside_the_window(store):
    with pytest.raises(ValueError, match="on the hour"):
        store.check_origin(ORIGIN + pd.Timedelta(minutes=30))
    with pytest.raises(ValueError, match="between"):
        store.check_origin(START)


def test_routing_fallback_and_alerts(store):
    features = store.features_at(ORIGIN)
    rows = forecast_rows(features, ConstantModel(), RULES)
    sources = rows.groupby("station_key")["source"].first()
    assert sources.to_dict() == {"busy": MODEL, "quiet": FALLBACK, "light": FALLBACK}

    summary = station_summary(features, rows).set_index("station_key")
    assert summary.loc["busy", "net_flow"] == pytest.approx(-9.0)  # 3 horizons x (1 - 4)
    assert summary.loc["busy", "pressure"] == pytest.approx(-9.0 / 15)
    assert np.isnan(summary.loc["light", "pressure"])  # no capacity, no pressure

    ranked = alerts(summary.reset_index(), RULES)
    assert list(ranked["draining"]["station_key"]) == ["busy"]
    assert ranked["filling"].empty


def test_fallback_uses_shorter_averages_for_new_stations():
    features = pd.DataFrame({
        "pickups_same_hour_8w_mean": [2.0, np.nan, np.nan, np.nan],
        "pickups_same_hour_4w_mean": [9.0, 3.0, np.nan, np.nan],
        "pickups_mean_168h": [9.0, 9.0, 0.5, np.nan],
    })
    assert fallback_forecast(features, "pickups").tolist() == [2.0, 3.0, 0.5, 0.0]


def test_model_store_reloads_only_when_alias_moves(monkeypatch):
    loads = []
    monkeypatch.setattr(mlflow.pyfunc, "load_model",
                        lambda uri: loads.append(uri) or ConstantModel())
    client = FakeClient()
    models = ModelStore("demand", "champion", client)
    assert models.refresh(), models.last_error
    assert models.current.version == "1"
    assert not models.refresh()
    client.version = "2"
    assert models.refresh() and models.current.version == "2"
    client.fail = True
    assert not models.refresh()
    assert models.current.version == "2" and "registry down" in models.last_error
    assert loads == ["models:/demand/1", "models:/demand/2"]


def test_model_load_does_not_leave_bundled_code_on_the_path(monkeypatch):
    import sys

    def load(uri):
        sys.path.insert(0, "/tmp/model-bundle/code")
        return ConstantModel()

    monkeypatch.setattr(mlflow.pyfunc, "load_model", load)
    before = list(sys.path)
    ModelStore("demand", "champion", FakeClient()).refresh()
    assert sys.path == before


@pytest.fixture
def client(data_dir, store, monkeypatch):
    monkeypatch.setattr(mlflow.pyfunc, "load_model", lambda uri: ConstantModel())
    settings = Settings(params_path=str(data_dir / "params.yaml"),
                        panel_path=str(data_dir / "panel"),
                        stations_path=str(data_dir / "stations.parquet"),
                        reports_dir="reports", default_origin=f"{ORIGIN:%Y-%m-%dT%H:%M}")
    models = ModelStore("demand", "champion", FakeClient())
    store.close = lambda: None  # the module fixture owns the pool
    app = create_app(settings, features=store, models=models)
    with TestClient(app) as test_client:
        yield test_client


def test_api_serves_alerts_station_detail_and_metrics(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json()["ready"] is True

    body = client.get("/alerts").json()
    assert body["origin"] == f"{ORIGIN:%Y-%m-%dT%H:%M}"
    assert [row["station_key"] for row in body["draining"]] == ["busy"]
    assert body["stations_on_model"] == 1

    detail = client.get("/station", params={"key": "busy", "history_hours": 6}).json()
    assert len(detail["history"]) == 6 and len(detail["observed_after_origin"]) == 3
    assert [row["horizon"] for row in detail["forecast"]] == [1, 2, 3]

    forecast = client.get("/forecast").json()
    assert {row["station_key"] for row in forecast["stations"]} == {"busy", "quiet", "light"}
    assert 'bikeshare_requests_total{endpoint="/alerts",status="200"}' in client.get(
        "/metrics").text


def test_api_rejects_bad_origins_and_unknown_stations(client):
    assert client.get("/alerts", params={"origin": "2020-01-01T08:00"}).status_code == 422
    assert client.get("/alerts", params={"origin": "not a time"}).status_code == 422
    assert client.get("/station", params={"key": "nowhere"}).status_code == 404


@pytest.mark.skipif(not Path("data/processed/features").exists(), reason="needs DVC data")
def test_online_features_match_offline_spark_features():
    """Decision D6 parity: the service must rebuild the exact model inputs Spark wrote."""
    day = pd.Timestamp("2026-07-04")  # a holiday, to cover the calendar features
    store = FeatureStore("data/processed/panel", "data/interim/stations.parquet",
                         [1, 2, 3], workers=4)
    try:
        online = store.day_features(day)
    finally:
        store.close()
    offline = pd.read_parquet("data/processed/features", filters=[("month", "=", "2026-07")])
    offline = offline[offline["origin_hour"].dt.floor("D") == day]
    merged = offline.merge(online, on=["station_key", "origin_hour", "horizon"],
                           suffixes=("_offline", "_online"), how="left", indicator=True)
    assert (merged["_merge"] == "both").all()
    for column in FEATURE_COLUMNS[1:]:
        np.testing.assert_array_equal(merged[f"{column}_offline"].to_numpy("float64"),
                                      merged[f"{column}_online"].to_numpy("float64"),
                                      err_msg=column)
