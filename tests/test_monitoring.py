"""Drift and error monitoring: PSI, rolling windows, the replay monitor and its API."""
from pathlib import Path

import mlflow.pyfunc
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from helpers import RULES, ConstantModel, FakeClient
from prometheus_client import REGISTRY

from bikeshare.modeling.metrics import wape
from bikeshare.monitoring.drift import bin_edges, build_profile, histogram, psi
from bikeshare.monitoring.replay import ReplayMonitor, Thresholds, wait_until_idle
from bikeshare.monitoring.windows import DayRecord, error_sums, window_metrics
from bikeshare.serving.app import Settings, create_app
from bikeshare.serving.forecast import fallback_forecast
from bikeshare.serving.model_store import ModelStore
from bikeshare.serving.online import FeatureStore, read_panel

TIERS = [5, 10, 20]
DRIFT_COLUMNS = ["pickups_mean_24h", "trips_per_day_28d", "pickups_forecast"]


def test_psi_is_near_zero_for_the_reference_and_large_for_a_shift():
    rng = np.random.default_rng(1)
    profile = build_profile(pd.DataFrame({"x": rng.poisson(2.0, 50_000)}), ["x"], bins=10)
    spec = profile["columns"]["x"]
    assert psi(spec["shares"], histogram(rng.poisson(2.0, 50_000), spec["edges"])) < 0.01
    assert psi(spec["shares"], histogram(rng.poisson(4.0, 50_000), spec["edges"])) > 0.25
    assert np.isnan(psi(spec["shares"], np.zeros(len(spec["shares"]))))


def test_zero_heavy_reference_keeps_zeros_in_a_bin_of_their_own():
    values = np.r_[np.zeros(900), np.arange(1, 101)]
    edges = bin_edges(values, 10)
    assert edges[0] == 0 and len(edges) == len(set(edges))
    assert histogram(values, edges)[0] == 900
    assert histogram(np.array([np.nan, 0.0]), edges).sum() == 1  # NaN is skipped


def scored_rows(actual: list[float], forecast: float, baseline: float) -> pd.DataFrame:
    count = len(actual)
    return pd.DataFrame({
        "horizon": [1, 2, 3] * (count // 3), "trips_per_day_28d": 12.0,
        "pickups_target": actual, "pickups_forecast": forecast, "pickups_baseline": baseline,
        "dropoffs_target": actual, "dropoffs_forecast": forecast, "dropoffs_baseline": baseline,
    })


def test_window_sums_days_before_dividing():
    quiet, busy = [0.0, 1.0, 0.0], [6.0, 9.0, 3.0]
    days = [DayRecord(pd.Timestamp("2026-03-01"), "1", error_sums(scored_rows(quiet, 2, 1), TIERS)),
            DayRecord(pd.Timestamp("2026-03-02"), "1", error_sums(scored_rows(busy, 2, 1), TIERS))]
    table = window_metrics(days, None)["errors"]
    overall = table.loc[("pickups", "all", "all")]
    actual = np.array(quiet + busy)
    assert overall["model_wape"] == pytest.approx(wape(actual, np.full(6, 2.0)))
    assert overall["baseline_wape"] == pytest.approx(wape(actual, np.full(6, 1.0)))
    assert overall["relative_wape"] == pytest.approx(
        overall["model_wape"] / overall["baseline_wape"])
    assert table.loc[("pickups", "tier", "10-20"), "rows"] == 6
    assert table.loc[("pickups", "horizon", "2"), "actual"] == 10.0


def make_profile(store: FeatureStore) -> dict:
    frame = store.day_features(store.first_origin).assign(pickups_forecast=4.0)
    return build_profile(frame[frame["station_key"] == "busy"], DRIFT_COLUMNS, bins=5)


@pytest.fixture
def models(monkeypatch):
    monkeypatch.setattr(mlflow.pyfunc, "load_model", lambda uri: ConstantModel())
    store = ModelStore("demand", "champion", FakeClient())
    store.refresh()
    return store


def test_replay_scores_observed_hours_and_publishes_signals(store, data_dir, models):
    profile = make_profile(store)
    # Every full window breaches the error threshold of 0, so the streak counts full days.
    thresholds = Thresholds(max_psi=100.0, relative_wape=0.0, sustain_days=2)
    monitor = ReplayMonitor(store, models, RULES, str(data_dir / "panel"), thresholds,
                            window_days=2, tier_bounds=TIERS, load_profile=lambda run: profile)
    first = store.first_origin
    monitor.start(first, first + pd.Timedelta(hours=3 * 24 - 1))
    wait_until_idle(monitor, timeout=120)

    summary = monitor.summary()
    assert summary["error"] is None, summary["error"]
    assert [row["window_full"] for row in summary["history"]] == [False, True, True]
    assert summary["breach_days"] == {"drift": 0, "error": 2}
    assert summary["alerting"] == {"drift": False, "error": True}

    # Days two and three: every target hour is forecast by three origins (h = 1, 2, 3),
    # all for the busy station, with a constant pickup forecast of 4.
    day_two = first + pd.Timedelta(days=1)
    panel = read_panel(str(data_dir / "panel"), day_two, day_two + pd.Timedelta(hours=47))
    pickups = panel.loc[panel["station_key"] == "busy", "pickups"].to_numpy("float64")
    expected = wape(pickups, np.full(len(pickups), 4.0))
    latest = pd.DataFrame(summary["latest"]["errors"]).set_index(["target", "group", "label"])
    assert latest.loc[("pickups", "all", "all"), "model_wape"] == pytest.approx(expected)
    assert latest.loc[("pickups", "all", "all"), "rows"] == 3 * 48
    assert REGISTRY.get_sample_value(
        "bikeshare_monitor_wape",
        {"target": "pickups", "forecaster": "model", "group": "all", "label": "all"},
    ) == pytest.approx(expected)
    assert set(summary["latest"]["psi"]) == set(DRIFT_COLUMNS)
    assert REGISTRY.get_sample_value("bikeshare_monitor_breach_days", {"signal": "error"}) == 2


def test_new_champion_restarts_the_windows(store, data_dir, models):
    monitor = ReplayMonitor(store, models, RULES, str(data_dir / "panel"),
                            Thresholds(1.0, 1.0, 3), window_days=2, tier_bounds=TIERS,
                            load_profile=lambda run: None)
    assert np.isnan(REGISTRY.get_sample_value("bikeshare_replay_clock_seconds"))
    first = store.first_origin
    for origin in pd.date_range(first, first + pd.Timedelta(hours=23), freq="h"):
        monitor.step(origin)
    assert monitor.status()["window_days"] == 1
    models.client.version = "2"
    models.refresh()
    monitor.step(first + pd.Timedelta(hours=24))
    status = monitor.status()
    assert status["model_version"] == "2" and status["window_days"] == 0
    assert status["profile_loaded"] is False


def test_replay_api_starts_stops_and_rejects_overlap(store, data_dir, models):
    settings = Settings(params_path=str(data_dir / "params.yaml"),
                        panel_path=str(data_dir / "panel"),
                        stations_path=str(data_dir / "stations.parquet"), reports_dir="reports")
    store.close = lambda: None  # the module fixture owns the pool
    app = create_app(settings, features=store, models=models, load_profile=lambda run: None)
    first = f"{store.first_origin:%Y-%m-%dT%H:%M}"
    with TestClient(app) as client:
        # Published from the loaded model at scrape time, before any request.
        assert REGISTRY.get_sample_value("bikeshare_model_version") == 1.0
        assert client.post("/monitoring/replay/start",
                           params={"first": "2020-01-01T00:00"}).status_code == 422
        started = client.post("/monitoring/replay/start",
                              params={"first": first, "delay_seconds": 0.5})
        assert started.status_code == 200 and started.json()["running"] is True
        assert client.post("/monitoring/replay/start").status_code == 409
        stopped = client.post("/monitoring/replay/stop").json()
        assert stopped["running"] is False and stopped["clock"] is not None
        body = client.get("/monitoring").json()
        assert body["thresholds"]["sustain_days"] > 0 and "history" in body
        assert "bikeshare_replay_hours_total" in client.get("/metrics").text


@pytest.mark.skipif(not Path("data/processed/features").exists(), reason="needs DVC data")
def test_live_day_record_equals_the_offline_calibration_record(models):
    """The thresholds are calibrated offline, so the replay must compute the same numbers."""
    store = FeatureStore("data/processed/panel", "data/interim/stations.parquet", [1, 2, 3],
                         workers=4)
    monitor = ReplayMonitor(store, models, RULES, "data/processed/panel", Thresholds(1, 1, 3),
                            window_days=7, tier_bounds=TIERS, load_profile=lambda run: None)
    first, day = pd.Timestamp("2026-07-04"), pd.Timestamp("2026-07-05")
    try:
        for origin in pd.date_range(first, day + pd.Timedelta(hours=23), freq="h"):
            monitor.step(origin)
    finally:
        store.close()
    live = monitor.records[-1].errors

    offline = pd.read_parquet("data/processed/features", filters=[("month", "=", "2026-07")])
    offline = offline[(offline["target_hour"].dt.floor("D") == day)
                      & (offline["origin_hour"] >= first)].dropna(subset=["pickups_target"])
    for demand, value in (("pickups", 4.0), ("dropoffs", 1.0)):
        offline[f"{demand}_forecast"] = value
        offline[f"{demand}_baseline"] = fallback_forecast(offline, demand)
    expected = error_sums(offline, TIERS)
    pd.testing.assert_frame_equal(live.sort_index(), expected.sort_index(), check_dtype=False)
