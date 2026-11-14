"""Render every page against a fake backend. Run with the frontend venv:

    cd frontend && ../.venv-frontend/bin/python -m pytest tests
"""
from pathlib import Path
from unittest import mock
from urllib.parse import urlsplit

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parents[1] / "app.py")
ORIGIN = "2026-06-17T08:00"


def summary(key, pickups, dropoffs, capacity, source="model"):
    net = dropoffs - pickups
    return {"station_key": key, "pickups": pickups, "dropoffs": dropoffs, "source": source,
            "lat": 41.88, "lng": -87.63, "capacity": capacity, "station_type": "classic",
            "trips_per_day_28d": 20.0, "net_flow": net,
            "pressure": None if capacity is None else net / capacity,
            "confidence": "normal" if source == "model" else "low"}


DRAIN = summary("Harper Ave & 55th St", 6.8, 3.9, 15.0)
FILL = summary("LaSalle St & Washington St", 4.6, 11.8, 15.0)
QUIET = summary("Quiet Rack", 0.1, 0.1, None, source="historical_average")
ENVELOPE = {"origin": ORIGIN, "timezone": "America/Chicago", "horizons": [1, 2, 3],
            "model_version": "1"}
RESPONSES = {
    "/replay": {"first_origin": "2026-03-01T00:00", "last_origin": "2026-09-30T20:00",
                "default_origin": ORIGIN, "timezone": "America/Chicago"},
    "/alerts": {**ENVELOPE, "min_pressure": 0.1, "stations_forecast": 3,
                "stations_on_model": 2, "draining": [DRAIN], "filling": [FILL]},
    "/forecast": {**ENVELOPE, "stations": [DRAIN, FILL, QUIET]},
    "/station": {
        **ENVELOPE, "station": DRAIN,
        "forecast": [{"origin_hour": ORIGIN, "horizon": h, "target_hour": f"2026-06-17T{8 + h:02d}:00",
                      "source": "model", "pickups_forecast": 2.0, "dropoffs_forecast": 1.0}
                     for h in (1, 2, 3)],
        "history": [{"hour": f"2026-06-17T0{h}:00", "pickups": 1, "dropoffs": 0}
                    for h in range(5, 9)],
        "observed_after_origin": [{"hour": f"2026-06-17T{8 + h:02d}:00", "pickups": 2, "dropoffs": 1}
                                  for h in (1, 2, 3)],
    },
    "/model": {"name": "bikeshare-demand", "alias": "champion", "version": "1",
               "run_id": "abc", "loaded_at": "2026-11-13T06:21:31+00:00",
               "tags": {"git_commit": "3b28d186d5f1", "git_dirty": "False",
                        "dvc_features_md5": "b6dc.dir", "split_test": "2026-02-2026-02"},
               "last_refresh_error": None},
    "/pipeline": {
        "data_quality": {"rows_read": 100, "rows_written": 97, "rejected_share": 0.03},
        "stations": {"stations": 10, "trip_share_with_capacity": 0.99},
        "panel": {"rows": 1000, "zero_hour_share": 0.8},
        "features": {"rows": 500, "by_month": {"2026-03": {}}},
        "metrics": {"summary": {"lightgbm": {"pickups_wape": 0.8, "dropoffs_wape": 0.8,
                                             "net_flow_mae": 1.2}}},
    },
    "/monitoring": {
        "running": True, "first": "2026-03-01T00:00", "last": "2026-09-30T20:00",
        "clock": "2026-03-10T05:00", "delay_seconds": 0.0, "model_version": "1",
        "profile_loaded": True, "window_days": 7, "window_size": 7,
        "breach_days": {"drift": 3, "error": 0}, "alerting": {"drift": True, "error": False},
        "thresholds": {"max_psi": 0.5, "relative_wape": 1.0, "sustain_days": 3}, "error": None,
        "latest": {
            "first_day": "2026-03-03", "last_day": "2026-03-09",
            "errors": [{"target": target, "group": group, "label": label, "relative_wape": 0.97,
                        "model_wape": 0.8, "baseline_wape": 0.82, "model_abs_error": 1.0,
                        "baseline_abs_error": 1.0, "actual": 1.0, "rows": 1}
                       for target in ("pickups", "dropoffs")
                       for group, label in (("all", "all"), ("horizon", "1"), ("tier", "5-10"))],
            "psi": {"trips_per_day_28d": 0.8, "pickups_forecast": 0.3},
        },
        "history": [{"day": f"2026-03-{day:02d}", "model_version": "1", "window_full": day >= 7,
                     "max_psi": 0.8 if day >= 7 else None, "drift_days": max(day - 6, 0),
                     "error_days": 0, "pickups_relative_wape": 0.97,
                     "dropoffs_relative_wape": 1.01} for day in range(1, 10)],
    },
}


GATE_FAILED = "dropoffs tier 5-10: challenger WAPE 1.0100 is more than the tolerance worse than champion (0.9700)"
RESPONSES.update({
    "/status": {"running": True, "cooldown_days": 7.0, "last_as_of": "2026-03-11T23:00",
                "champion_version": "1",
                "job": {"trigger": "BikeshareForecastDegraded", "source": "alertmanager",
                        "started_at": "2026-11-14T13:00:00+00:00", "as_of": "2026-03-18T23:00",
                        "from_replay": True, "phase": "training"},
                "history": [{"outcome": "rejected", "as_of": "2026-03-11T23:00",
                             "gate": {"passed": False, "failed": [GATE_FAILED]}}]},
    "/jobs": [{"run_id": "r2", "status": "FINISHED", "started_at": "2026-11-14T12:00:00+00:00",
               "as_of": "2026-03-11T23:00", "trigger": "BikeshareForecastDegraded",
               "gate": "failed", "outcome": "rejected", "version": "2",
               "champion_version": "1", "split_test": "2026-02-26 00:00 to 2026-03-11 23:00",
               "rolled_back_at": None,
               "challenger_pickups_wape": 0.81, "champion_pickups_wape": 0.83,
               "challenger_dropoffs_wape": 0.84, "champion_dropoffs_wape": 0.85}],
    "/rollback": {"from": "2", "to": "1", "backend_version": "1"},
})


def fake_request(method, url, timeout, params):
    path = urlsplit(url).path
    response = mock.Mock(status_code=200)
    response.json.return_value = RESPONSES[path]
    return response


@pytest.fixture(autouse=True)
def backend():
    st.cache_data.clear()  # api.py caches responses across app runs in one process
    with mock.patch("requests.request", side_effect=fake_request) as patched:
        yield patched


def run(page: str | None = None) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    if page:
        app.switch_page(page).run()
    assert not app.exception, [error.value for error in app.exception]
    return app


def test_alerts_page_lists_both_directions():
    app = run()
    assert app.title[0].value == "Rebalancing alerts"
    assert {metric.label: metric.value for metric in app.metric}["Likely draining"] == "1"
    tables = [frame.value for frame in app.dataframe]
    assert tables[0]["Station"].tolist() == ["Harper Ave & 55th St"]
    assert tables[1]["Station"].tolist() == ["LaSalle St & Washington St"]


def test_station_page_starts_with_the_top_alert():
    app = run("views/station.py")
    assert app.selectbox(key="station").value == "Harper Ave & 55th St"
    assert app.dataframe[0].value["Bikes out (actual)"].tolist() == [2, 2, 2]


def test_pipeline_console_shows_lineage_and_reloads(backend):
    app = run("views/pipeline.py")
    assert "3b28d186d5f1" in app.markdown[0].value
    app.button[0].click().run()
    assert any(call.args[:1] == ("POST",) for call in backend.call_args_list)


def test_pipeline_console_shows_retraining_jobs_and_gate_reasons(backend):
    app = run("views/pipeline.py")
    assert any("Retraining in progress (training)" in info.value for info in app.info)
    assert any(GATE_FAILED in warning.value for warning in app.warning)
    jobs = app.dataframe[0].value
    assert jobs["Outcome"].tolist() == ["rejected"] and jobs["Version"].tolist() == ["2"]
    assert next(button for button in app.button if button.label == "Retrain now").disabled
    next(button for button in app.button if button.label == "Confirm rollback").click().run()
    assert ("POST", "http://localhost:8001/rollback") in [
        call.args[:2] for call in backend.call_args_list]
    assert any("from version 2 to 1" in success.value for success in app.success)


def test_unreachable_backend_shows_an_error(backend):
    import requests

    backend.side_effect = requests.ConnectionError("refused")
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    assert "Cannot reach the backend" in app.error[0].value


def test_monitoring_page_shows_signals_and_the_firing_alert():
    app = run("views/monitoring.py")
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Drift breach days"] == "3 of 3"
    assert metrics["Simulated time"] == "10 Mar 05:00"
    assert any("Inputs have drifted" in warning.value for warning in app.warning)
    assert app.dataframe[1].value["Column"].tolist() == ["trips_per_day_28d", "pickups_forecast"]
    slices = app.dataframe[0].value.index.tolist()
    assert slices == ["All stations", "1 hour ahead", "5-10 trips/day"]
