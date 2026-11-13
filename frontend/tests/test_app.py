"""Render every page against a fake backend. Run with the frontend venv:

    cd frontend && ../.venv-frontend/bin/python -m pytest tests
"""
from pathlib import Path
from unittest import mock

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
}


def fake_request(method, url, timeout, params):
    path = url.split(":8000", 1)[1]
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


def test_unreachable_backend_shows_an_error(backend):
    import requests

    backend.side_effect = requests.ConnectionError("refused")
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    assert "Cannot reach the backend" in app.error[0].value
