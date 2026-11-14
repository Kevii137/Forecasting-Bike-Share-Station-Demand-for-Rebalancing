"""FastAPI backend: forecasts and demand-pressure alerts from the champion model.

Times are Chicago wall-clock hours without a zone, as in the Divvy data. The service
replays history: an origin is the last complete hour at which a dispatcher would ask.
"""
import json
import logging
import os
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from bikeshare.config import load_params
from bikeshare.modeling.tracking import DEFAULT_TRACKING_URI
from bikeshare.monitoring.profile import load_profile as load_logged_profile
from bikeshare.monitoring.replay import ReplayMonitor, Thresholds
from bikeshare.serving.forecast import Rules, alerts, forecast_rows, station_summary
from bikeshare.serving.model_store import ModelStore
from bikeshare.serving.online import FeatureStore

log = logging.getLogger(__name__)

TIMEZONE = "America/Chicago"
REPORTS = ("data_quality", "stations", "panel", "features", "metrics")

REQUESTS = Counter("bikeshare_requests_total", "HTTP requests", ["endpoint", "status"])
# Buckets reach 60 s: computing a new day's features takes 15 to 35 s.
LATENCY = Histogram("bikeshare_request_seconds", "HTTP request latency", ["endpoint"],
                    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 60))
FORECASTS = Counter("bikeshare_station_forecasts_total", "Station forecasts served", ["source"])
MODEL_VERSION = Gauge("bikeshare_model_version", "Registry version of the loaded champion")


def _env(name: str, default: str | None = None):
    return field(default_factory=lambda: os.environ.get(name, default))


@dataclass
class Settings:
    """Paths and defaults, read from the environment when the app is created."""
    params_path: str = _env("PARAMS_PATH", "params.yaml")
    panel_path: str = _env("PANEL_PATH", "data/processed/panel")
    stations_path: str = _env("STATIONS_PATH", "data/interim/stations.parquet")
    reports_dir: str = _env("REPORTS_DIR", "reports")
    default_origin: str | None = _env("DEFAULT_ORIGIN")
    tracking_uri: str = _env("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def records(frame: pd.DataFrame) -> list[dict]:
    """JSON-safe rows: NaN becomes null and timestamps become ISO strings."""
    frame = frame.copy()
    for column in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            frame[column] = frame[column].dt.strftime("%Y-%m-%dT%H:%M")
    frame = frame.astype(object).where(frame.notna(), None)
    return [{key: (value.item() if isinstance(value, np.generic) else value)
             for key, value in row.items()} for row in frame.to_dict("records")]


def replay_window(params: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    first, last = params["split"]["replay"]
    return pd.Timestamp(first), (pd.Period(last, "M").end_time).floor("h")


def create_app(settings: Settings | None = None, features: FeatureStore | None = None,
               models: ModelStore | None = None,
               load_profile: Callable[[str], dict | None] = load_logged_profile) -> FastAPI:
    settings = settings or Settings()
    params = load_params(Path(settings.params_path))
    serving, monitoring = params["serving"], params["monitoring"]
    rules = Rules(
        min_trips_per_day=params["stations"]["min_trips_per_day"],
        allowed_types=tuple(params["stations"]["allowed_types"]),
        alert_limit=serving["alert_limit"],
        min_pressure=serving["min_pressure"],
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.features = features or FeatureStore(
            settings.panel_path, settings.stations_path, params["forecast"]["horizons"],
            workers=serving["feature_workers"], allowed=replay_window(params),
        )
        mlflow.set_tracking_uri(settings.tracking_uri)
        app.state.models = models or ModelStore(serving["model_name"], serving["model_alias"])
        app.state.models.refresh()
        app.state.models.poll(serving["model_poll_seconds"])
        # Read at scrape time, so a model swapped in by the poller shows without a request.
        MODEL_VERSION.set_function(lambda: float(app.state.models.current.version)
                                   if app.state.models.current else float("nan"))
        app.state.monitor = ReplayMonitor(
            app.state.features, app.state.models, rules, settings.panel_path,
            Thresholds(**monitoring["thresholds"]), monitoring["window_days"],
            params["evaluation"]["tier_bounds"], load_profile,
        )
        # Compute the default day in the background so the first page load is quick.
        threading.Thread(target=app.state.features.features_at,
                         args=(parse_origin(None),), daemon=True).start()
        yield
        app.state.monitor.stop()
        app.state.models.stop()
        app.state.features.close()

    app = FastAPI(title="Bike-share demand forecasting API", version="1.0", lifespan=lifespan)

    @app.middleware("http")
    async def measure(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        # Label by route template, not raw path, to keep metric cardinality bounded.
        endpoint = getattr(request.scope.get("route"), "path", "unmatched")
        LATENCY.labels(endpoint).observe(time.perf_counter() - started)
        REQUESTS.labels(endpoint, str(response.status_code)).inc()
        return response

    def loaded_model():
        current = app.state.models.current
        if current is None:
            raise HTTPException(503, f"model not loaded: {app.state.models.last_error}")
        return current

    def parse_origin(origin: str | None) -> pd.Timestamp:
        store: FeatureStore = app.state.features
        text = origin or settings.default_origin
        try:
            value = pd.Timestamp(text) if text else store.first_origin
            store.check_origin(value)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        return value

    def forecast_for(origin: pd.Timestamp):
        current = loaded_model()
        features = app.state.features.features_at(origin)
        if features.empty:
            raise HTTPException(404, f"no active stations at {origin}")
        rows = forecast_rows(features, current.model, rules)
        summary = station_summary(features, rows)
        for source, count in summary["source"].value_counts().items():
            FORECASTS.labels(source).inc(int(count))
        return current, features, rows, summary

    def envelope(origin: pd.Timestamp, current) -> dict:
        return {"origin": origin.strftime("%Y-%m-%dT%H:%M"), "timezone": TIMEZONE,
                "horizons": params["forecast"]["horizons"], "model_version": current.version}

    @app.get("/health")
    def health() -> dict:
        """Liveness: the process is running."""
        return {"status": "ok"}

    @app.get("/ready")
    def ready(response: Response) -> dict:
        """Readiness: a model is loaded and the replay data is available."""
        current = app.state.models.current
        if current is None:
            response.status_code = 503
            return {"ready": False, "reason": app.state.models.last_error}
        return {"ready": True, "model_version": current.version}

    @app.get("/model")
    def model_info() -> dict:
        current = loaded_model()
        store: ModelStore = app.state.models
        return {"name": store.name, "alias": store.alias, "version": current.version,
                "run_id": current.run_id, "loaded_at": current.loaded_at, "tags": current.tags,
                "last_refresh_error": store.last_error}

    @app.post("/model/reload")
    def reload_model() -> dict:
        """Check the registry now instead of waiting for the next poll."""
        changed = app.state.models.refresh()
        return {"reloaded": changed, "version": loaded_model().version}

    @app.get("/replay")
    def replay() -> dict:
        store: FeatureStore = app.state.features
        default = parse_origin(None)
        return {"first_origin": store.first_origin.strftime("%Y-%m-%dT%H:%M"),
                "last_origin": store.last_origin.strftime("%Y-%m-%dT%H:%M"),
                "default_origin": default.strftime("%Y-%m-%dT%H:%M"), "timezone": TIMEZONE}

    @app.get("/stations")
    def stations() -> list[dict]:
        return records(app.state.features.stations)

    @app.get("/forecast")
    def forecast(origin: str | None = Query(None, examples=["2026-06-17T08:00"])) -> dict:
        """Forecast totals over the horizon for every active station."""
        when = parse_origin(origin)
        current, _, _, summary = forecast_for(when)
        return {**envelope(when, current), "stations": records(summary)}

    @app.get("/alerts")
    def alert_list(origin: str | None = Query(None, examples=["2026-06-17T08:00"]),
                   limit: int | None = Query(None, ge=1, le=100)) -> dict:
        """Stations ranked by demand pressure: likely draining and likely filling."""
        when = parse_origin(origin)
        current, _, _, summary = forecast_for(when)
        chosen = Rules(**{**rules.__dict__, "alert_limit": limit or rules.alert_limit})
        ranked = alerts(summary, chosen)
        return {**envelope(when, current), "min_pressure": rules.min_pressure,
                "stations_forecast": len(summary),
                "stations_on_model": int((summary["source"] == "model").sum()),
                "draining": records(ranked["draining"]), "filling": records(ranked["filling"])}

    @app.get("/station")
    def station(key: str, origin: str | None = None,
                history_hours: int = Query(24, ge=1, le=168)) -> dict:
        """One station: recent observed demand, the forecast, and what happened next."""
        when = parse_origin(origin)
        current, _, rows, summary = forecast_for(when)
        selected = summary[summary["station_key"] == key]
        if selected.empty:
            raise HTTPException(404, f"station {key!r} is not active at {when}")
        horizon = max(params["forecast"]["horizons"])
        store: FeatureStore = app.state.features
        history = store.actuals(key, when - pd.Timedelta(hours=history_hours - 1), when)
        # Replay only: the hours after the origin have already happened in the data.
        observed = store.actuals(key, when + pd.Timedelta(hours=1),
                                 when + pd.Timedelta(hours=horizon))
        return {**envelope(when, current), "station": records(selected)[0],
                "forecast": records(rows[rows["station_key"] == key].drop(columns="station_key")),
                "history": records(history), "observed_after_origin": records(observed)}

    @app.get("/pipeline")
    def pipeline() -> dict:
        """Data-quality and evaluation reports written by the DVC pipeline."""
        reports = {}
        for name in REPORTS:
            path = Path(settings.reports_dir) / f"{name}.json"
            reports[name] = json.loads(path.read_text()) if path.exists() else None
        if reports["metrics"]:
            reports["metrics"] = {key: reports["metrics"].get(key) for key in
                                  ("summary", "test_rows", "lightgbm_run_id")}
        return reports

    @app.get("/monitoring")
    def monitoring_summary() -> dict:
        """Replay status, the latest rolling window and one row per simulated day."""
        return app.state.monitor.summary()

    @app.post("/monitoring/replay/start")
    def start_replay(first: str | None = Query(None, examples=["2026-03-01T00:00"]),
                     last: str | None = Query(None, examples=["2026-09-30T20:00"]),
                     delay_seconds: float = Query(0.0, ge=0, le=60)) -> dict:
        """Advance a simulated clock hour by hour, scoring forecasts as hours are observed."""
        loaded_model()
        store: FeatureStore = app.state.features
        start = parse_origin(first) if first else store.first_origin
        end = parse_origin(last) if last else store.last_origin
        try:
            app.state.monitor.start(start, end, delay_seconds)
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        return app.state.monitor.status()

    @app.post("/monitoring/replay/stop")
    def stop_replay() -> dict:
        app.state.monitor.stop()
        return app.state.monitor.status()

    @app.post("/monitoring/replay/pause")
    def pause_replay() -> dict:
        """Hold the simulated clock; returns once the hour in progress has finished."""
        try:
            app.state.monitor.pause()
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error
        except TimeoutError as error:
            raise HTTPException(504, str(error)) from error
        return app.state.monitor.status()

    @app.post("/monitoring/replay/resume")
    def resume_replay() -> dict:
        app.state.monitor.resume()
        return app.state.monitor.status()

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    uvicorn.run(create_app(), host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
