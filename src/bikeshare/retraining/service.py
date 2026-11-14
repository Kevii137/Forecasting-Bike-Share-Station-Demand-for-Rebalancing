"""Trainer service: turns retraining alerts into retraining jobs (Decision D9).

    POST /alertmanager   Alertmanager webhook: firing alerts labelled action=retrain start a job
    POST /retrain        start a job by hand; skips the cooldown
    POST /rollback       move the champion alias back to the version it replaced
    GET  /status         the job in progress and recent results
    GET  /jobs           retraining runs recorded in MLflow, newest first
    GET  /metrics        Prometheus metrics

A job pauses the backend's replay, so simulated time stands still while the model trains,
runs bikeshare.retraining.job in a subprocess, asks the backend to load a promoted model,
and resumes the replay. Only one job runs at a time. After an attempt, alerts are ignored
for `cooldown_days` simulated days, because a rejected challenger leaves the alert firing
and Alertmanager repeats it.
"""
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, Response
from mlflow import MlflowClient
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, generate_latest

from bikeshare.config import load_params
from bikeshare.modeling.register import MODEL_NAME
from bikeshare.modeling.tracking import DEFAULT_TRACKING_URI, EXPERIMENT
from bikeshare.retraining import registry
from bikeshare.retraining.job import STAGE

log = logging.getLogger(__name__)

OUTCOMES = ("promoted", "rejected", "stale", "failed")
RUNNING = Gauge("bikeshare_retrain_running", "1 while a retraining job runs")
JOBS = Counter("bikeshare_retrain_jobs_total", "Finished retraining jobs", ["outcome"])
WEBHOOKS = Counter("bikeshare_retrain_webhooks_total", "Alertmanager notifications",
                   ["decision"])
DURATION = Gauge("bikeshare_retrain_last_duration_seconds", "Wall-clock time of the last job")
ROLLBACKS = Counter("bikeshare_rollbacks_total", "Champion rollbacks")
for _outcome in OUTCOMES:
    JOBS.labels(_outcome)
for _decision in ("started", "ignored"):
    WEBHOOKS.labels(_decision)


class Busy(RuntimeError):
    """A job is already running."""


class Backend:
    """The backend's replay and model endpoints, over plain HTTP."""

    def __init__(self, url: str, timeout: float = 240.0):
        self.url = url.rstrip("/")
        self.timeout = timeout  # pausing waits for the hour in progress, up to about 45 s

    def _call(self, method: str, path: str) -> dict:
        request = urllib.request.Request(f"{self.url}{path}", method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"{method} {path} returned {error.code}: "
                               f"{error.read().decode(errors='replace')}") from error

    def replay_status(self) -> dict:
        return self._call("GET", "/monitoring")

    def pause(self) -> dict:
        return self._call("POST", "/monitoring/replay/pause")

    def resume(self) -> dict:
        return self._call("POST", "/monitoring/replay/resume")

    def reload_model(self) -> dict:
        return self._call("POST", "/model/reload")


def run_subprocess(as_of: pd.Timestamp, trigger: str) -> dict:
    """Run one job in a child process and return its result."""
    with tempfile.TemporaryDirectory() as scratch:
        result = Path(scratch) / "result.json"
        command = [sys.executable, "-m", "bikeshare.retraining.job",
                   "--as-of", f"{as_of:%Y-%m-%dT%H:%M}", "--trigger", trigger,
                   "--result", str(result)]
        code = subprocess.run(command, check=False).returncode  # its log goes to the service's output
        if code != 0:
            hint = " (killed, most likely out of memory)" if code in (-9, 137) else ""
            raise RuntimeError(f"the retraining job exited with code {code}{hint}")
        return json.loads(result.read_text())


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Trainer:
    def __init__(self, backend: Backend, runner: Callable[[pd.Timestamp, str], dict],
                 cooldown_days: float):
        self.backend = backend
        self.runner = runner
        self.cooldown = pd.Timedelta(days=cooldown_days)
        self._lock = threading.Lock()
        self.job: dict | None = None
        self.history: deque[dict] = deque(maxlen=20)
        self.last_as_of: pd.Timestamp | None = None
        self._thread: threading.Thread | None = None

    # Starting jobs ---------------------------------------------------------------------

    def on_alerts(self, payload: dict) -> dict:
        """Decide what to do with one Alertmanager notification."""
        wanted = [alert for alert in payload.get("alerts", [])
                  if alert.get("status") == "firing"
                  and alert.get("labels", {}).get("action") == "retrain"]
        if not wanted:
            return self._ignored("no firing alert asks for retraining")
        trigger = wanted[0]["labels"].get("alertname", "alert")
        try:
            return self.start(trigger, source="alertmanager", enforce_cooldown=True)
        except (Busy, ValueError) as error:
            return self._ignored(str(error))

    def _ignored(self, reason: str) -> dict:
        WEBHOOKS.labels("ignored").inc()
        log.info("Alert ignored: %s", reason)
        return {"decision": "ignored", "reason": reason}

    def start(self, trigger: str, source: str, as_of: pd.Timestamp | None = None,
              enforce_cooldown: bool = False) -> dict:
        with self._lock:
            if self.job is not None:
                raise Busy(f"a retraining job is already running (as of {self.job['as_of']})")
            if as_of is None:
                clock = self.backend.replay_status()["clock"]
                if clock is None:
                    raise ValueError("the replay has no simulated time yet; give as_of")
                planned = pd.Timestamp(clock)
            else:
                planned = pd.Timestamp(as_of)
            if enforce_cooldown and self.last_as_of is not None:
                since = planned - self.last_as_of
                if pd.Timedelta(0) <= since < self.cooldown:
                    raise ValueError(f"cooldown: the last attempt was as of {self.last_as_of}, "
                                     f"{since} of simulated time ago")
            self.job = {"trigger": trigger, "source": source, "started_at": now(),
                        "as_of": f"{planned:%Y-%m-%dT%H:%M}", "from_replay": as_of is None,
                        "phase": "starting"}
            job = dict(self.job)
            self._thread = threading.Thread(target=self._execute, args=(job,),
                                            name="retrain", daemon=True)
            RUNNING.set(1)
            self._thread.start()
        if source == "alertmanager":
            WEBHOOKS.labels("started").inc()
        log.info("Retraining started by %s (%s)", source, trigger)
        return {"decision": "started", "job": job}

    # Running a job ---------------------------------------------------------------------

    def _phase(self, phase: str, **details) -> None:
        with self._lock:
            if self.job is not None:
                self.job.update(phase=phase, **details)

    def _execute(self, job: dict) -> None:
        started = time.monotonic()
        paused = False
        as_of = pd.Timestamp(job["as_of"])
        result: dict = {}
        try:
            if job["from_replay"]:
                status = self.backend.replay_status()
                if status["running"] and not status["paused"]:
                    self._phase("pausing the replay")
                    status = self.backend.pause()
                    paused = True
                as_of = pd.Timestamp(status["clock"])
                self._phase("training", as_of=f"{as_of:%Y-%m-%dT%H:%M}")
            else:
                self._phase("training")
            result = self.runner(as_of, job["trigger"])
            if result.get("outcome") == "promoted":
                self._phase("loading the new champion")
                self.backend.reload_model()
            outcome = result.get("outcome", "failed")
        except Exception as error:  # any failure must still resume the replay
            log.exception("Retraining failed")
            result = {**result, "error": f"{type(error).__name__}: {error}"}
            outcome = "failed"
        finally:
            if paused:
                try:
                    self.backend.resume()
                except Exception:  # report it, but record the job outcome anyway
                    log.exception("Could not resume the replay")
        seconds = round(time.monotonic() - started, 1)
        record = {**job, **result, "as_of": f"{as_of:%Y-%m-%dT%H:%M}", "outcome": outcome,
                  "finished_at": now(), "seconds": seconds}
        record.pop("phase", None)
        JOBS.labels(outcome).inc()
        DURATION.set(seconds)
        with self._lock:
            self.last_as_of = as_of
            self.history.appendleft(record)
            self.job = None
            RUNNING.set(0)
        log.info("Retraining finished: %s in %.0f s", outcome, seconds)

    def wait(self, timeout: float = 60.0) -> None:
        if self._thread:
            self._thread.join(timeout)

    # Reporting -------------------------------------------------------------------------

    def status(self) -> dict:
        with self._lock:
            return {"running": self.job is not None, "job": self.job,
                    "cooldown_days": self.cooldown / pd.Timedelta(days=1),
                    "last_as_of": None if self.last_as_of is None
                    else f"{self.last_as_of:%Y-%m-%dT%H:%M}",
                    "history": list(self.history)}


def recorded_jobs(client: MlflowClient, limit: int = 20) -> list[dict]:
    """Retraining runs from MLflow, which outlive the service's memory."""
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        return []
    runs = client.search_runs([experiment.experiment_id], f"tags.stage = '{STAGE}'",
                              order_by=["attributes.start_time DESC"], max_results=limit)
    jobs = []
    for run in runs:
        tags, metrics = run.data.tags, run.data.metrics
        rolled_back = None
        if tags.get("outcome") == "promoted" and tags.get("version"):
            rolled_back = client.get_model_version(
                MODEL_NAME, tags["version"]).tags.get("rolled_back_at")
        jobs.append({
            "run_id": run.info.run_id, "status": run.info.status,
            "started_at": datetime.fromtimestamp(run.info.start_time / 1000, UTC)
            .isoformat(timespec="seconds"),
            **{key: tags.get(key) for key in ("as_of", "trigger", "gate", "outcome", "version",
                                              "champion_version", "split_test")},
            "rolled_back_at": rolled_back,
            **{f"{who}_{demand}_wape": metrics.get(f"gate_{who}_{demand}_wape")
               for who in ("challenger", "champion") for demand in ("pickups", "dropoffs")},
        })
    return jobs


def create_app(trainer: Trainer | None = None, client: MlflowClient | None = None) -> FastAPI:
    if trainer is None:
        params = load_params()
        backend = Backend(os.environ.get("BACKEND_URL", "http://localhost:8000"))
        trainer = Trainer(backend, run_subprocess, params["retraining"]["cooldown_days"])
    if client is None:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
        client = MlflowClient()
    app = FastAPI(title="Bike-share retraining service", version="1.0")
    app.state.trainer, app.state.client = trainer, client

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/alertmanager")
    async def alertmanager(request: Request) -> dict:
        return trainer.on_alerts(await request.json())

    @app.post("/retrain")
    def retrain(as_of: str | None = Query(None, examples=["2026-03-11T23:00"])) -> dict:
        """Start a job now. Without as_of it retrains at the replay's simulated time."""
        try:
            return trainer.start("manual", source="manual",
                                 as_of=pd.Timestamp(as_of) if as_of else None)
        except Busy as error:
            raise HTTPException(409, str(error)) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/rollback")
    def rollback() -> dict:
        try:
            moved = registry.rollback(client)
        except registry.RegistryError as error:
            raise HTTPException(409, str(error)) from error
        ROLLBACKS.inc()
        log.info("Rolled back from version %s to %s", moved["from"], moved["to"])
        try:
            loaded = trainer.backend.reload_model()["version"]
        except Exception as error:  # noqa: BLE001 - the backend's poller will pick it up
            loaded = f"not confirmed ({error}); the backend checks every 30 s"
        return {**moved, "backend_version": loaded}

    @app.get("/status")
    def status() -> dict:
        return {**trainer.status(), "champion_version": registry.champion_version(client)}

    @app.get("/jobs")
    def jobs(limit: int = Query(20, ge=1, le=100)) -> list[dict]:
        return recorded_jobs(client, limit)

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    uvicorn.run(create_app(), host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "8001")))


if __name__ == "__main__":
    main()
