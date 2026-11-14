"""Replay the monitoring months as simulated production traffic (Decision D9).

A simulated clock advances one origin hour at a time. At each hour the monitor forecasts
every active station through the same path as the API (shared features, champion model,
D2 routing). At the last hour of each day it scores the forecasts whose target hours have
now been observed, adds the day's drift counts, and publishes rolling-window metrics for
Prometheus.

A day breaches when its window crosses a threshold. Alerts need the breach to hold for
several consecutive simulated days, so a slow or fast replay gives the same alerts.

When the champion changes, the windows restart: the new model is judged on its own
forecasts, against its own reference profile. An alert therefore cannot fire again
until a full window has passed, which also acts as a cooldown.
"""
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from prometheus_client import Counter, Gauge

from bikeshare.features import DEMAND_COLUMNS
from bikeshare.monitoring.windows import (
    DayRecord,
    drift_counts,
    error_sums,
    window_metrics,
)
from bikeshare.serving.forecast import MODEL, Rules, fallback_forecast, forecast_rows
from bikeshare.serving.model_store import ModelStore
from bikeshare.serving.online import FeatureStore, read_panel

log = logging.getLogger(__name__)

CLOCK = Gauge("bikeshare_replay_clock_seconds",
              "Simulated origin hour, Chicago wall-clock time encoded as if it were UTC")
RUNNING = Gauge("bikeshare_replay_running", "1 while the replay is advancing or paused")
PAUSED = Gauge("bikeshare_replay_paused", "1 while the replay is paused, for example for retraining")
HOURS = Counter("bikeshare_replay_hours_total", "Simulated origin hours processed")
WINDOW_DAYS = Gauge("bikeshare_monitor_window_days", "Days in the current rolling window")
WAPE = Gauge("bikeshare_monitor_wape", "Rolling-window WAPE",
             ["target", "forecaster", "group", "label"])
RELATIVE = Gauge("bikeshare_monitor_relative_wape",
                 "Rolling-window model WAPE divided by historical-average WAPE",
                 ["target", "group", "label"])
PSI = Gauge("bikeshare_monitor_psi", "Rolling-window PSI against the training reference",
            ["column"])
MAX_PSI = Gauge("bikeshare_monitor_max_psi", "Largest PSI over the monitored columns")
STREAK = Gauge("bikeshare_monitor_breach_days",
               "Consecutive simulated days on which a signal crossed its threshold", ["signal"])
THRESHOLD = Gauge("bikeshare_monitor_threshold", "Thresholds from params.yaml", ["name"])

SIGNALS = ("drift", "error")


@dataclass(frozen=True)
class Thresholds:
    max_psi: float          # drift: the largest monitored PSI
    relative_wape: float    # error: model WAPE / historical-average WAPE, worse target
    sustain_days: int       # consecutive breaching days before an alert

    def publish(self) -> None:
        THRESHOLD.labels("max_psi").set(self.max_psi)
        THRESHOLD.labels("relative_wape").set(self.relative_wape)
        THRESHOLD.labels("sustain_days").set(self.sustain_days)


def stamp(value: pd.Timestamp | None) -> str | None:
    return None if value is None else value.strftime("%Y-%m-%dT%H:%M")


class ReplayMonitor:
    def __init__(self, features: FeatureStore, models: ModelStore, rules: Rules, panel_path: str,
                 thresholds: Thresholds, window_days: int, tier_bounds: list[float],
                 load_profile: Callable[[str], dict | None]):
        self.features = features
        self.models = models
        self.rules = rules
        self.panel_path = panel_path
        self.thresholds = thresholds
        self.window_days = window_days
        self.tier_bounds = tier_bounds
        self.load_profile = load_profile
        self._lock = threading.RLock()  # summary() calls status() while holding it
        self._stop = threading.Event()
        self._resume = threading.Event()  # cleared while paused
        self._resume.set()
        self._idle = threading.Event()  # set while the paused loop waits between hours
        self._thread: threading.Thread | None = None
        self.first = self.last = self.clock = None
        self.delay_seconds = 0.0
        self.error: str | None = None
        self.history: list[dict] = []
        self._reset(None)
        thresholds.publish()
        RUNNING.set(0)
        PAUSED.set(0)
        CLOCK.set(float("nan"))  # no simulated time until a replay starts

    # State -----------------------------------------------------------------------------

    def _reset(self, current) -> None:
        """Start empty windows for a newly loaded model."""
        self.version = current.version if current else None
        self.profile = self.load_profile(current.run_id) if current else None
        self.records: deque[DayRecord] = deque(maxlen=self.window_days)
        self.pending: list[pd.DataFrame] = []
        self.day_counts: dict[str, np.ndarray] = {}
        self.streaks = dict.fromkeys(SIGNALS, 0)
        self.latest: dict | None = None
        WINDOW_DAYS.set(0)
        for signal in SIGNALS:
            STREAK.labels(signal).set(0)
        for gauge in (WAPE, RELATIVE, PSI):
            gauge.clear()
        MAX_PSI.set(float("nan"))
        if current:
            log.info("Monitoring model version %s (reference profile %s)", self.version,
                     "loaded" if self.profile else "missing: drift is not measured")

    # Replay control ----------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def paused(self) -> bool:
        return self.running and not self._resume.is_set()

    def start(self, first: pd.Timestamp, last: pd.Timestamp, delay_seconds: float = 0.0) -> None:
        if self.running:
            raise RuntimeError("replay is already running")
        self.features.check_origin(first)
        self.features.check_origin(last)
        if last < first:
            raise ValueError("the replay must end after it starts")
        with self._lock:
            self._reset(self.models.current)
            self.history = []
            self.first, self.last, self.clock = first, last, None
            self.delay_seconds, self.error = delay_seconds, None
        self._stop.clear()
        self._resume.set()
        self._idle.clear()
        self._thread = threading.Thread(target=self._run, name="replay", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=120)

    def pause(self, timeout: float = 180.0) -> None:
        """Hold the clock after the hour in progress, and return once it has finished.

        Retraining pauses the replay so that simulated time does not run ahead while a
        model trains on the data observed so far.
        """
        if not self.running:
            raise RuntimeError("replay is not running")
        self._resume.clear()
        PAUSED.set(1)
        deadline = time.monotonic() + timeout
        while self.running and not self._idle.is_set():
            if time.monotonic() > deadline:
                raise TimeoutError("the replay did not finish its current hour in time")
            time.sleep(0.05)

    def resume(self) -> None:
        self._resume.set()
        PAUSED.set(0)

    def _hold_while_paused(self) -> None:
        if self._resume.is_set():
            return
        self._idle.set()
        while not self._resume.wait(0.5) and not self._stop.is_set():
            pass
        self._idle.clear()

    def _run(self) -> None:
        RUNNING.set(1)
        try:
            for origin in pd.date_range(self.first, self.last, freq="h"):
                self._hold_while_paused()
                if self._stop.is_set():
                    break
                self.step(origin)
                if self.delay_seconds:
                    self._stop.wait(self.delay_seconds)
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
            log.exception("Replay stopped at %s", self.clock)
        finally:
            self._idle.set()  # a pause() waiting on this thread can return
            RUNNING.set(0)
            PAUSED.set(0)

    # One simulated hour ------------------------------------------------------------------

    def step(self, origin: pd.Timestamp) -> None:
        current = self.models.current
        if current is None:
            raise RuntimeError("no model loaded")
        with self._lock:
            if current.version != self.version:
                self._reset(current)
        features = self.features.features_at(origin)
        rows = forecast_rows(features, current.model, self.rules)
        routed = (rows["source"] == MODEL).to_numpy()
        monitored = features.loc[routed].copy()
        for demand in DEMAND_COLUMNS:
            monitored[f"{demand}_forecast"] = rows.loc[routed, f"{demand}_forecast"].to_numpy()
            monitored[f"{demand}_baseline"] = fallback_forecast(monitored, demand).to_numpy()

        with self._lock:
            self.pending.append(monitored[["station_key", "target_hour", "horizon",
                                           "trips_per_day_28d"]
                                          + [f"{d}_{kind}" for d in DEMAND_COLUMNS
                                             for kind in ("forecast", "baseline")]])
            if self.profile:
                for column, counts in drift_counts(monitored, self.profile).items():
                    self.day_counts[column] = self.day_counts.get(column, 0) + counts
            self.clock = origin
        CLOCK.set(origin.tz_localize("UTC").timestamp())
        HOURS.inc()
        if origin.hour == 23:
            self._close_day(origin.floor("D"), current.version)

    def _close_day(self, day: pd.Timestamp, version: str) -> None:
        """Score every forecast whose target hour is now observed and update the window."""
        last_hour = day + pd.Timedelta(hours=23)
        with self._lock:
            pending = pd.concat(self.pending, ignore_index=True)
            due = pending[pending["target_hour"] <= last_hour]
            self.pending = [pending[pending["target_hour"] > last_hour]]
            counts, self.day_counts = self.day_counts, {}
        errors = (error_sums(self._with_actuals(due), self.tier_bounds) if not due.empty
                  else pd.DataFrame())
        record = DayRecord(day, version, errors, counts)
        with self._lock:
            self.records.append(record)
            metrics = window_metrics(list(self.records), self.profile)
            full = len(self.records) == self.window_days
            if full:
                self._update_streaks(metrics)
            self.latest = metrics
            self.history.append(self._history_row(day, metrics, full))
        self._publish(metrics, full)

    def _with_actuals(self, due: pd.DataFrame) -> pd.DataFrame:
        """Join observed demand at each target hour. A station with no panel row at that
        hour has closed; offline its target is missing too, so the row is not scored."""
        panel = read_panel(self.panel_path, due["target_hour"].min(), due["target_hour"].max())
        actual = panel.rename(columns={"hour": "target_hour", "pickups": "pickups_target",
                                       "dropoffs": "dropoffs_target"})
        return due.merge(actual, on=["station_key", "target_hour"], how="inner")

    # Signals -----------------------------------------------------------------------------

    def _breaches(self, metrics: dict) -> dict[str, bool]:
        overall = metrics["errors"].xs(("all", "all"), level=["group", "label"])
        return {
            "drift": bool(metrics["max_psi"] > self.thresholds.max_psi),
            "error": bool(overall["relative_wape"].max() > self.thresholds.relative_wape),
        }

    def _update_streaks(self, metrics: dict) -> None:
        for signal, breached in self._breaches(metrics).items():
            self.streaks[signal] = self.streaks[signal] + 1 if breached else 0

    def _publish(self, metrics: dict, full: bool) -> None:
        WINDOW_DAYS.set(metrics["days"])
        if not full:
            return  # a partial window would make the first days look noisy
        for (target, group, label), row in metrics["errors"].iterrows():
            WAPE.labels(target, "model", group, label).set(row["model_wape"])
            WAPE.labels(target, "historical_average", group, label).set(row["baseline_wape"])
            RELATIVE.labels(target, group, label).set(row["relative_wape"])
        for column, value in metrics["psi"].items():
            PSI.labels(column).set(value)
        MAX_PSI.set(metrics["max_psi"])
        for signal, days in self.streaks.items():
            STREAK.labels(signal).set(days)

    def _history_row(self, day: pd.Timestamp, metrics: dict, full: bool) -> dict:
        row = {"day": day.strftime("%Y-%m-%d"), "model_version": self.version,
               "window_full": full, "max_psi": None, "drift_days": self.streaks["drift"],
               "error_days": self.streaks["error"]}
        if full:
            overall = metrics["errors"].xs(("all", "all"), level=["group", "label"])
            row["max_psi"] = None if np.isnan(metrics["max_psi"]) else metrics["max_psi"]
            for target, values in overall.iterrows():
                for column in ("model_wape", "baseline_wape", "relative_wape"):
                    row[f"{target}_{column}"] = float(values[column])
        return row

    # Reporting ---------------------------------------------------------------------------

    def status(self) -> dict:
        with self._lock:
            sustain = self.thresholds.sustain_days
            return {
                "running": self.running, "paused": self.paused, "first": stamp(self.first), "last": stamp(self.last),
                "clock": stamp(self.clock), "delay_seconds": self.delay_seconds,
                "model_version": self.version, "profile_loaded": self.profile is not None,
                "window_days": len(self.records), "window_size": self.window_days,
                "breach_days": dict(self.streaks),
                "alerting": {signal: days >= sustain for signal, days in self.streaks.items()},
                "thresholds": self.thresholds.__dict__, "error": self.error,
            }

    def summary(self) -> dict:
        """Status, the latest window in detail, and one row per closed day."""
        with self._lock:
            latest = None
            if self.latest and len(self.records) == self.window_days:
                table = self.latest["errors"].reset_index()
                latest = {
                    "first_day": self.latest["first_day"].strftime("%Y-%m-%d"),
                    "last_day": self.latest["last_day"].strftime("%Y-%m-%d"),
                    "errors": table.replace({np.nan: None}).to_dict("records"),
                    "psi": {key: (None if np.isnan(value) else value)
                            for key, value in self.latest["psi"].items()},
                }
            history = list(self.history)
        return {**self.status(), "latest": latest, "history": history}


def wait_until_idle(monitor: ReplayMonitor, timeout: float) -> None:
    """Block until the replay thread finishes (used by tests and scripts)."""
    deadline = time.monotonic() + timeout
    while monitor.running and time.monotonic() < deadline:
        time.sleep(0.05)
