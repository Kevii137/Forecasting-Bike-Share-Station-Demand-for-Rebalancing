"""Calibrate drift and error thresholds from the offline features (Decision D9).

The live monitor reduces each replayed day to a DayRecord. This script builds the same
records from the Spark feature table for every month after training, so the rolling
metrics can be inspected before choosing thresholds:

- validation and test months (January, February): the model is known to be fit for use,
  so these show normal variation;
- replay months (March to September): what the live monitor will see.

Online features equal the offline ones bit for bit (tests/test_serving.py), so these
series are what the replay computes, without running the service for an hour.

    PYTHONPATH=src python -m bikeshare.monitoring.calibrate
"""
import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib
import mlflow
import mlflow.pyfunc  # explicit: mlflow exposes pyfunc through a lazy proxy
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from bikeshare.config import load_params, month_range
from bikeshare.features import DEMAND_COLUMNS
from bikeshare.modeling.dataset import (
    FEATURE_COLUMNS,
    FEATURES_PATH,
    KEY_COLUMNS,
    TARGETS,
)
from bikeshare.modeling.tracking import start_tracking
from bikeshare.monitoring.profile import METRICS_PATH, PROFILE_PATH, logged_model_uri
from bikeshare.monitoring.windows import (
    DayRecord,
    drift_counts,
    error_sums,
    window_metrics,
)
from bikeshare.serving.forecast import fallback_forecast

matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

REPORT_PATH = Path("reports/monitoring_calibration.json")
PLOT_PATH = Path("reports/monitoring_calibration.png")
FALLBACK_COLUMNS = [f"{demand}_{suffix}" for demand in DEMAND_COLUMNS
                    for suffix in ("same_hour_8w_mean", "same_hour_4w_mean", "mean_168h")]


def scored_month(month: str, model) -> pd.DataFrame:
    """Feature rows of one month with model forecasts, baseline forecasts and targets."""
    columns = list(dict.fromkeys(KEY_COLUMNS + FEATURE_COLUMNS + FALLBACK_COLUMNS
                                 + list(TARGETS.values())))
    frame = pd.read_parquet(FEATURES_PATH, columns=columns, filters=[("month", "=", month)])
    predicted = model.predict(frame[FEATURE_COLUMNS].astype("float64"))
    for demand in DEMAND_COLUMNS:
        frame[f"{demand}_forecast"] = predicted[f"{demand}_forecast"].to_numpy()
        frame[f"{demand}_baseline"] = fallback_forecast(frame, demand).to_numpy()
    return frame


def daily_records(months: list[str], model, profile: dict, tier_bounds: list[float],
                  version: str) -> list[DayRecord]:
    errors: dict[pd.Timestamp, list[pd.DataFrame]] = defaultdict(list)
    counts: dict[pd.Timestamp, dict] = {}
    for month in months:
        frame = scored_month(month, model)
        log.info("Scored %s: %d rows", month, len(frame))
        observed = frame.dropna(subset=list(TARGETS.values()))
        for day, rows in observed.groupby(observed["target_hour"].dt.floor("D")):
            errors[day].append(error_sums(rows, tier_bounds))
        for day, rows in frame.groupby(frame["origin_hour"].dt.floor("D")):
            counts[day] = drift_counts(rows, profile)
    days = sorted(set(counts) & set(errors))
    return [DayRecord(day, version,
                      pd.concat(errors[day]).groupby(level=["target", "group", "label"]).sum(),
                      counts[day]) for day in days]


def rolling_series(records: list[DayRecord], profile: dict, window: int) -> pd.DataFrame:
    rows = []
    for end in range(window, len(records) + 1):
        metrics = window_metrics(records[end - window:end], profile)
        table = metrics["errors"]
        row = {"day": metrics["last_day"], "max_psi": metrics["max_psi"],
               **{f"psi_{column}": value for column, value in metrics["psi"].items()}}
        for (target, group, label), values in table.iterrows():
            prefix = target if group == "all" else f"{target}_{group}_{label}"
            row[f"{prefix}_model_wape"] = values["model_wape"]
            row[f"{prefix}_baseline_wape"] = values["baseline_wape"]
            row[f"{prefix}_relative_wape"] = values["relative_wape"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("day")


def periods(params: dict) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Window end days by period. Training-window errors are in-sample, so only their PSI is
    used: it shows how far a week can sit from the reference while the model still fits it."""
    split = params["split"]
    def bounds(months: list[str]) -> tuple[pd.Timestamp, pd.Timestamp]:
        return pd.Timestamp(months[0]), pd.Period(months[1], "M").end_time.floor("D")
    return {"training": bounds(split["train"]),
            "normal": bounds([split["validation"][0], split["test"][1]]),
            "replay": bounds(split["replay"])}


def worst_relative_wape(series: pd.DataFrame) -> pd.Series:
    return series[[f"{demand}_relative_wape" for demand in DEMAND_COLUMNS]].max(axis=1)


def simulate_alerts(series: pd.DataFrame, thresholds: dict) -> dict[str, list[dict]]:
    """Alert episodes the live monitor would raise: a signal breaches on consecutive days,
    and fires on the day the streak reaches sustain_days. Mirrors ReplayMonitor."""
    breaches = {"drift": series["max_psi"] > thresholds["max_psi"],
                "error": worst_relative_wape(series) > thresholds["relative_wape"]}
    episodes = {}
    for signal, breached in breaches.items():
        runs, streak, start = [], 0, None
        for day, value in breached.items():
            streak = streak + 1 if value else 0
            if streak == 1:
                start = day
            if streak == thresholds["sustain_days"]:
                runs.append({"breach_from": start.strftime("%Y-%m-%d"),
                             "fires": day.strftime("%Y-%m-%d")})
            if streak >= thresholds["sustain_days"]:
                runs[-1]["last_day"] = day.strftime("%Y-%m-%d")
        episodes[signal] = runs
    return episodes


def plot(series: pd.DataFrame, spans: dict, thresholds: dict) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for demand, colour in zip(DEMAND_COLUMNS, ("#3b6ea5", "#c0504d")):
        axes[0].plot(series.index, series[f"{demand}_model_wape"], color=colour,
                     label=f"{demand}: model")
        axes[0].plot(series.index, series[f"{demand}_baseline_wape"], color=colour,
                     linestyle="--", label=f"{demand}: historical average")
        axes[1].plot(series.index, series[f"{demand}_relative_wape"], color=colour, label=demand)
    axes[0].set_ylabel("7-day WAPE")
    axes[1].axhline(thresholds["relative_wape"], color="black", linewidth=0.8,
                    label="error threshold")
    axes[1].set_ylabel("Model WAPE / baseline WAPE")
    for column in [c for c in series.columns if c.startswith("psi_")]:
        axes[2].plot(series.index, series[column], label=column.removeprefix("psi_"))
    axes[2].axhline(thresholds["max_psi"], color="black", linewidth=0.8, label="drift threshold")
    axes[2].set_ylabel("7-day PSI vs training reference")
    axes[2].set_yscale("log")
    first, last = spans["training"]
    for axis in axes:
        axis.axvspan(first, last, color="grey", alpha=0.12, linewidth=0)
        axis.axvline(spans["replay"][0], color="black", linestyle=":", linewidth=1)
        axis.legend(fontsize=7, loc="upper right")
    axes[0].set_title("Rolling 7-day monitoring metrics. Shaded: training window (errors "
                      "in-sample). Dotted: replay starts.", fontsize=10)
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=110)
    plt.close(figure)


def describe(frame: pd.DataFrame) -> dict:
    return {column: {"median": round(float(np.nanmedian(values)), 4),
                     "p95": round(float(np.nanquantile(values, 0.95)), 4),
                     "max": round(float(np.nanmax(values)), 4)}
            for column, values in frame.items()}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    params = load_params()
    monitoring = params["monitoring"]
    start_tracking()
    run_id = json.loads(METRICS_PATH.read_text())["lightgbm_run_id"]
    model = mlflow.pyfunc.load_model(logged_model_uri(run_id, MlflowClient()))
    profile = json.loads(PROFILE_PATH.read_text())

    split = params["split"]
    months = month_range(split["train"][0], split["replay"][1])
    records = daily_records(months, model, profile, params["evaluation"]["tier_bounds"], run_id)
    series = rolling_series(records, profile, monitoring["window_days"])
    series["worst_relative_wape"] = worst_relative_wape(series)
    spans = periods(params)
    thresholds = monitoring["thresholds"]
    plot(series, spans, thresholds)

    report = {
        "run_id": run_id,
        "window_days": monitoring["window_days"],
        "thresholds": thresholds,
        "periods": {name: {"from": first.strftime("%Y-%m-%d"), "to": last.strftime("%Y-%m-%d"),
                           **describe(series.loc[first:last])}
                    for name, (first, last) in spans.items()},
        "replay_alerts": simulate_alerts(series.loc[spans["replay"][0]:], thresholds),
        "series": {day.strftime("%Y-%m-%d"): {key: (None if pd.isna(value) else round(value, 4))
                                             for key, value in row.items()}
                   for day, row in series.iterrows()},
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    log.info("Replay alerts with the configured thresholds: %s", report["replay_alerts"])


if __name__ == "__main__":
    main()
