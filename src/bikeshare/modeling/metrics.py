"""Forecast error metrics and the breakdowns used to compare models."""
from itertools import pairwise

import numpy as np
import pandas as pd

PEAK_LABEL = {True: "peak", False: "off_peak"}


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Total absolute error divided by total demand. Defined for zero-heavy counts."""
    total = np.sum(actual)
    if total == 0:
        return float("nan")
    return float(np.sum(np.abs(actual - predicted)) / total)


def poisson_deviance(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Poisson deviance, the loss the models are trained on."""
    predicted = np.clip(predicted, 1e-9, None)
    # np.where evaluates both branches, so log(0) is computed and then discarded.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(actual > 0, actual * np.log(actual / predicted), 0.0)
    return float(np.mean(2 * (ratio - (actual - predicted))))


def score(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = actual - predicted
    return {
        "wape": wape(actual, predicted),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "poisson_deviance": poisson_deviance(actual, predicted),
    }


def volume_tier(trips_per_day: pd.Series, bounds: list[float]) -> pd.Series:
    """Label stations by recent volume, for example 5-10, 10-20 and 20+ trips a day."""
    edges = [*bounds, np.inf]
    labels = [f"{low:g}+" if high == np.inf else f"{low:g}-{high:g}"
              for low, high in pairwise(edges)]
    return pd.cut(trips_per_day, edges, labels=labels, right=False).astype(str)


def is_peak(frame: pd.DataFrame, peak_hours: list[int]) -> pd.Series:
    weekday = frame["target_is_weekend"] == 0
    return weekday & frame["target_hour_of_day"].isin(peak_hours)


def breakdown(frame: pd.DataFrame, actual: str, predicted: str,
              tier_bounds: list[float], peak_hours: list[int]) -> dict[str, dict]:
    """Score overall and by horizon, station volume tier and peak period."""
    def scored(rows: pd.DataFrame) -> dict[str, float]:
        return {**score(rows[actual].to_numpy(), rows[predicted].to_numpy()), "rows": len(rows)}

    groups = {
        "horizon": frame["horizon"].astype(str),
        "tier": volume_tier(frame["trips_per_day_28d"], tier_bounds),
        "period": is_peak(frame, peak_hours).map(PEAK_LABEL),
    }
    result = {"overall": scored(frame)}
    for name, labels in groups.items():
        result[name] = {label: scored(rows) for label, rows in frame.groupby(labels)}
    return result


def net_flow_error(frame: pd.DataFrame, pickups: str, dropoffs: str) -> dict[str, float]:
    """Error on net flow summed over the full horizon, which drives rebalancing alerts.

    Only origins with every horizon present are scored.
    """
    horizons = frame["horizon"].nunique()
    flows = frame.assign(
        actual=frame["dropoffs_target"] - frame["pickups_target"],
        predicted=frame[dropoffs] - frame[pickups],
    )
    totals = flows.groupby(["station_key", "origin_hour"]).agg(
        actual=("actual", "sum"), predicted=("predicted", "sum"), count=("actual", "size")
    )
    totals = totals[totals["count"] == horizons]
    error = totals["actual"] - totals["predicted"]
    return {"mae": float(error.abs().mean()), "rmse": float(np.sqrt((error**2).mean())),
            "origins": len(totals)}
