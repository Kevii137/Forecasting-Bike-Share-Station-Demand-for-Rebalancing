"""Turn feature rows into station forecasts and demand-pressure alerts.

Decision D2: eligible classic stations use the model. Every other active station gets the
historical-average forecast, is marked low confidence, and raises no alert.

Decision D1: alerts rank stations by forecast net flow over the horizon divided by
capacity. Net flow is drop-offs minus pickups, so a negative value means bikes are
leaving faster than they arrive ("likely draining").
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bikeshare.features import DEMAND_COLUMNS
from bikeshare.modeling.dataset import FEATURE_COLUMNS

MODEL = "model"
FALLBACK = "historical_average"
# The fallback is the historical-average baseline. Newer stations lack 8 weeks of
# history, so shorter averages stand in, then zero.
FALLBACK_ORDER = ("same_hour_8w_mean", "same_hour_4w_mean", "mean_168h")


@dataclass(frozen=True)
class Rules:
    min_trips_per_day: float
    allowed_types: tuple[str, ...]
    alert_limit: int
    min_pressure: float


def uses_model(features: pd.DataFrame, rules: Rules) -> pd.Series:
    return features["station_type"].isin(rules.allowed_types) & (
        features["trips_per_day_28d"] >= rules.min_trips_per_day
    )


def fallback_forecast(features: pd.DataFrame, demand: str) -> pd.Series:
    forecast = pd.Series(np.nan, index=features.index)
    for suffix in FALLBACK_ORDER:
        forecast = forecast.fillna(features[f"{demand}_{suffix}"])
    return forecast.fillna(0.0).astype("float64")


def forecast_rows(features: pd.DataFrame, model, rules: Rules) -> pd.DataFrame:
    """One row per station and horizon with pickup and drop-off forecasts."""
    routed = uses_model(features, rules)
    result = features[["station_key", "origin_hour", "horizon", "target_hour"]].copy()
    result["source"] = np.where(routed, MODEL, FALLBACK)
    for demand in DEMAND_COLUMNS:
        result[f"{demand}_forecast"] = fallback_forecast(features, demand)
    if routed.any():
        # The logged signature is float64 for every input, as in training.
        predicted = model.predict(features.loc[routed, FEATURE_COLUMNS].astype("float64"))
        for demand in DEMAND_COLUMNS:
            result.loc[routed, f"{demand}_forecast"] = predicted[f"{demand}_forecast"].to_numpy()
    return result


def station_summary(features: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Totals over all horizons per station, with demand pressure."""
    totals = rows.groupby("station_key").agg(
        pickups=("pickups_forecast", "sum"),
        dropoffs=("dropoffs_forecast", "sum"),
        source=("source", "first"),
    )
    details = features.drop_duplicates("station_key").set_index("station_key")[
        ["lat", "lng", "capacity", "station_type", "trips_per_day_28d"]
    ]
    summary = totals.join(details)
    summary["net_flow"] = summary["dropoffs"] - summary["pickups"]
    capacity = summary["capacity"].where(summary["capacity"] > 0)
    summary["pressure"] = summary["net_flow"] / capacity
    summary["confidence"] = np.where(summary["source"] == MODEL, "normal", "low")
    return summary.reset_index()


def alerts(summary: pd.DataFrame, rules: Rules) -> dict[str, pd.DataFrame]:
    """Stations most likely to drain or fill, strongest pressure first."""
    eligible = summary[(summary["source"] == MODEL) & summary["pressure"].notna()]
    draining = eligible[eligible["pressure"] <= -rules.min_pressure].nsmallest(
        rules.alert_limit, "pressure")
    filling = eligible[eligible["pressure"] >= rules.min_pressure].nlargest(
        rules.alert_limit, "pressure")
    return {"draining": draining, "filling": filling}
