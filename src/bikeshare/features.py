"""Forecast features, shared by the Spark pipeline and the online service (Decision D6).

Terms used throughout:
- origin: the last complete hour when a forecast is made.
- target hour: origin + horizon, the hour being forecast.

Every feature uses only data up to and including the origin hour.
"""
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

HOURS_PER_WEEK = 168
TRAILING_DAYS = 28
DEMAND_COLUMNS = ("pickups", "dropoffs")


def _us_holidays(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return USFederalHolidayCalendar().holidays(start=start.floor("D"), end=end.ceil("D"))


def _weekly_mean(series: pd.Series, horizon: int, weeks: int) -> pd.Series:
    """Mean demand at the target hour's weekday and hour over the previous weeks."""
    shifted = [series.shift(HOURS_PER_WEEK * week - horizon) for week in range(1, weeks + 1)]
    return pd.concat(shifted, axis=1).mean(axis=1, skipna=False)


def _features_for_horizon(hourly: pd.DataFrame, horizon: int) -> pd.DataFrame:
    frame = pd.DataFrame(index=hourly.index)
    frame["horizon"] = horizon
    target_hour = hourly.index + pd.Timedelta(hours=horizon)
    frame["target_hour"] = target_hour

    for column in DEMAND_COLUMNS:
        series = hourly[column]
        for lag in range(3):
            frame[f"{column}_lag_{lag}"] = series.shift(lag)
        for window in (3, 24, HOURS_PER_WEEK):
            frame[f"{column}_mean_{window}h"] = series.rolling(window).mean()
        frame[f"{column}_same_hour_1d"] = series.shift(24 - horizon)
        frame[f"{column}_same_hour_1w"] = series.shift(HOURS_PER_WEEK - horizon)
        frame[f"{column}_same_hour_4w_mean"] = _weekly_mean(series, horizon, 4)
        frame[f"{column}_same_hour_8w_mean"] = _weekly_mean(series, horizon, 8)
        frame[f"{column}_target"] = series.shift(-horizon)

    trailing_pickups = hourly["pickups"].rolling(TRAILING_DAYS * 24, min_periods=1).sum()
    frame["trips_per_day_28d"] = trailing_pickups / TRAILING_DAYS

    holidays = _us_holidays(target_hour.min(), target_hour.max())
    frame["target_hour_of_day"] = target_hour.hour
    frame["target_day_of_week"] = target_hour.dayofweek
    frame["target_is_weekend"] = (target_hour.dayofweek >= 5).astype(int)
    frame["target_is_holiday"] = target_hour.floor("D").isin(holidays).astype(int)
    return frame


def build_features(history: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Build one row per origin hour and horizon for a single station.

    history needs columns station_key, hour, pickups and dropoffs. Target columns are
    NaN for origins whose target hour has not happened yet.
    """
    if history["station_key"].nunique() != 1:
        raise ValueError("build_features expects exactly one station")
    station_key = history["station_key"].iloc[0]
    hourly = (
        history.assign(hour=pd.to_datetime(history["hour"]))
        .set_index("hour")
        .sort_index()[list(DEMAND_COLUMNS)]
        .astype("float64")
    )
    full_range = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
    hourly = hourly.reindex(full_range, fill_value=0.0)

    frames = [_features_for_horizon(hourly, horizon) for horizon in horizons]
    features = pd.concat(frames).rename_axis("origin_hour").reset_index()
    return features.assign(station_key=station_key)
