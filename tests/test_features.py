"""Tests for the shared feature function, including temporal leakage."""
import numpy as np
import pandas as pd
import pytest

from bikeshare.features import build_features

HORIZONS = [1, 2, 3]


def station_history(hours: int = 24 * 70, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "station_key": "A",
        "hour": pd.date_range("2026-01-01", periods=hours, freq="h"),
        "pickups": rng.poisson(3, hours),
        "dropoffs": rng.poisson(2, hours),
    })


def row(features: pd.DataFrame, origin: pd.Timestamp, horizon: int) -> pd.Series:
    match = features[(features["origin_hour"] == origin) & (features["horizon"] == horizon)]
    return match.iloc[0]


def test_changing_data_after_origin_does_not_change_features():
    history = station_history()
    origin = pd.Timestamp("2026-03-05 14:00")
    changed = history.copy()
    after_origin = changed["hour"] > origin
    changed.loc[after_origin, ["pickups", "dropoffs"]] += 1000

    for horizon in HORIZONS:
        original_row = row(build_features(history, HORIZONS), origin, horizon)
        changed_row = row(build_features(changed, HORIZONS), origin, horizon)
        inputs = [name for name in original_row.index if not name.endswith("_target")]
        pd.testing.assert_series_equal(original_row[inputs], changed_row[inputs])


def test_target_is_demand_at_origin_plus_horizon():
    history = station_history()
    features = build_features(history, HORIZONS)
    origin = pd.Timestamp("2026-03-05 14:00")
    actual = history.set_index("hour")["pickups"]
    for horizon in HORIZONS:
        target_hour = origin + pd.Timedelta(hours=horizon)
        assert row(features, origin, horizon)["pickups_target"] == actual[target_hour]


def test_seasonal_features_look_back_from_the_target_hour():
    history = station_history()
    features = build_features(history, HORIZONS)
    origin = pd.Timestamp("2026-03-05 14:00")
    actual = history.set_index("hour")["dropoffs"]
    result = row(features, origin, 3)
    target_hour = origin + pd.Timedelta(hours=3)
    week = pd.Timedelta(hours=168)

    assert result["dropoffs_same_hour_1w"] == actual[target_hour - week]
    assert result["dropoffs_same_hour_1d"] == actual[target_hour - pd.Timedelta(hours=24)]
    expected = np.mean([actual[target_hour - week * count] for count in range(1, 9)])
    assert result["dropoffs_same_hour_8w_mean"] == pytest.approx(expected)


def test_calendar_describes_target_hour_and_holidays():
    history = station_history()
    features = build_features(history, HORIZONS)
    result = row(features, pd.Timestamp("2026-01-18 23:00"), 1)
    assert result["target_hour_of_day"] == 0
    assert result["target_day_of_week"] == 0
    assert result["target_is_holiday"] == 1  # Martin Luther King Jr. Day


def test_unobserved_targets_are_missing_not_zero():
    features = build_features(station_history(), HORIZONS)
    latest = features[features["origin_hour"] == features["origin_hour"].max()]
    assert latest["pickups_target"].isna().all()


def test_rejects_more_than_one_station():
    history = station_history()
    history.loc[0, "station_key"] = "B"
    with pytest.raises(ValueError):
        build_features(history, HORIZONS)
