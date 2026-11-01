"""Tests for Spark cleaning rules and zero filling."""
from datetime import datetime

import pandas as pd
import pytest

from bikeshare.data.clean import apply_rules
from bikeshare.data.panel import dense_panel
from bikeshare.spark import get_spark


@pytest.fixture(scope="module")
def spark():
    session = get_spark("tests", master="local[2]", driver_memory="1g")
    yield session
    session.stop()


def test_spark_reads_timestamps_written_the_way_stations_are(spark, tmp_path):
    path = tmp_path / "stations.parquet"
    frame = pd.DataFrame({"first_seen": pd.to_datetime(["2026-01-01 08:00:00.123"])})
    frame.to_parquet(path, index=False, coerce_timestamps="us")
    assert spark.read.parquet(str(path)).count() == 1


def test_rules_reject_bad_trips_and_count_each_rule(spark):
    rows = [
        ("good", "2026-01-01 08:00:00.000", "2026-01-01 08:10:00.000"),
        ("short", "2026-01-01 08:00:00.000", "2026-01-01 08:00:30.000"),
        ("reversed", "2026-01-01 08:10:00.000", "2026-01-01 08:00:00.000"),
        ("long", "2026-01-01 08:00:00.000", "2026-01-03 08:00:00.000"),
        ("garbled", "not a time", "2026-01-01 08:00:00.000"),
    ]
    frame = spark.createDataFrame(rows, ["ride_id", "started_at", "ended_at"])
    valid, counts = apply_rules(frame)

    assert [row["ride_id"] for row in valid.collect()] == ["good"]
    assert counts == {
        "rows_read": 5, "missing_ride_id": 0, "bad_timestamp": 1,
        "negative_duration": 1, "under_60_seconds": 1, "over_24_hours": 1,
    }


def test_panel_fills_zeros_only_between_first_and_last_activity(spark):
    hour_8, hour_10 = datetime(2026, 1, 1, 8), datetime(2026, 1, 1, 10)
    pickups = spark.createDataFrame([("A", hour_8, 2)], ["station_key", "hour", "pickups"])
    dropoffs = spark.createDataFrame([("A", hour_10, 1)], ["station_key", "hour", "dropoffs"])

    panel = dense_panel(pickups, dropoffs).orderBy("hour").collect()

    assert [(row["hour"].hour, row["pickups"], row["dropoffs"]) for row in panel] == [
        (8, 2, 0), (9, 0, 0), (10, 0, 1),
    ]
