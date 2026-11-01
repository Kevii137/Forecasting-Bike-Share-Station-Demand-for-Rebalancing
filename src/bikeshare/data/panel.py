"""Aggregate clean trips into a dense hourly panel, one row per station and hour."""
import json
import logging
import time
from pathlib import Path

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from bikeshare.spark import get_spark

log = logging.getLogger(__name__)

TRIPS_PATH = "data/interim/trips"
NAME_MAP_PATH = "data/interim/station_name_map.parquet"
STATIONS_PATH = "data/interim/stations.parquet"
OUTPUT_PATH = "data/processed/panel"
REPORT_PATH = Path("reports/panel.json")
SIGNAL_WINDOWS_HOURS = (3, 6)


def hourly_counts(trips: DataFrame, name_map: DataFrame, end: str, column: str) -> DataFrame:
    """Count trips per station and hour for the start or end of each trip."""
    time_column = "started_at" if end == "start" else "ended_at"
    return (
        trips.join(F.broadcast(name_map), trips[f"{end}_station_name"] == name_map["raw_name"])
        .groupBy("station_key", F.date_trunc("hour", time_column).alias("hour"))
        .agg(F.count(F.lit(1)).alias(column))
    )


def dense_panel(pickups: DataFrame, dropoffs: DataFrame) -> DataFrame:
    """Fill zero-demand hours, only between each station's first and last activity.

    Outside that span a station may not exist, so a zero would be false.
    """
    counts = pickups.join(dropoffs, ["station_key", "hour"], "full").fillna(0)
    spans = counts.groupBy("station_key").agg(
        F.min("hour").alias("first_hour"), F.max("hour").alias("last_hour")
    )
    hours = spans.select(
        "station_key",
        F.explode(
            F.expr("sequence(first_hour, last_hour, interval 1 hour)")
        ).alias("hour"),
    )
    return hours.join(counts, ["station_key", "hour"], "left").fillna(0)


def missing_station_shares(trips: DataFrame) -> dict[str, dict[str, float]]:
    """Report by month the share of trips excluded for lacking a station (Decision D2)."""
    rows = trips.groupBy("month").agg(
        F.mean(F.col("start_station_name").isNull().cast("double")).alias("start"),
        F.mean(F.col("end_station_name").isNull().cast("double")).alias("end"),
    ).orderBy("month").collect()
    return {row["month"]: {"start": round(row["start"], 4), "end": round(row["end"], 4)}
            for row in rows}


def net_flow_signal(panel: DataFrame, stations: DataFrame, hours: int) -> dict[str, float]:
    """Measure how often net flow over a window is large relative to capacity."""
    window = Window.partitionBy("station_key").orderBy("hour").rowsBetween(-(hours - 1), 0)
    with_capacity = panel.join(
        F.broadcast(stations.filter("station_type = 'classic' AND capacity > 0")), "station_key"
    )
    ratio = F.abs(F.sum(F.col("dropoffs") - F.col("pickups")).over(window)) / F.col("capacity")
    row = with_capacity.select(ratio.alias("ratio")).agg(
        F.mean((F.col("ratio") >= 0.5).cast("double")).alias("at_least_half"),
        F.mean((F.col("ratio") >= 1.0).cast("double")).alias("at_least_full"),
    ).first()
    return {"share_at_least_half_capacity": round(row["at_least_half"], 4),
            "share_at_least_full_capacity": round(row["at_least_full"], 4)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = time.perf_counter()
    spark = get_spark("build-panel")
    trips = spark.read.parquet(TRIPS_PATH)
    name_map = spark.read.parquet(NAME_MAP_PATH)

    pickups = hourly_counts(trips, name_map, "start", "pickups")
    dropoffs = hourly_counts(trips, name_map, "end", "dropoffs")
    panel = dense_panel(pickups, dropoffs).withColumn("month", F.date_format("hour", "yyyy-MM"))
    panel.write.mode("overwrite").partitionBy("month").parquet(OUTPUT_PATH)

    written = spark.read.parquet(OUTPUT_PATH)
    stations = spark.read.parquet(STATIONS_PATH)
    report = {
        "rows": written.count(),
        "stations": written.select("station_key").distinct().count(),
        "zero_hour_share": round(
            written.agg(F.mean(
                ((F.col("pickups") + F.col("dropoffs")) == 0).cast("double")
            )).first()[0], 4
        ),
        "missing_station_share_by_month": missing_station_shares(trips),
        "net_flow_signal": {
            f"{hours}h": net_flow_signal(written, stations, hours)
            for hours in SIGNAL_WINDOWS_HOURS
        },
    }
    spark.stop()
    report["seconds"] = round(time.perf_counter() - started, 1)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    log.info("Panel: %s", report)


if __name__ == "__main__":
    main()
