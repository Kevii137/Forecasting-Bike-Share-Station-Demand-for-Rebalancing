"""Run the shared feature function over every station with Spark applyInPandas."""
import json
import logging
import time
from pathlib import Path

import pandas as pd
from pyspark.sql import functions as F

from bikeshare.config import load_params
from bikeshare.features import DEMAND_COLUMNS, build_features
from bikeshare.spark import get_spark

log = logging.getLogger(__name__)

PANEL_PATH = "data/processed/panel"
STATIONS_PATH = "data/interim/stations.parquet"
OUTPUT_PATH = "data/processed/features"
REPORT_PATH = Path("reports/features.json")


def output_schema() -> str:
    """Declare the columns applyInPandas returns, in build_features order."""
    demand = []
    for column in DEMAND_COLUMNS:
        demand += [f"{column}_lag_{lag} float" for lag in range(3)]
        demand += [f"{column}_mean_{window}h float" for window in (3, 24, 168)]
        demand += [
            f"{column}_same_hour_1d float", f"{column}_same_hour_1w float",
            f"{column}_same_hour_4w_mean float", f"{column}_same_hour_8w_mean float",
            f"{column}_target float",
        ]
    calendar = [
        "trips_per_day_28d float", "target_hour_of_day int", "target_day_of_week int",
        "target_is_weekend int", "target_is_holiday int",
    ]
    keys = ["origin_hour timestamp", "horizon int", "target_hour timestamp", "station_key string"]
    return ", ".join(keys + demand + calendar)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = time.perf_counter()
    params = load_params()
    horizons = params["forecast"]["horizons"]
    rule = params["stations"]
    schema = output_schema()
    columns = [item.split()[0] for item in schema.split(", ")]

    first_origin = pd.Period(params["split"]["train"][0]).to_timestamp()
    min_trips = rule["min_trips_per_day"]

    def station_features(history: pd.DataFrame) -> pd.DataFrame:
        """Filter here so that rows which would be dropped never return to the JVM."""
        features = build_features(history, horizons)
        keep = (features["origin_hour"] >= first_origin) & (
            features["trips_per_day_28d"] >= min_trips
        )
        return features.loc[keep, columns]

    spark = get_spark("build-features")
    panel = spark.read.parquet(PANEL_PATH).drop("month")
    stations = spark.read.parquet(STATIONS_PATH).select(
        "station_key", "lat", "lng", "capacity", "station_type"
    )
    features = (
        panel.groupBy("station_key")
        .applyInPandas(station_features, schema)
        .join(F.broadcast(stations), "station_key")
        .filter(F.col("station_type").isin(rule["allowed_types"]))
        .withColumn("month", F.date_format("origin_hour", "yyyy-MM"))
    )
    features.write.mode("overwrite").partitionBy("month").parquet(OUTPUT_PATH)

    written = spark.read.parquet(OUTPUT_PATH)
    monthly = written.groupBy("month").agg(
        F.count(F.lit(1)).alias("rows"),
        F.countDistinct("station_key").alias("stations"),
    ).orderBy("month").collect()
    spark.stop()
    report = {
        "rows": sum(row["rows"] for row in monthly),
        "by_month": {row["month"]: {"rows": row["rows"], "stations": row["stations"]}
                     for row in monthly},
        "seconds": round(time.perf_counter() - started, 1),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    log.info("Features: %s", report)


if __name__ == "__main__":
    main()
