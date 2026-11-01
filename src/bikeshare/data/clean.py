"""Validate raw trips with explicit rules and write the clean trip table."""
import gzip
import json
import logging
import time
from pathlib import Path

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from bikeshare.config import load_params
from bikeshare.spark import get_spark

log = logging.getLogger(__name__)

RAW_GLOB = "data/raw/trips/*.csv.gz"
OUTPUT_PATH = "data/interim/trips"
REPORT_PATH = Path("reports/data_quality.json")
TIMESTAMP_FORMAT = "yyyy-MM-dd HH:mm:ss.SSS"
EXPECTED_COLUMNS = [
    "ride_id", "rideable_type", "started_at", "ended_at", "start_station_name",
    "start_station_id", "end_station_name", "end_station_id", "start_lat", "start_lng",
    "end_lat", "end_lng", "member_casual",
]
MIN_DURATION_SECONDS = 60
MAX_DURATION_SECONDS = 24 * 3600


def check_schema(paths: list[Path]) -> None:
    """Stop the pipeline if any file's header differs from the expected schema."""
    for path in paths:
        with gzip.open(path, "rt") as handle:
            columns = handle.readline().strip().replace('"', "").split(",")
        if columns != EXPECTED_COLUMNS:
            raise ValueError(f"unexpected schema in {path}: {columns}")


def rejection_rules(duration: Column) -> dict[str, Column]:
    """Return each rejection rule as a condition that is true for bad rows."""
    return {
        "missing_ride_id": F.col("ride_id").isNull(),
        "bad_timestamp": F.col("started_ts").isNull() | F.col("ended_ts").isNull(),
        "negative_duration": duration < 0,
        "under_60_seconds": (duration >= 0) & (duration < MIN_DURATION_SECONDS),
        "over_24_hours": duration > MAX_DURATION_SECONDS,
    }


def apply_rules(trips: DataFrame) -> tuple[DataFrame, dict[str, int]]:
    """Drop rows failing any rule and return row counts for each rule."""
    parsed = trips.withColumns({
        "started_ts": F.to_timestamp("started_at", TIMESTAMP_FORMAT),
        "ended_ts": F.to_timestamp("ended_at", TIMESTAMP_FORMAT),
    })
    duration = F.col("ended_ts").cast("long") - F.col("started_ts").cast("long")
    rules = rejection_rules(duration)
    counts = parsed.agg(
        F.count(F.lit(1)).alias("rows_read"),
        *[F.sum(F.coalesce(rule, F.lit(False)).cast("long")).alias(name)
          for name, rule in rules.items()],
    ).first().asDict()

    any_rejected = F.lit(False)
    for rule in rules.values():
        any_rejected = any_rejected | F.coalesce(rule, F.lit(False))
    return parsed.filter(~any_rejected), counts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = time.perf_counter()
    max_rejected_share = load_params()["validation"]["max_rejected_share"]
    check_schema(sorted(Path("data/raw/trips").glob("*.csv.gz")))

    spark = get_spark("clean-trips")
    raw = spark.read.csv(RAW_GLOB, header=True)
    valid, counts = apply_rules(raw)
    clean = (
        valid.dropDuplicates(["ride_id"])
        .select(
            "ride_id", "rideable_type",
            F.col("started_ts").alias("started_at"),
            F.col("ended_ts").alias("ended_at"),
            "start_station_name", "start_station_id", "end_station_name", "end_station_id",
            F.col("start_lat").cast("double"), F.col("start_lng").cast("double"),
            F.col("end_lat").cast("double"), F.col("end_lng").cast("double"),
        )
        .withColumn("month", F.date_format("started_at", "yyyy-MM"))
    )
    clean.write.mode("overwrite").partitionBy("month").parquet(OUTPUT_PATH)

    rows_written = spark.read.parquet(OUTPUT_PATH).count()
    spark.stop()
    seconds = time.perf_counter() - started
    rejected_share = 1 - rows_written / counts["rows_read"]
    report = {
        **counts,
        "rows_written": rows_written,
        "rejected_share": round(rejected_share, 5),
        "seconds": round(seconds, 1),
        "rows_per_second": round(counts["rows_read"] / seconds),
    }
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    log.info("Clean trips: %s", report)
    if rejected_share > max_rejected_share:
        raise ValueError(f"rejected share {rejected_share:.3f} above {max_rejected_share}")


if __name__ == "__main__":
    main()
