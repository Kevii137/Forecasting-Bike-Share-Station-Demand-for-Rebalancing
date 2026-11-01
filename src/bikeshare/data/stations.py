"""Build the station table: one key per physical station, with capacity where known.

Raw IDs and names are both unreliable alone. IDs get reused for different sites, and one
site can appear under several labels. Names are normalised, then two names are merged
only when they share an ID and sit close together, which separates renames from moves.
"""
import json
import logging
import math
import re
from pathlib import Path

import pandas as pd
from pyspark.sql import functions as F

from bikeshare.config import load_params
from bikeshare.spark import get_spark

log = logging.getLogger(__name__)

TRIPS_PATH = "data/interim/trips"
STATION_INFO_PATH = Path("data/raw/station_information.json")
OUTPUT_PATH = Path("data/interim/stations.parquet")
NAME_MAP_PATH = Path("data/interim/station_name_map.parquet")
REPORT_PATH = Path("reports/stations.json")
ARCHIVE_ID_PREFIX = "(ARCHIVE)"


def normalise_name(name: str) -> str:
    """Remove labels that describe a station's status rather than its location."""
    cleaned = re.sub(r"^\[Archived\]\s*", "", name.strip())
    cleaned = re.sub(r"\s*\(Temp\)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\*$", "", cleaned)
    cleaned = re.sub(r"\s+-?\s*\d$", "", cleaned)
    return cleaned.strip()


def distance_metres(lat_a: float, lng_a: float, lat_b: float, lng_b: float) -> float:
    """Approximate distance, accurate enough for points within one city."""
    metres_per_degree = 111_320
    north = (lat_a - lat_b) * metres_per_degree
    east = (lng_a - lng_b) * metres_per_degree * math.cos(math.radians(lat_a))
    return math.hypot(north, east)


def group_names(pairs: pd.DataFrame, merge_distance_metres: float) -> dict[str, str]:
    """Map each normalised name to its group's busiest name.

    pairs needs columns norm_name, station_id, trips, lat and lng.
    """
    by_name = pairs.groupby("norm_name").agg(
        trips=("trips", "sum"), lat=("lat", "median"), lng=("lng", "median")
    )
    parent = {name: name for name in by_name.index}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for _, group in pairs.groupby("station_id"):
        names = list(group["norm_name"].unique())
        for index, first in enumerate(names):
            for second in names[index + 1:]:
                a, b = by_name.loc[first], by_name.loc[second]
                if distance_metres(a.lat, a.lng, b.lat, b.lng) <= merge_distance_metres:
                    parent[find(first)] = find(second)

    groups = by_name.assign(root=[find(name) for name in by_name.index])
    busiest = groups.reset_index().sort_values("trips", ascending=False).groupby("root").first()
    return {name: busiest.loc[root, "norm_name"] for name, root in groups["root"].items()}


def collect_station_pairs(trips) -> pd.DataFrame:
    """Summarise every raw station label from both ends of each trip."""
    ends = [
        trips.select(
            F.col(f"{end}_station_name").alias("raw_name"),
            F.col(f"{end}_station_id").alias("raw_id"),
            F.col(f"{end}_lat").alias("lat"),
            F.col(f"{end}_lng").alias("lng"),
            F.col("started_at"),
        )
        for end in ("start", "end")
    ]
    return (
        ends[0].unionByName(ends[1]).dropna(subset=["raw_name", "raw_id"])
        .groupBy("raw_name", "raw_id")
        .agg(
            F.count(F.lit(1)).alias("trips"),
            F.percentile_approx("lat", 0.5).alias("lat"),
            F.percentile_approx("lng", 0.5).alias("lng"),
            F.min("started_at").alias("first_seen"),
            F.max("started_at").alias("last_seen"),
        )
        .toPandas()
    )


def load_station_info() -> pd.DataFrame:
    stations = json.loads(STATION_INFO_PATH.read_text())["data"]["stations"]
    info = pd.DataFrame(stations)[["name", "short_name", "capacity", "station_type"]]
    return info.assign(norm_name=info["name"].map(normalise_name))


def build_station_table(pairs: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    """Create one row per station_key and attach capacity from the station snapshot."""
    by_name = info.drop_duplicates("norm_name").set_index("norm_name")
    by_id = info.dropna(subset=["short_name"]).drop_duplicates("short_name")
    by_id = by_id.set_index("short_name")

    rows = []
    for station_key, group in pairs.groupby("station_key"):
        ordered = group.sort_values("trips", ascending=False)
        match = None
        for name in ordered["norm_name"]:
            if name in by_name.index:
                match = by_name.loc[name]
                break
        if match is None:
            for station_id in ordered["station_id"]:
                if station_id in by_id.index:
                    match = by_id.loc[station_id]
                    break
        rows.append({
            "station_key": station_key,
            "lat": ordered["lat"].iloc[0],
            "lng": ordered["lng"].iloc[0],
            "trips": int(group["trips"].sum()),
            "first_seen": group["first_seen"].min(),
            "last_seen": group["last_seen"].max(),
            "capacity": None if match is None else float(match["capacity"]),
            "station_type": "unknown" if match is None else match["station_type"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    merge_distance = load_params()["stations"]["merge_distance_metres"]

    spark = get_spark("build-stations")
    pairs = collect_station_pairs(spark.read.parquet(TRIPS_PATH))
    spark.stop()

    pairs["station_id"] = pairs["raw_id"].str.replace(ARCHIVE_ID_PREFIX, "", regex=False)
    pairs["norm_name"] = pairs["raw_name"].map(normalise_name)
    pairs["station_key"] = pairs["norm_name"].map(group_names(pairs, merge_distance))

    stations = build_station_table(pairs, load_station_info())
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Spark 3.5 cannot read the nanosecond timestamps pandas writes by default.
    stations.to_parquet(OUTPUT_PATH, index=False, coerce_timestamps="us")
    pairs[["raw_name", "station_key"]].drop_duplicates("raw_name").to_parquet(
        NAME_MAP_PATH, index=False
    )

    trips = stations["trips"]
    report = {
        "raw_labels": len(pairs),
        "stations": len(stations),
        "with_capacity": int(stations["capacity"].notna().sum()),
        "trip_share_with_capacity": round(
            float(trips[stations["capacity"].notna()].sum() / trips.sum()), 4
        ),
        "by_type": stations["station_type"].value_counts().to_dict(),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    log.info("Stations: %s", report)


if __name__ == "__main__":
    main()
