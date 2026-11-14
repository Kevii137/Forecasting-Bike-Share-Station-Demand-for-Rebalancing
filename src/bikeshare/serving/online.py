"""Online features: the offline feature function run on recent station history (Decision D6).

The service has no live trip feed, so it replays the hourly panel. For an origin hour it
reads only panel rows up to the end of that day, and every feature uses data up to and
including the origin, exactly as offline.

build_features returns every origin in the history it is given, so one call per station
covers a whole day. Results are cached by day, which makes hourly replay cheap.
"""
import logging
import threading
from collections import OrderedDict
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing import get_context

import pandas as pd

from bikeshare.config import month_range
from bikeshare.features import DEMAND_COLUMNS, build_features

log = logging.getLogger(__name__)

# The 8-week same-hour mean looks back 8 * 168 - 1 = 1,343 hours and the 28-day volume
# 672 hours. With at least this much history, online values equal the offline values.
HISTORY_HOURS = 1400
STATION_COLUMNS = ["station_key", "lat", "lng", "capacity", "station_type"]
TARGET_COLUMNS = [f"{column}_target" for column in DEMAND_COLUMNS]


def _float32_columns() -> list[str]:
    """Columns the Spark stage stores as 32-bit floats. Matching them keeps parity exact."""
    suffixes = ("lag_0", "lag_1", "lag_2", "mean_3h", "mean_24h", "mean_168h",
                "same_hour_1d", "same_hour_1w", "same_hour_4w_mean", "same_hour_8w_mean")
    return [f"{column}_{suffix}" for column in DEMAND_COLUMNS for suffix in suffixes] + [
        "trips_per_day_28d"
    ]


FLOAT32_COLUMNS = _float32_columns()


def _station_day(task: tuple) -> pd.DataFrame:
    """Worker: features for one station, kept only for origins inside the requested day."""
    history, horizons, first_origin, last_origin = task
    features = build_features(history, horizons)
    keep = features["origin_hour"].between(first_origin, last_origin)
    return features.loc[keep].drop(columns=TARGET_COLUMNS)


def read_panel(path: str, first_hour: pd.Timestamp, last_hour: pd.Timestamp) -> pd.DataFrame:
    """Panel rows from first_hour to last_hour inclusive."""
    months = month_range(f"{first_hour:%Y-%m}", f"{last_hour:%Y-%m}")
    panel = pd.read_parquet(path, columns=["station_key", "hour", "pickups", "dropoffs"],
                            filters=[("month", "in", months)])
    return panel[panel["hour"].between(first_hour, last_hour)]


class FeatureStore:
    """Compute and cache forecast features for replayed origin hours."""

    def __init__(self, panel_path: str, stations_path: str, horizons: list[int],
                 workers: int = 4, cached_days: int = 8,
                 allowed: tuple[pd.Timestamp, pd.Timestamp] | None = None):
        self.panel_path = panel_path
        self.horizons = horizons
        self.stations = pd.read_parquet(stations_path, columns=STATION_COLUMNS)
        self.cached_days = cached_days
        self._cache: OrderedDict[pd.Timestamp, pd.DataFrame] = OrderedDict()
        self._lock = threading.Lock()
        self._computing: dict[pd.Timestamp, Future] = {}
        self._pool = ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))
        hours = pd.read_parquet(panel_path, columns=["hour"])["hour"]
        first_hour, last_hour = hours.min(), hours.max()
        # Earliest origin with full history behind it, and latest whose forecast hours
        # are all inside the panel. allowed narrows this, for example to the replay months.
        self.first_origin = (first_hour + pd.Timedelta(hours=HISTORY_HOURS)).ceil("D")
        self.last_origin = last_hour - pd.Timedelta(hours=max(horizons))
        if allowed:
            self.first_origin = max(self.first_origin, allowed[0])
            self.last_origin = min(self.last_origin, allowed[1])

    def check_origin(self, origin: pd.Timestamp) -> None:
        if origin != origin.floor("h"):
            raise ValueError("origin must be on the hour")
        if not self.first_origin <= origin <= self.last_origin:
            raise ValueError(f"origin must be between {self.first_origin} and {self.last_origin}")

    def features_at(self, origin: pd.Timestamp) -> pd.DataFrame:
        """One row per active station and horizon for this origin, with station details."""
        self.check_origin(origin)
        day = self.day_features(origin.floor("D"))
        return day[day["origin_hour"] == origin].reset_index(drop=True)

    def day_features(self, day: pd.Timestamp) -> pd.DataFrame:
        """Cached day features. The lock guards only the cache: a day being computed (15 to
        30 s) does not block requests for other days, and callers wanting the same day share
        one computation."""
        with self._lock:
            if day in self._cache:
                self._cache.move_to_end(day)
                return self._cache[day]
            computing = self._computing.get(day)
            owner = computing is None
            if owner:
                computing = self._computing[day] = Future()
        if not owner:
            return computing.result()
        try:
            features = self._compute_day(day)
        except BaseException as error:
            with self._lock:
                del self._computing[day]
            computing.set_exception(error)
            raise
        with self._lock:
            self._cache[day] = features
            if len(self._cache) > self.cached_days:
                self._cache.popitem(last=False)
            del self._computing[day]
        computing.set_result(features)
        return features

    def _compute_day(self, day: pd.Timestamp) -> pd.DataFrame:
        first_origin = day
        last_origin = day + pd.Timedelta(hours=23)
        history = read_panel(self.panel_path,
                             first_origin - pd.Timedelta(hours=HISTORY_HOURS), last_origin)
        # A station needs a row at an origin to be forecast, as offline.
        active = history.loc[history["hour"] >= first_origin, "station_key"].unique()
        history = history[history["station_key"].isin(active)]
        tasks = [(group, self.horizons, first_origin, last_origin)
                 for _, group in history.groupby("station_key", sort=False)]
        parts = list(self._pool.map(_station_day, tasks, chunksize=16))
        features = pd.concat(parts, ignore_index=True)
        features[FLOAT32_COLUMNS] = features[FLOAT32_COLUMNS].astype("float32")
        log.info("Computed features for %s: %d stations, %d rows",
                 day.date(), len(tasks), len(features))
        return features.merge(self.stations, on="station_key", how="left")

    def actuals(self, station_key: str, first_hour: pd.Timestamp,
                last_hour: pd.Timestamp) -> pd.DataFrame:
        """Observed hourly pickups and drop-offs for one station."""
        panel = read_panel(self.panel_path, first_hour, last_hour)
        rows = panel[panel["station_key"] == station_key].set_index("hour")
        full = pd.date_range(first_hour, last_hour, freq="h", name="hour")
        return rows[list(DEMAND_COLUMNS)].reindex(full, fill_value=0).reset_index()

    def close(self) -> None:
        self._pool.shutdown(cancel_futures=True)
