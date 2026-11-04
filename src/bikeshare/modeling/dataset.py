"""Load feature rows for a chronological split and define the model's inputs."""
import pandas as pd

from bikeshare.config import month_range
from bikeshare.features import DEMAND_COLUMNS

FEATURES_PATH = "data/processed/features"
TARGETS = {column: f"{column}_target" for column in DEMAND_COLUMNS}

# Station ID is deliberately not an input: with a short training window it cannot
# generalise to stations that open later. Location, capacity and each station's own
# recent demand carry that information instead.
FEATURE_COLUMNS = (
    ["horizon"]
    + [
        f"{column}_{suffix}"
        for column in DEMAND_COLUMNS
        for suffix in (
            "lag_0", "lag_1", "lag_2", "mean_3h", "mean_24h", "mean_168h",
            "same_hour_1d", "same_hour_1w", "same_hour_4w_mean", "same_hour_8w_mean",
        )
    ]
    + [
        "trips_per_day_28d", "target_hour_of_day", "target_day_of_week",
        "target_is_weekend", "target_is_holiday", "capacity", "lat", "lng",
    ]
)
KEY_COLUMNS = ["station_key", "origin_hour", "target_hour"]


def load_split(months: list[str], path: str = FEATURES_PATH) -> pd.DataFrame:
    """Read the feature rows whose origin hour falls in the given first and last month.

    Rows whose target has not been observed are dropped, since they cannot be scored.
    """
    selected = month_range(months[0], months[1])
    columns = KEY_COLUMNS + FEATURE_COLUMNS + list(TARGETS.values())
    frame = pd.read_parquet(path, columns=columns, filters=[("month", "in", selected)])
    if frame.empty:
        raise ValueError(f"no feature rows for months {months}")
    return frame.dropna(subset=list(TARGETS.values())).reset_index(drop=True)
