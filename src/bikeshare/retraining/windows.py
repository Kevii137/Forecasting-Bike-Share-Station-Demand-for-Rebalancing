"""Split the data observed up to a simulated time into fitting, early-stopping and gate windows.

Rows are assigned by target hour, the hour whose demand is being forecast:

    train_start ... stop_start ... gate_start ... as_of
    |-- fit the model --|-- early stop --|----- gate ------|

Nothing after as_of is read, so a retrained model never sees demand that the replay has
not yet reached. The gate window is unseen by the challenger and, because the champion
was trained before the previous gate window, by the champion too.
"""
from dataclasses import dataclass

import pandas as pd

from bikeshare.config import month_range
from bikeshare.modeling.dataset import (
    FEATURE_COLUMNS,
    FEATURES_PATH,
    KEY_COLUMNS,
    TARGETS,
)

HOUR = pd.Timedelta(hours=1)
# Feature partitions are keyed by origin month; a target can be up to 3 hours later.
MAX_HORIZON = pd.Timedelta(hours=3)


@dataclass(frozen=True)
class Windows:
    train_start: pd.Timestamp  # first target hour used for fitting
    stop_start: pd.Timestamp   # first target hour of the early-stopping window
    gate_start: pd.Timestamp   # first target hour of the gate window
    as_of: pd.Timestamp        # last observed hour, inclusive

    def describe(self) -> dict[str, str]:
        last = {"train": self.stop_start - HOUR, "early_stopping": self.gate_start - HOUR,
                "gate": self.as_of}
        first = {"train": self.train_start, "early_stopping": self.stop_start,
                 "gate": self.gate_start}
        return {name: f"{first[name]:%Y-%m-%d %H:%M} to {last[name]:%Y-%m-%d %H:%M}"
                for name in first}


def plan(as_of: pd.Timestamp, first_month: str, gate_days: int,
         early_stopping_days: int) -> Windows:
    as_of = pd.Timestamp(as_of).floor("h")
    gate_start = as_of + HOUR - pd.Timedelta(days=gate_days)
    stop_start = gate_start - pd.Timedelta(days=early_stopping_days)
    train_start = pd.Timestamp(first_month)
    if stop_start - train_start < pd.Timedelta(days=28):
        raise ValueError(f"as_of {as_of} leaves less than 28 days to train on")
    return Windows(train_start, stop_start, gate_start, as_of)


def load_rows(start: pd.Timestamp, end: pd.Timestamp, keys: bool = False,
              path: str = FEATURES_PATH) -> pd.DataFrame:
    """Feature rows whose target hour is in [start, end] and has been observed."""
    months = month_range(f"{start - MAX_HORIZON:%Y-%m}", f"{end:%Y-%m}")
    columns = (KEY_COLUMNS if keys else ["target_hour"]) + FEATURE_COLUMNS + list(TARGETS.values())
    frame = pd.read_parquet(path, columns=columns, filters=[("month", "in", months)])
    frame = frame[(frame["target_hour"] >= start) & (frame["target_hour"] <= end)]
    return frame.dropna(subset=list(TARGETS.values())).reset_index(drop=True)
