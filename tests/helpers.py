"""Shared test doubles and constants for the serving and monitoring tests."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from bikeshare.modeling.dataset import FEATURE_COLUMNS
from bikeshare.serving.forecast import Rules
from bikeshare.serving.online import HISTORY_HOURS

RULES = Rules(min_trips_per_day=5, allowed_types=("classic",), alert_limit=10, min_pressure=0.1)
START = pd.Timestamp("2026-01-01")
HOURS = HISTORY_HOURS + 24 * 5
ORIGIN = START + pd.Timedelta(hours=HISTORY_HOURS + 24 * 2 + 8)


class ConstantModel:
    """Stands in for the pyfunc: a busy pickup station that is draining."""

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        assert list(frame.columns) == FEATURE_COLUMNS
        assert (frame.dtypes == "float64").all()
        return pd.DataFrame({"pickups_forecast": np.full(len(frame), 4.0),
                             "dropoffs_forecast": np.full(len(frame), 1.0)})


class FakeClient:
    def __init__(self):
        self.version = "1"
        self.fail = False

    def get_model_version_by_alias(self, name, alias):
        if self.fail:
            raise ConnectionError("registry down")
        return SimpleNamespace(version=self.version, run_id=f"run-{self.version}", tags={})
