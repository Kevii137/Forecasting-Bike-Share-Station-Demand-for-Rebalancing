"""MLflow pyfunc wrapper that serves both demand models as one versioned model."""
import json
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from bikeshare.modeling.dataset import TARGETS


class DemandModel(mlflow.pyfunc.PythonModel):
    """Predict pickups and drop-offs from feature rows."""

    def load_context(self, context) -> None:
        self.boosters = {
            demand: lgb.Booster(model_file=context.artifacts[demand]) for demand in TARGETS
        }
        self.feature_columns = json.loads(Path(context.artifacts["feature_columns"]).read_text())

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        missing = set(self.feature_columns) - set(model_input.columns)
        if missing:
            raise ValueError(f"missing feature columns: {sorted(missing)}")
        inputs = model_input[self.feature_columns].astype("float64")
        return pd.DataFrame({
            f"{demand}_forecast": np.clip(booster.predict(inputs), 0, None)
            for demand, booster in self.boosters.items()
        })
