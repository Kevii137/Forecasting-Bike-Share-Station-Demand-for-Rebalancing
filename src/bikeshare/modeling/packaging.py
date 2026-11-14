"""Package trained boosters as one versioned MLflow pyfunc model."""
import json
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow.models import infer_signature

from bikeshare.modeling.dataset import FEATURE_COLUMNS
from bikeshare.modeling.pyfunc import DemandModel


def log_model(boosters: dict[str, lgb.Booster], sample: pd.DataFrame, model_dir: Path) -> None:
    """Log both boosters as one pyfunc named "model" in the active MLflow run."""
    model_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for demand, booster in boosters.items():
        path = model_dir / f"{demand}.txt"
        booster.save_model(path, num_iteration=booster.best_iteration)
        artifacts[demand] = str(path)
    columns_path = model_dir / "feature_columns.json"
    columns_path.write_text(json.dumps(FEATURE_COLUMNS))
    artifacts["feature_columns"] = str(columns_path)

    inputs = sample[FEATURE_COLUMNS].astype("float64")
    outputs = pd.DataFrame({f"{demand}_forecast": [0.0] * len(inputs) for demand in boosters})
    mlflow.pyfunc.log_model(
        name="model", python_model=DemandModel(), artifacts=artifacts,
        code_paths=["src/bikeshare"], signature=infer_signature(inputs, outputs),
        input_example=inputs.head(3),
        pip_requirements=[f"lightgbm=={lgb.__version__}", f"pandas=={pd.__version__}",
                          f"numpy=={np.__version__}", f"mlflow=={mlflow.__version__}",
                          f"pyyaml=={yaml.__version__}"],
    )
