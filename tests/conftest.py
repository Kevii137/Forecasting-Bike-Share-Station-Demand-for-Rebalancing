"""Fixtures shared by the serving and monitoring tests: a small synthetic panel."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from helpers import HOURS, START

from bikeshare.serving.online import FeatureStore


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> Path:
    """Three stations: busy classic, quiet classic, busy lightweight."""
    root = tmp_path_factory.mktemp("serving")
    hours = pd.date_range(START, periods=HOURS, freq="h")
    rng = np.random.default_rng(0)
    rates = {"busy": 1.0, "quiet": 0.05, "light": 1.0}
    panel = pd.concat([
        pd.DataFrame({"station_key": key, "hour": hours,
                      "pickups": rng.poisson(rate, HOURS), "dropoffs": rng.poisson(rate, HOURS)})
        for key, rate in rates.items()
    ])
    panel["month"] = panel["hour"].dt.strftime("%Y-%m")
    panel.to_parquet(root / "panel", partition_cols=["month"])
    pd.DataFrame({
        "station_key": ["busy", "quiet", "light"], "lat": [41.9, 41.8, 41.7],
        "lng": [-87.6, -87.7, -87.8], "capacity": [15.0, 11.0, np.nan],
        "station_type": ["classic", "classic", "lightweight"],
    }).to_parquet(root / "stations.parquet")
    params = yaml.safe_load(Path("params.yaml").read_text())
    params["split"]["replay"] = [f"{START:%Y-%m}", f"{hours[-1]:%Y-%m}"]
    params["serving"]["feature_workers"] = 1
    (root / "params.yaml").write_text(yaml.safe_dump(params))
    return root


@pytest.fixture(scope="module")
def store(data_dir) -> FeatureStore:
    store = FeatureStore(str(data_dir / "panel"), str(data_dir / "stations.parquet"),
                         [1, 2, 3], workers=1)
    yield store
    store.close()
