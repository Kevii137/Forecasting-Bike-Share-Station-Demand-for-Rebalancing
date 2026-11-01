"""Access to the central project configuration in params.yaml."""
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PARAMS_PATH = Path("params.yaml")


def load_params(path: Path = PARAMS_PATH) -> dict[str, Any]:
    """Read params.yaml, the single source of pipeline configuration."""
    with path.open() as handle:
        return yaml.safe_load(handle)


def month_range(first: str, last: str) -> list[str]:
    """Return every month from first to last inclusive, formatted as YYYY-MM."""
    return [str(period) for period in pd.period_range(first, last, freq="M")]
