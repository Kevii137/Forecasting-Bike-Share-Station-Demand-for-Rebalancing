"""Score a model's forecasts and flatten the scores into MLflow metric names."""
import pandas as pd

from bikeshare.modeling.dataset import TARGETS
from bikeshare.modeling.metrics import breakdown, net_flow_error


def metric_key(*parts: str) -> str:
    """MLflow metric names cannot contain '+', used in tier labels such as 20+."""
    return "_".join(parts).replace("+", "plus")


def flatten(prefix: str, scores: dict) -> dict[str, float]:
    flat = {}
    for metric, value in scores["overall"].items():
        flat[metric_key(prefix, metric)] = value
    for group in ("horizon", "tier", "period"):
        for label, values in scores[group].items():
            flat[metric_key(prefix, group, label, "wape")] = values["wape"]
    return flat


def evaluate(frame: pd.DataFrame, forecasts: dict[str, str], evaluation: dict) -> dict:
    """Score one model's forecasts. forecasts maps each demand to its forecast column."""
    result = {
        demand: breakdown(frame, TARGETS[demand], column,
                          evaluation["tier_bounds"], evaluation["peak_hours"])
        for demand, column in forecasts.items()
    }
    result["net_flow"] = net_flow_error(frame, forecasts["pickups"], forecasts["dropoffs"])
    return result
