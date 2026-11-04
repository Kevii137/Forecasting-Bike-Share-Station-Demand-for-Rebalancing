"""The two baselines the model must beat, read from features already built.

Both use only data before the origin, so they face the same information as the model.
"""
import pandas as pd

BASELINES = {
    "seasonal_naive": "same_hour_1w",          # same hour one week earlier
    "historical_average": "same_hour_8w_mean",  # same weekday and hour, previous 8 weeks
}


def baseline_forecast(frame: pd.DataFrame, baseline: str, demand: str) -> pd.Series:
    return frame[f"{demand}_{BASELINES[baseline]}"]
