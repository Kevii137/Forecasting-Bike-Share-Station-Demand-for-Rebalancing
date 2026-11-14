"""Daily monitoring records and rolling-window metrics.

The live replay and the offline calibration both reduce each day to a DayRecord, so
thresholds calibrated offline mean the same thing in the running service.

- Errors are attributed to the day of the target hour, and only scored once observed.
- Drift counts are attributed to the day of the origin, when the inputs were seen.
- A window sums the records of its days before dividing, so busy days weigh more, as
  in WAPE itself.

Only model-routed stations are monitored. The baseline is the historical average the
service would have served instead (the fallback forecast), scored on the same rows.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from bikeshare.features import DEMAND_COLUMNS
from bikeshare.modeling.metrics import volume_tier
from bikeshare.monitoring.drift import histogram, psi

ERROR_COLUMNS = ["model_abs_error", "baseline_abs_error", "actual", "rows"]


@dataclass
class DayRecord:
    day: pd.Timestamp
    model_version: str
    # Index (target, group, label), columns ERROR_COLUMNS. group is all, horizon or tier.
    errors: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Monitored column -> counts per profile bin, over origins on this day.
    counts: dict[str, np.ndarray] = field(default_factory=dict)


def error_sums(scored: pd.DataFrame, tier_bounds: list[float]) -> pd.DataFrame:
    """Absolute-error and demand sums by target, overall, by horizon and by volume tier.

    scored needs, for each demand, {demand}_target, {demand}_forecast and {demand}_baseline,
    plus horizon and trips_per_day_28d.
    """
    labels = {
        "all": pd.Series("all", index=scored.index),
        "horizon": scored["horizon"].astype(int).astype(str),
        "tier": volume_tier(scored["trips_per_day_28d"], tier_bounds),
    }
    parts = []
    for demand in DEMAND_COLUMNS:
        actual = scored[f"{demand}_target"].astype("float64")
        frame = pd.DataFrame({
            "model_abs_error": (actual - scored[f"{demand}_forecast"]).abs(),
            "baseline_abs_error": (actual - scored[f"{demand}_baseline"]).abs(),
            "actual": actual,
            "rows": 1,
        })
        for group, label in labels.items():
            sums = frame.groupby(label.to_numpy()).sum()
            sums.index = pd.MultiIndex.from_product([[demand], [group], sums.index],
                                                    names=["target", "group", "label"])
            parts.append(sums)
    return pd.concat(parts)[ERROR_COLUMNS]


def drift_counts(rows: pd.DataFrame, profile: dict) -> dict[str, np.ndarray]:
    return {column: histogram(rows[column].to_numpy(), spec["edges"])
            for column, spec in profile["columns"].items()}


def combine(records: list[DayRecord]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Sum the error tables and drift counts of several days."""
    tables = [record.errors for record in records if not record.errors.empty]
    errors = (pd.concat(tables).groupby(level=["target", "group", "label"]).sum()
              if tables else pd.DataFrame(columns=ERROR_COLUMNS))
    counts: dict[str, np.ndarray] = {}
    for record in records:
        for column, values in record.counts.items():
            counts[column] = counts.get(column, 0) + values
    return errors, counts


def window_metrics(records: list[DayRecord], profile: dict | None) -> dict:
    """WAPE of model and baseline, their ratio, and PSI per monitored column.

    relative_wape below 1 means the model beats the historical average on these days.
    """
    errors, counts = combine(records)
    table = errors.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        table["model_wape"] = table["model_abs_error"] / table["actual"]
        table["baseline_wape"] = table["baseline_abs_error"] / table["actual"]
        table["relative_wape"] = table["model_abs_error"] / table["baseline_abs_error"]
    drift = {}
    if profile:
        drift = {column: psi(profile["columns"][column]["shares"], values)
                 for column, values in counts.items() if column in profile["columns"]}
    return {
        "days": len(records),
        "first_day": records[0].day if records else None,
        "last_day": records[-1].day if records else None,
        "errors": table,
        "psi": drift,
        "max_psi": max((value for value in drift.values() if np.isfinite(value)), default=np.nan),
    }
