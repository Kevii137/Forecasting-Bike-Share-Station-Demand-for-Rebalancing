"""Population stability index (PSI) of model inputs and forecasts against a reference.

The reference profile stores, for each monitored column, bin edges taken from quantiles
of the training window and the share of reference rows in each bin. Monitoring then
counts recent rows into the same bins and compares the two distributions:

    PSI = sum over bins of (recent share - reference share) * ln(recent share / reference share)

A common rule of thumb reads PSI below 0.1 as stable and above 0.25 as a major shift.
This project calibrates its own threshold instead (Decision D9).
"""
import numpy as np
import pandas as pd

PROFILE_VERSION = 1
# Empty bins would make the logarithm infinite, so shares are floored at this value.
MIN_SHARE = 1e-4


def bin_edges(values: np.ndarray, bins: int) -> list[float]:
    """Interior edges at the reference quantiles. Repeated quantiles collapse into one edge.

    Demand counts are mostly zero, so several quantiles are often 0. Collapsing them leaves
    a bin that holds exactly the zeros, because a value equal to an edge falls in the bin
    below it (see histogram).
    """
    values = np.asarray(values, dtype="float64")
    values = values[~np.isnan(values)]
    quantiles = np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1])
    return np.unique(quantiles).tolist()


def histogram(values: np.ndarray, edges: list[float]) -> np.ndarray:
    """Count values into the bins (-inf, e0], (e0, e1], ..., (e_last, inf). NaN is skipped."""
    values = np.asarray(values, dtype="float64")
    values = values[~np.isnan(values)]
    index = np.searchsorted(np.asarray(edges, dtype="float64"), values, side="left")
    return np.bincount(index, minlength=len(edges) + 1)


def psi(reference_shares: list[float], counts: np.ndarray) -> float:
    """PSI of the counted rows against the reference shares. NaN when nothing was counted."""
    counts = np.asarray(counts, dtype="float64")
    if counts.sum() == 0:
        return float("nan")
    expected = np.clip(np.asarray(reference_shares, dtype="float64"), MIN_SHARE, None)
    actual = np.clip(counts / counts.sum(), MIN_SHARE, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def build_profile(frame: pd.DataFrame, columns: list[str], bins: int, **details) -> dict:
    """Reference bins and shares for each column. details are stored alongside, for lineage."""
    profile = {"version": PROFILE_VERSION, "bins": bins, "rows": len(frame), **details,
               "columns": {}}
    for column in columns:
        edges = bin_edges(frame[column].to_numpy(), bins)
        counts = histogram(frame[column].to_numpy(), edges)
        profile["columns"][column] = {
            "edges": edges,
            "shares": (counts / counts.sum()).tolist(),
        }
    return profile
