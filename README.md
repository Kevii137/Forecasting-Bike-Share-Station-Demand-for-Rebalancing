# Forecasting Bike-Share Station Demand for Proactive Rebalancing

An end-to-end AIOps pipeline that forecasts hourly pickups and drop-offs for Divvy (Chicago) stations and shows an operations dispatcher which stations need attention. DA3408 AI Operations project, IIT Madras.

**Status:** data pipeline and models complete. Serving, frontend and monitoring are not built yet.

## Setup

Requires Python 3.12 and Java 17.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.lock
PYTHONPATH=src .venv/bin/python -m pytest tests
```

## Data pipeline

Requires about 7 GB of free memory. Spark is capped in `src/bikeshare/spark.py`.

```bash
source .venv/bin/activate
dvc pull     # restore data from the DVC remote, or
dvc repro    # rebuild from source, rerunning only changed stages
dvc dag      # show the pipeline
dvc metrics show
```

| Stage | Output | Time |
|---|---|---|
| `ingest` | Monthly trips and one station snapshot | Depends on network |
| `clean` | 8.87M validated trips | 50 s |
| `stations` | 2,037 stations, with capacity for 99.1% of trips | Under 1 min |
| `panel` | 18.4M station-hours | 37 s |
| `features` | 14.1M rows for eligible stations | 62 s |

## Models

Requires the MLflow tracking server, which stores runs in `mlflow.db` and artifacts in `mlartifacts/`, both ignored by Git:

```bash
mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts
```

`dvc repro train` runs `tune` (Ray Tune random search, 12 trials per target, under 45 min) and then `train` (final fit, test scoring and baselines, about 20 min). Run it with about 5 GB of free memory. Each MLflow run is tagged with the Git commit and the DVC hash of the feature data.

February 2026 test set, 543,618 station-hour forecasts. WAPE is total absolute error divided by total trips, so lower is better.

| Model | Pickups WAPE | Drop-offs WAPE | Peak pickups WAPE | 3-hour net flow MAE |
|---|---|---|---|---|
| LightGBM (Poisson) | **0.817** | **0.819** | **0.632** | **1.22** |
| Historical average, 8-week same hour | 0.842 | 0.826 | 0.673 | 1.26 |
| Seasonal naive, same hour last week | 0.991 | 0.963 | 0.796 | 1.55 |

Full breakdowns by horizon, station tier and peak period are in `reports/metrics.json`.

## Documentation

- [Project statement](Bike-Share%20Demand%20Forecasting%20—%20AIOps%20Project%20Statement.md)
- [Decision log](docs/decision-log.md)
- [AI disclosure](AI_DISCLOSURE.md)
