# Forecasting Bike-Share Station Demand for Proactive Rebalancing

An end-to-end AIOps pipeline that forecasts hourly pickups and drop-offs for Divvy (Chicago) stations and shows an operations dispatcher which stations need attention. DA3408 AI Operations project, IIT Madras.

**Status:** data pipeline, models, serving API and frontend complete. Monitoring (Prometheus and Grafana) and retraining are not built yet.

## Quick start

Requires Docker with Compose, about 4 GB of free memory, and the data and MLflow store from the earlier stages (`dvc pull`, then `dvc repro train` and `python -m bikeshare.modeling.register` if `mlflow.db` is empty).

```bash
docker compose up -d --build
python3 scripts/smoke_test.py     # end-to-end checks of the running stack
```

| Service | URL | What it is |
|---|---|---|
| Frontend | http://localhost:8501 | Rebalancing alerts, station forecasts, pipeline console |
| Backend API | http://localhost:8000/docs | FastAPI, with interactive OpenAPI docs |
| MLflow | http://localhost:5000 | Experiment tracking and model registry |

All ports bind to localhost only. Stop any other MLflow server on port 5000 first, since both would use `mlflow.db`.

The app replays history: pick a date and hour between March and September 2026 in the sidebar, and it forecasts the next 3 hours from data up to that hour. The first hour of a new day takes 15 to 30 seconds while the backend computes features; other hours that day take under a second.

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

## Serving

The backend loads the model holding the `champion` alias in the MLflow registry, and checks every 30 seconds whether the alias has moved (`POST /model/reload` checks immediately). It builds features with the same function the Spark pipeline uses, on replayed panel history. A test checks that every online feature equals the Spark value exactly.

| Endpoint | Returns |
|---|---|
| `GET /alerts?origin=2026-07-15T17:00` | Stations likely to drain or fill over the next 3 hours, ranked by net change as a share of docks |
| `GET /forecast?origin=...` | 3-hour totals for every active station, with forecast source and confidence |
| `GET /station?key=...&origin=...` | One station: last 24 hours observed, the forecast, and what happened next in the replay |
| `GET /model`, `POST /model/reload` | Serving model version, lineage tags, and a registry check |
| `GET /replay` | The forecast-time window the service accepts |
| `GET /pipeline` | Data-quality and evaluation reports from the DVC pipeline |
| `GET /health`, `GET /ready`, `GET /metrics` | Liveness, readiness (model loaded), Prometheus metrics |

Times are Chicago wall-clock hours, as in the Divvy data.

## Development

```bash
# Backend tests, including online/offline feature parity on real data (about 1 min)
PYTHONPATH=src .venv/bin/python -m pytest tests

# Frontend: its own environment, matching its container
uv venv .venv-frontend --python 3.12
uv pip install --python .venv-frontend/bin/python -r frontend/requirements.lock pytest
cd frontend && ../.venv-frontend/bin/python -m pytest tests

# Run the backend outside Docker, against an MLflow server on port 5000
PYTHONPATH=src .venv/bin/python -m bikeshare.serving.app
```

## Documentation

- [Project statement](Bike-Share%20Demand%20Forecasting%20—%20AIOps%20Project%20Statement.md)
- [Decision log](docs/decision-log.md)
- [AI disclosure](AI_DISCLOSURE.md)
