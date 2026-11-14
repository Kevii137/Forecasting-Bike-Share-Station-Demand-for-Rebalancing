# Forecasting Bike-Share Station Demand for Proactive Rebalancing

An end-to-end AIOps pipeline that forecasts hourly pickups and drop-offs for Divvy (Chicago) stations and shows an operations dispatcher which stations need attention. DA3408 AI Operations project, IIT Madras.

**Status:** data pipeline, models, serving API, frontend and monitoring (Prometheus and Grafana) complete. Automated retraining and promotion are not built yet.

## Quick start

Requires Docker with Compose, about 4 GB of free memory, and the data and MLflow store from the earlier stages (`dvc pull`, then `dvc repro train` and `python -m bikeshare.modeling.register` if `mlflow.db` is empty).

```bash
cp .env.example .env              # then set GRAFANA_ADMIN_PASSWORD
docker compose up -d --build
python3 scripts/smoke_test.py     # end-to-end checks of the running stack
```

| Service | URL | What it is |
|---|---|---|
| Frontend | http://localhost:8501 | Rebalancing alerts, station forecasts, model monitoring, pipeline console |
| Backend API | http://localhost:8000/docs | FastAPI, with interactive OpenAPI docs |
| MLflow | http://localhost:5000 | Experiment tracking and model registry |
| Grafana | http://localhost:3000 | Monitoring dashboard, read-only without login |
| Prometheus | http://localhost:9090 | Metrics and alert rules |

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

| `GET /monitoring` | Replay status, the latest 7-day window and one row per simulated day |
| `POST /monitoring/replay/start?first=...&last=...&delay_seconds=0`, `POST /monitoring/replay/stop` | Control the simulated clock |

Times are Chicago wall-clock hours, as in the Divvy data.

## Monitoring

The replay monitor treats March to September 2026 as simulated production. Start it from the frontend's **Model monitoring** page, or with `curl -X POST localhost:8000/monitoring/replay/start`. A simulated clock advances one hour at a time. At each hour the backend forecasts every station. At the end of each simulated day it scores the forecasts against the trips that then happened, and compares the week's inputs with the training data. In Docker each simulated day takes about 45 seconds, mostly computing that day's features, so all of March to September takes about 2.5 hours. A replay from 1 March reaches its first alerts in about 8 minutes.

| Signal | Measure, over the last 7 simulated days | Alert after 5 consecutive days above |
|---|---|---|
| Forecast error | Model WAPE divided by the WAPE of the historical-average fallback | 1.0 |
| Drift | Largest population stability index (PSI) of 7 inputs and forecasts against the training window | 0.5 |

Thresholds are in `params.yaml` (`monitoring.thresholds`) and were calibrated by replaying every day offline (`dvc repro calibrate`, results in `reports/monitoring_calibration.json` and `.png`). In a replay started on 1 March, both alerts fire on simulated 11 March: 7 days fill the window, then 5 days breach. Only the forecast-error alert will trigger retraining; the drift alert is advisory, because calibration found high drift while the model still beat its fallback. The reasoning is in the decision log (Milestone 5).

Grafana shows the same signals with request rate, latency, errors and firing alerts. Prometheus alert rules are in `monitoring/prometheus/rules.yml`; the forecast-error alert carries `action=retrain` for Milestone 6, and the drift alert `action=investigate`.

## Development

```bash
# Backend tests, including online/offline parity on real data (about 3 min)
PYTHONPATH=src .venv/bin/python -m pytest tests

# Frontend: its own environment, matching its container
uv venv .venv-frontend --python 3.12
uv pip install --python .venv-frontend/bin/python -r frontend/requirements.lock pytest
cd frontend && ../.venv-frontend/bin/python -m pytest tests

# Prometheus alert-rule unit tests
docker run --rm -v "$PWD/monitoring/prometheus:/cfg:ro" --entrypoint promtool \
  prom/prometheus:v3.5.0 test rules /cfg/rules_test.yml

# Run the backend outside Docker, against an MLflow server on port 5000
PYTHONPATH=src .venv/bin/python -m bikeshare.serving.app
```

## Documentation

- [Project statement](Bike-Share%20Demand%20Forecasting%20—%20AIOps%20Project%20Statement.md)
- [Decision log](docs/decision-log.md)
- [AI disclosure](AI_DISCLOSURE.md)
