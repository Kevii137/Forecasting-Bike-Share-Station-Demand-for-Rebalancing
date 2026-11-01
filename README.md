# Forecasting Bike-Share Station Demand for Proactive Rebalancing

An end-to-end AIOps pipeline that forecasts hourly pickups and drop-offs for Divvy (Chicago) stations and shows an operations dispatcher which stations need attention. DA3408 AI Operations project, IIT Madras.

**Status:** data pipeline complete. No model or application exists yet.

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

## Documentation

- [Project statement](Bike-Share%20Demand%20Forecasting%20—%20AIOps%20Project%20Statement.md)
- [Decision log](docs/decision-log.md)
- [AI disclosure](AI_DISCLOSURE.md)
