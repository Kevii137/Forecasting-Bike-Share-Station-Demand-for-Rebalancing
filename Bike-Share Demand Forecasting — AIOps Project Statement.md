# Bike-Share Demand Forecasting — AIOps Project Statement

## Project title

**Forecasting Bike-Share Station Demand for Proactive Rebalancing: An End-to-End AIOps Pipeline**

## Problem and application domain

Public bike-share systems face a recurring operational problem: during peak hours some stations run out of bikes while others fill up completely, leaving riders unable to pick up or return a bike. Operators fix this by "rebalancing", moving bikes between stations by truck, but this works only if they know in advance where shortages will occur.

Demand changes with time of day, season, weather and events, so fixed schedules and manual judgement quickly become unreliable. My project addresses this problem in the domain of urban mobility and transport operations.

I will use the publicly available [Divvy (Chicago) bike-share trip data](https://divvybikes.com/system-data), which provides millions of anonymized trips released month by month.

## Objective of the AI system

The system forecasts hourly pickups and drop-offs for every station over the next few hours, and uses these forecasts to flag stations likely to become empty or full.

I plan to use a gradient boosting model trained on lagged demand, calendar and station features. I will compare it against seasonal-naive and historical-average baselines.

The forecasts will be served through a web application that lets a non-technical operations dispatcher view predicted demand and a ranked list of stations needing attention.

## Expected outcome

The expected outcome is a deployed forecasting service that clearly outperforms the baselines on time-based held-out data and produces actionable rebalancing alerts. The service should keep its accuracy over time through monitoring and retraining.

## Building and managing the system with AIOps

The project will be built and managed following AIOps practices throughout the lifecycle:

| Lifecycle stage | Tools | Role in the project |
| --- | --- | --- |
| Data engineering | Apache Spark | Distributed data ingestion, validation and feature engineering |
| Versioning and CI | Git, DVC | Versioning code and data, with a DVC pipeline as the CI DAG |
| Model tuning | Ray | Parallel hyperparameter tuning |
| Experiment tracking | MLflow | Experiment tracking and the model registry |
| Serving and packaging | MLflow, FastAPI, Docker | Model serving, with the frontend and backend packaged as separate Docker containers |
| Monitoring | Prometheus, Grafana | Monitoring latency, data drift and live prediction error |

Because bike demand shifts strongly across seasons, I will replay later months of data as simulated production traffic. This will show data drift and live forecast error being monitored, with sustained forecast degradation triggering automated retraining and model promotion.
