# Instructions for the TA

**Project:** Forecasting Bike-Share Station Demand for Proactive Rebalancing (DA3408 AI Operations)
**Student:** K Kevin Sona, DA24B007

This page maps each rubric item to the file or screen that shows it. Screenshots of the running system are in [`evidence/`](evidence/), numbered in rubric order and captured on 2026-11-14. Images whose file names end in `tests`, `history`, `status`, `services` or `error-handling` show real command output, rendered as an image.

## Running it

You need Docker with Compose and about 9 GB of free memory. The data and the MLflow store are not in Git (see [Source control](#source-control-and-continuous-integration-2) below), so a fresh clone needs the DVC remote first:

```bash
dvc pull                      # data from the DVC remote
cp .env.example .env          # set a Grafana admin password
docker compose up -d --build  # seven services
python3 scripts/smoke_test.py # ends with "All checks passed"
```

The MLflow store (`mlflow.db`, `mlartifacts/`) is local and not in Git. On a fresh machine, run `dvc repro train` and then `python -m bikeshare.modeling.register` before starting Compose, so the backend has a champion model to load.

| Open | Address |
|---|---|
| App (frontend) | http://localhost:8501 |
| Backend API docs | http://localhost:8000/docs |
| Trainer API docs | http://localhost:8001/docs |
| MLflow | http://localhost:5000 |
| Prometheus / Alertmanager | http://localhost:9090 / http://localhost:9093 |
| Grafana (read-only, no login) | http://localhost:3000/d/bikeshare |

Full setup is in the [README](README.md). The [user manual](docs/user-manual.md) walks through every screen.

## Mandatory documents

| Document | File |
|---|---|
| Architecture diagram with block explanations | [docs/architecture.md](docs/architecture.md) |
| High-level design, with design choices and rationale | [docs/hld.md](docs/hld.md) |
| Low-level design, with endpoint input/output | [docs/lld.md](docs/lld.md) |
| Test plan, test cases, test report, acceptance criteria | [docs/test-plan.md](docs/test-plan.md) |
| User manual for non-technical users | [docs/user-manual.md](docs/user-manual.md) |
| AI disclosure | [AI_DISCLOSURE.md](AI_DISCLOSURE.md) |
| Project statement | [Bike-Share Demand Forecasting — AIOps Project Statement.md](Bike-Share%20Demand%20Forecasting%20—%20AIOps%20Project%20Statement.md) |

## Demonstration

### Front-end UI/UX [3]

| Rubric question | Where to look | Screenshot |
|---|---|---|
| Intuitive for the problem | Rebalancing alerts page: two ranked lists ("send bikes", "free up docks") and a map | [01](evidence/01-frontend-rebalancing-alerts.png) |
| Usable by non-technical users | Plain labels, a "How to read these alerts" panel, station drill-down with forecast against what happened | [01](evidence/01-frontend-rebalancing-alerts.png), [02](evidence/02-frontend-station-forecast.png) |
| Foolproof | The date picker allows only the replay months. The API rejects bad input with a clear message (422 / 404 / 409), and the UI shows that message instead of a made-up forecast. | [17](evidence/17-api-error-handling.png) |
| Free of UI errors | 6 frontend tests render every page against a fake backend | [23](evidence/23-unit-tests-frontend-and-lint.png) |
| Responsive | A cached hour returns in about 0.08 s. The first request for a new day takes 15–35 s (features are computed online) and shows a spinner. | [docs/hld.md](docs/hld.md), *Performance and capacity* |
| User manual | [docs/user-manual.md](docs/user-manual.md) | |

### ML pipeline visualization [2]

| Rubric question | Where to look | Screenshot |
|---|---|---|
| Separate UI screen for the pipeline, including data engineering | **Pipeline console** page: serving model and its lineage, retraining runs, February test results, and row counts per DVC stage | [04](evidence/04-frontend-pipeline-console.png) |
| Using MLOps tool UIs, orchestrated together | The pipeline console links to MLflow and Grafana; the monitoring page links to Grafana | [04](evidence/04-frontend-pipeline-console.png), [03](evidence/03-frontend-model-monitoring.png) |
| Pipeline management console | Pipeline console: **Retrain now**, **Roll back the champion**, **Check the registry now**. Model monitoring page: start/stop the replay. | [04](evidence/04-frontend-pipeline-console.png), [03](evidence/03-frontend-model-monitoring.png) |
| Errors, failures and successful runs | Retraining table (outcome: promoted, rejected, rolled back; failed jobs show a red box). Grafana *Retraining* row (promoted / rejected / failed counters) and *Service* row (errors per second). | [04](evidence/04-frontend-pipeline-console.png), [21](evidence/21-grafana-dashboard.png) |
| Speed and throughput | `clean` 50 s for 9.1M trips, `panel` 37 s, `features` 62 s for 14.1M rows; replay about 45 s per simulated day; retraining about 12.5 min | [docs/hld.md](docs/hld.md), *Performance and capacity* |

## Software engineering

### Design principle [2]

| Rubric question | Where to look |
|---|---|
| Design document | [docs/hld.md](docs/hld.md) and [docs/lld.md](docs/lld.md) |
| OO or functional | Functional core (features, metrics, PSI, gate) with small classes for state: `ModelStore`, `FeatureStore`, `ReplayMonitor`, `Trainer`. See [docs/hld.md](docs/hld.md), *Quality and failure behaviour*. |
| LLD endpoint definitions with I/O | [docs/lld.md](docs/lld.md): `GET /alerts` in full with an example JSON response, then every other endpoint in tables. Live schemas: [15](evidence/15-fastapi-backend-docs.png), [16](evidence/16-fastapi-trainer-docs.png). |
| Architecture diagram | [docs/architecture.md](docs/architecture.md) (Mermaid; it renders on GitHub) |
| HLD diagram | Retraining flowchart in [docs/hld.md](docs/hld.md) |
| Loose coupling, frontend and backend joined only by configurable REST | The frontend never imports backend code. It calls REST only, through [frontend/api.py](frontend/api.py), with URLs from `BACKEND_URL` and `TRAINER_URL` set in [compose.yaml](compose.yaml). They are separate images: [frontend/Dockerfile](frontend/Dockerfile) and [docker/backend.Dockerfile](docker/backend.Dockerfile). |

### Implementation [2]

| Rubric question | Where to look |
|---|---|
| Python coding style | ruff ([ruff.toml](ruff.toml)) passes on all code. Evidence: [23](evidence/23-unit-tests-frontend-and-lint.png). |
| Logging | Python `logging` in 16 modules, for example [src/bikeshare/serving/app.py](src/bikeshare/serving/app.py) and [src/bikeshare/retraining/service.py](src/bikeshare/retraining/service.py) |
| Exception handling | The API maps bad input to 422/404/409 and a missing model to 503 ([src/bikeshare/serving/app.py](src/bikeshare/serving/app.py)). A failed retraining job still resumes the replay ([src/bikeshare/retraining/service.py](src/bikeshare/retraining/service.py)). The frontend catches `BackendError` and shows the message. Evidence: [17](evidence/17-api-error-handling.png). |
| Design document followed; APIs match the spec | Endpoint paths and fields in [docs/lld.md](docs/lld.md) match the live OpenAPI pages ([15](evidence/15-fastapi-backend-docs.png), [16](evidence/16-fastapi-trainer-docs.png)) |
| Inline documentation | Module and function docstrings throughout `src/bikeshare/`. Comments explain the non-obvious choices. |
| Unit tests | 60 backend tests in [tests/](tests/) and 6 frontend tests in [frontend/tests/](frontend/tests/). Evidence: [22](evidence/22-unit-tests-backend.png), [23](evidence/23-unit-tests-frontend-and-lint.png). |

### Testing [1]

| Rubric question | Where to look |
|---|---|
| Test plan | [docs/test-plan.md](docs/test-plan.md), sections 1–3 |
| List of test cases | Section 5: TC-01 to TC-56, TC-F1 to TC-F6, smoke checks SM-01 to SM-17, and end-to-end scenarios E2E-1 to E2E-4 |
| Test report with pass and fail counts | Section 6: 87 of 87 passed. Evidence: [22](evidence/22-unit-tests-backend.png), [23](evidence/23-unit-tests-frontend-and-lint.png), [24](evidence/24-alert-rule-tests.png), [25](evidence/25-smoke-test.png). |
| Acceptance criteria, and whether they were met | Section 4 defines AC1–AC10; section 6 records each one as met |

## AIOps implementation

### Data engineering [2]

| Rubric question | Where to look |
|---|---|
| Ingestion and transformation pipeline on Spark | [src/bikeshare/data/](src/bikeshare/data/) (`ingest`, `clean`, `stations`, `panel`, `build_features`), Spark session in [src/bikeshare/spark.py](src/bikeshare/spark.py). One feature function is shared by Spark (`applyInPandas`) and the online service: [src/bikeshare/features.py](src/bikeshare/features.py). |
| Throughput and speed | 9.1M raw trips cleaned in 50 s; 18.4M station-hours and 14.1M feature rows. Row counts: [04](evidence/04-frontend-pipeline-console.png). Timings: [docs/hld.md](docs/hld.md). |

### Source control and continuous integration [2]

| Rubric question | Where to look | Screenshot |
|---|---|---|
| CI with DVC | [dvc.yaml](dvc.yaml): nine stages, ingest → clean → stations → panel → features → tune → train → profile → calibrate. `dvc repro` reruns only stages whose code, data or params changed. [dvc.lock](dvc.lock) pins every hash. | [05](evidence/05-dvc-dag-and-status.png) |
| DVC DAG | `dvc dag` | [05](evidence/05-dvc-dag-and-status.png) |
| Git / DVC versioning of source, data and models | Code in Git. Data versioned by DVC with a local remote (`.dvc/config`). Models versioned in the MLflow registry; each model is tagged with its git commit and DVC data hash. | [05](evidence/05-dvc-dag-and-status.png), [11](evidence/11-mlflow-registry-versions-and-tags.png), [13](evidence/13-git-history.png) |

### Experiment tracking [2]

| Rubric question | Where to look | Screenshot |
|---|---|---|
| How experiments are tracked | MLflow experiment `bikeshare-demand`. Ray Tune trials are nested runs under each tuning run; training, baselines and retraining each log their own run. Code: [src/bikeshare/modeling/tune.py](src/bikeshare/modeling/tune.py), [train.py](src/bikeshare/modeling/train.py), [tracking.py](src/bikeshare/modeling/tracking.py). | [06](evidence/06-mlflow-experiment-runs.png), [09](evidence/09-mlflow-ray-tune-trials.png) |
| Metrics, parameters, artifacts | 18 parameters, 29 metrics (overall, per horizon, per station tier, peak and off-peak), feature importance, the model, a drift reference profile | [07](evidence/07-mlflow-training-run.png), [08](evidence/08-mlflow-training-artifacts.png) |
| Beyond autolog | Logged by hand, not by autolog: tags `git_commit`, `git_dirty`, `dvc_features_md5` and `stage`; for retrained models also `trigger`, `as_of`, `gate`, `promoted_from` and `rolled_back_at`; and a `gate.json` artifact with every promotion check | [07](evidence/07-mlflow-training-run.png), [11](evidence/11-mlflow-registry-versions-and-tags.png), [12](evidence/12-mlflow-retrain-gate-result.png) |
| Model registry | `bikeshare-demand`. The `champion` alias marks the serving version; version 2 was rejected by the gate, version 3 was promoted and then rolled back. | [10](evidence/10-mlflow-registry-champion.png), [11](evidence/11-mlflow-registry-versions-and-tags.png) |

### Exporter instrumentation and visualization [2]

| Rubric question | Where to look | Screenshot |
|---|---|---|
| Prometheus instrumentation | `prometheus_client` in the backend ([serving/app.py](src/bikeshare/serving/app.py), [monitoring/replay.py](src/bikeshare/monitoring/replay.py)) and the trainer ([retraining/service.py](src/bikeshare/retraining/service.py)). Scrape config: [monitoring/prometheus/prometheus.yml](monitoring/prometheus/prometheus.yml). | [18](evidence/18-prometheus-targets.png) |
| What is monitored | Requests, latency and errors by endpoint; forecasts by source; model version; replay clock; 7-day WAPE and relative WAPE by target, horizon and tier; PSI per feature; breach streaks; retraining jobs and rollbacks. Full list: [docs/lld.md](docs/lld.md), *Monitoring*. | [21](evidence/21-grafana-dashboard.png) |
| All components monitored | Backend, trainer, Alertmanager and Prometheus itself are scraped, all `UP`. Frontend and MLflow have health checks in Compose. | [18](evidence/18-prometheus-targets.png), [14](evidence/14-docker-compose-services.png) |
| Alerting | Five rules in [monitoring/prometheus/rules.yml](monitoring/prometheus/rules.yml) (tested with promtool). Alertmanager sends `action=retrain` to the trainer's webhook ([alertmanager.yml](monitoring/alertmanager/alertmanager.yml)). | [19](evidence/19-prometheus-alert-rules.png), [20](evidence/20-alertmanager-routing.png), [24](evidence/24-alert-rule-tests.png) |
| Grafana in near real time | One provisioned dashboard, refreshing every 5 s: [monitoring/grafana/dashboards/bikeshare.json](monitoring/grafana/dashboards/bikeshare.json). The screenshot covers 12 hours, including a retraining run (champion version 1 → 3 → 1). | [21](evidence/21-grafana-dashboard.png) |

### Software packaging [2]

| Rubric question | Where to look | Screenshot |
|---|---|---|
| MLflow for APIfication | **Partly.** The model is logged as an MLflow pyfunc ([src/bikeshare/modeling/pyfunc.py](src/bikeshare/modeling/pyfunc.py)) and FastAPI loads it from the MLflow registry by the `champion` alias. It is not served with `mlflow models serve`; the reason is in [docs/hld.md](docs/hld.md) (D7). | [10](evidence/10-mlflow-registry-champion.png) |
| MLprojects for identical dev and test environments | **Not done.** Environments are pinned instead: `requirements.lock` for the host and the backend image, and `frontend/requirements.lock` for the frontend image. | |
| FastAPI for the UI's APIs | Backend [src/bikeshare/serving/app.py](src/bikeshare/serving/app.py); trainer [src/bikeshare/retraining/service.py](src/bikeshare/retraining/service.py) | [15](evidence/15-fastapi-backend-docs.png), [16](evidence/16-fastapi-trainer-docs.png) |
| Backend and frontend dockerized | [docker/backend.Dockerfile](docker/backend.Dockerfile) (multi-stage: backend and trainer), [frontend/Dockerfile](frontend/Dockerfile) | [14](evidence/14-docker-compose-services.png) |
| Docker Compose with separate services | [compose.yaml](compose.yaml): frontend, backend, trainer, MLflow, Prometheus, Alertmanager, Grafana | [14](evidence/14-docker-compose-services.png) |

## Screenshot index

| # | File | Shows |
|---|---|---|
| 01 | [frontend-rebalancing-alerts](evidence/01-frontend-rebalancing-alerts.png) | Main dispatcher page: ranked lists and map |
| 02 | [frontend-station-forecast](evidence/02-frontend-station-forecast.png) | One station: history, forecast, what happened |
| 03 | [frontend-model-monitoring](evidence/03-frontend-model-monitoring.png) | Replay after 1–10 March: error and drift windows |
| 04 | [frontend-pipeline-console](evidence/04-frontend-pipeline-console.png) | Serving model, retraining runs, test results, data pipeline |
| 05 | [dvc-dag-and-status](evidence/05-dvc-dag-and-status.png) | `dvc dag`, `dvc status`, DVC remote |
| 06 | [mlflow-experiment-runs](evidence/06-mlflow-experiment-runs.png) | All runs: tuning, training, baselines, retraining |
| 07 | [mlflow-training-run](evidence/07-mlflow-training-run.png) | Training run: tags, metrics, parameters |
| 08 | [mlflow-training-artifacts](evidence/08-mlflow-training-artifacts.png) | Training run artifacts |
| 09 | [mlflow-ray-tune-trials](evidence/09-mlflow-ray-tune-trials.png) | Tuning run with its Ray Tune child runs |
| 10 | [mlflow-registry-champion](evidence/10-mlflow-registry-champion.png) | Registry versions and the `champion` alias |
| 11 | [mlflow-registry-versions-and-tags](evidence/11-mlflow-registry-versions-and-tags.png) | Every version's lineage and gate tags |
| 12 | [mlflow-retrain-gate-result](evidence/12-mlflow-retrain-gate-result.png) | `gate.json` of the promoted challenger |
| 13 | [git-history](evidence/13-git-history.png) | Commit history and remote |
| 14 | [docker-compose-services](evidence/14-docker-compose-services.png) | Seven running services |
| 15 | [fastapi-backend-docs](evidence/15-fastapi-backend-docs.png) | Backend endpoints |
| 16 | [fastapi-trainer-docs](evidence/16-fastapi-trainer-docs.png) | Trainer endpoints |
| 17 | [api-error-handling](evidence/17-api-error-handling.png) | 422, 404 and 409 responses |
| 18 | [prometheus-targets](evidence/18-prometheus-targets.png) | All scrape targets up |
| 19 | [prometheus-alert-rules](evidence/19-prometheus-alert-rules.png) | Five alert rules |
| 20 | [alertmanager-routing](evidence/20-alertmanager-routing.png) | `action=retrain` routed to the trainer |
| 21 | [grafana-dashboard](evidence/21-grafana-dashboard.png) | Full dashboard over 12 hours |
| 22 | [unit-tests-backend](evidence/22-unit-tests-backend.png) | 60 backend tests passed |
| 23 | [unit-tests-frontend-and-lint](evidence/23-unit-tests-frontend-and-lint.png) | 6 frontend tests passed, ruff clean |
| 24 | [alert-rule-tests](evidence/24-alert-rule-tests.png) | promtool and amtool checks |
| 25 | [smoke-test](evidence/25-smoke-test.png) | 17 smoke checks against the running stack |

## Known gaps

- **No MLproject file.** Environments are pinned with lockfiles instead.
- **No `mlflow models serve`.** The pyfunc is loaded inside FastAPI. The logged model would work with `mlflow models serve -m models:/bikeshare-demand@champion` unchanged.
- **No Git LFS and no GitHub Actions.** Data goes through DVC, and the DVC pipeline, tests and smoke test are run by hand.
- **Retrained versions 2 and 3 show `git_dirty = True`.** They were trained while the Milestone 6 code was not yet committed. The flag records this honestly; the code was committed afterwards.
