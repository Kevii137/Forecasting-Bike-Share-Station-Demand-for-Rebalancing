# Low-level design: REST contracts and data flow

## Service addresses and configuration

| Block | Local address | Configuration |
|---|---|---|
| Frontend (Streamlit) | `http://localhost:8501` | `BACKEND_URL`, `TRAINER_URL` |
| Backend (FastAPI) | `http://localhost:8000`, docs at `/docs` | `MLFLOW_TRACKING_URI`, `PANEL_PATH`, `STATIONS_PATH`, `DEFAULT_ORIGIN`; `params.yaml` mounted read-only |
| Trainer (FastAPI) | `http://localhost:8001`, docs at `/docs` | `MLFLOW_TRACKING_URI`, `BACKEND_URL`; the repository mounted read-only |
| MLflow | `http://localhost:5000` | `mlflow.db` and `mlartifacts/` |
| Prometheus / Alertmanager / Grafana | `:9090` / `:9093` / `:3000` | `monitoring/`; Grafana password from `.env` |

Ports bind to localhost. Inside Compose, services call each other by service name (`http://backend:8000`). The frontend calls only the backend and the trainer. Thresholds, the gate tolerance and retraining settings live in `params.yaml`, so changing them needs a restart, not a rebuild.

All bodies are JSON. Times are Chicago wall-clock hours, written `YYYY-MM-DDTHH:MM`. The *origin* is the last complete hour of data; forecasts cover the 3 hours after it. Errors return `{"detail": "..."}`.

## `GET /alerts`

The main endpoint: what the dispatcher's page calls.

| Parameter | Type | Accepted values |
|---|---|---|
| `origin` | string, optional | an hour from `2026-03-01T00:00` to `2026-09-30T20:00`; default `DEFAULT_ORIGIN` |
| `limit` | integer, optional | 1–100 stations per list; default 10 |

Example: `GET /alerts?origin=2026-07-15T17:00&limit=1`

How the backend answers:
1. Get that day's features from the cache, or compute them from the panel (about 15–35 s for a new day).
2. Busy classic stations (5+ trips/day) get the champion model's forecast; the rest get the 8-week same-hour average, marked low confidence.
3. Sum each station's forecast over the 3 hours. `net_flow = dropoffs - pickups` and `pressure = net_flow / capacity`.
4. List model stations with `pressure <= -0.1` under `draining` and `>= 0.1` under `filling`, strongest first.

Example HTTP 200 response:

```json
{"origin": "2026-07-15T17:00", "timezone": "America/Chicago", "horizons": [1, 2, 3],
 "model_version": "1", "min_pressure": 0.1, "stations_forecast": 1640, "stations_on_model": 560,
 "draining": [{"station_key": "Montrose Harbor", "pickups": 49.58, "dropoffs": 30.11,
               "net_flow": -19.46, "pressure": -0.628, "capacity": 31.0, "source": "model",
               "confidence": "normal", "station_type": "classic", "trips_per_day_28d": 120.1,
               "lat": 41.964, "lng": -87.638}],
 "filling": [{"station_key": "Clark St & Newport St", "net_flow": 9.01, "pressure": 0.563, "...": "..."}]}
```

Errors: 422 for an origin outside the replay window or not on the hour, 404 if no station is active, 503 if no model is loaded.

## Other backend endpoints

| Method and path | Input | Output / status |
|---|---|---|
| `GET /forecast` | `origin` | Same envelope, plus `stations`: every active station with the fields above |
| `GET /station` | `key` (required), `origin`, `history_hours` 1–168 | `station`; `forecast` per horizon (`target_hour`, `pickups_forecast`, `dropoffs_forecast`, `source`); `history` and `observed_after_origin` (`hour`, `pickups`, `dropoffs`). 404 if the station is not active. |
| `GET /stations` | none | All 2,037 stations: `station_key`, `lat`, `lng`, `capacity`, `station_type` |
| `GET /replay` | none | `first_origin`, `last_origin`, `default_origin`, `timezone` |
| `GET /model` | none | `version`, `run_id`, `loaded_at`, lineage `tags` (`git_commit`, `git_dirty`, `dvc_features_md5`, and for retrained models `trigger`, `as_of`, `gate`, `promoted_from`) |
| `POST /model/reload` | none | `{"reloaded": bool, "version": str}`. Checks the registry now instead of waiting for the 30 s poll. |
| `GET /pipeline` | none | DVC report files: `data_quality`, `stations`, `panel`, `features`, `metrics` |
| `GET /monitoring` | none | Replay status (`running`, `paused`, `clock`, `model_version`, `window_days`, `breach_days`, `alerting`, `thresholds`, `error`), `latest` 7-day window by target, horizon and tier, and one `history` row per simulated day |
| `POST /monitoring/replay/start` | `first`, `last`, `delay_seconds` 0–60 | Status. 409 if already running, 422 for bad times. |
| `POST /monitoring/replay/stop`, `/pause`, `/resume` | none | Status. Pause returns once the hour in progress finishes; 409 if not running, 504 on timeout. |
| `GET /health`, `GET /ready` | none | Liveness; readiness (`{"ready": true, "model_version": "1"}` or 503) |
| `GET /metrics` | none | Prometheus text format |

## Trainer endpoints

| Method and path | Input | Output / status |
|---|---|---|
| `POST /alertmanager` | Alertmanager webhook body | `{"decision": "started", "job": {...}}` or `{"decision": "ignored", "reason": "..."}` |
| `POST /retrain` | `as_of` (optional) | Starts a job. Without `as_of` it uses the replay clock and pauses the replay. Skips the cooldown. 409 if a job is running. |
| `POST /rollback` | none | `{"from": "3", "to": "1", "backend_version": "1"}`. 409 if there is nothing to roll back to. |
| `GET /status` | none | `running`, the current `job` (`trigger`, `as_of`, `phase`), `last_as_of`, `champion_version`, recent results with gate details |
| `GET /jobs` | `limit` 1–100 | Retraining runs from MLflow: `as_of`, `trigger`, `gate`, `outcome`, `version`, `rolled_back_at`, and challenger and champion WAPE |
| `GET /health`, `GET /metrics` | none | Liveness; Prometheus text format |

Webhook: the trainer reads only `alerts[].status` and `alerts[].labels`. It starts a job when an alert is `firing` with `action=retrain`, no job is running, and the replay clock is at least 7 simulated days past the last attempt. Example body:

```json
{"alerts": [{"status": "firing", "labels": {"alertname": "BikeshareForecastDegraded", "action": "retrain"}}]}
```

## Retraining job and gate

`python -m bikeshare.retraining.job --as-of 2026-03-12T00:00` (add `--dry-run` to write nothing):

1. Split data observed up to `as_of` by target hour: gate = last 14 days; early stopping = the 14 days before; fit = 1 Aug 2025 up to that point.
2. Fit pickups and drop-offs LightGBM models with the Ray Tune settings from `reports/best_params.json`.
3. Score the challenger, the champion and both baselines on the gate rows.
4. Gate, for each target:

   | Check | Passes if |
   |---|---|
   | Beat the champion | challenger overall WAPE < champion overall WAPE |
   | Within tolerance | on every horizon and tier, challenger WAPE ≤ champion WAPE × 1.03 |
   | Beat each baseline | challenger WAPE < baseline WAPE, unless the champion also trails that baseline; then beating the champion is enough (logged as waived) |

5. Log an MLflow run (`stage=retrain`) with the gate metrics, `gate.json`, the drift reference profile and the model. Register it as a new version.
6. If the gate passed and the champion has not changed, tag the new version `promoted_from=<old>` and move `champion` to it.

Rollback reads the champion's `promoted_from` tag and moves `champion` back to that version.

## Model input and output

- **Input:** 29 float64 columns:
  - `horizon`;
  - for pickups and for drop-offs: `lag_0`, `lag_1`, `lag_2`, `mean_3h`, `mean_24h`, `mean_168h`, `same_hour_1d`, `same_hour_1w`, `same_hour_4w_mean`, `same_hour_8w_mean`;
  - `trips_per_day_28d`, `target_hour_of_day`, `target_day_of_week`, `target_is_weekend`, `target_is_holiday`, `capacity`, `lat`, `lng`.
- **Output:** `pickups_forecast` and `dropoffs_forecast`, both non-negative.

## Monitoring and offline operations

- **Replay monitor:** at each simulated hour it forecasts every station and keeps the model-routed rows. At 23:00 it scores the day's due forecasts against the panel, adds the day to a 7-day window and updates two streaks:
  - error: model WAPE / fallback WAPE above 1.0;
  - drift: largest PSI above 0.5.
- **Prometheus metrics:**
  - backend: `bikeshare_requests_total`, `bikeshare_request_seconds`, `bikeshare_station_forecasts_total{source}`, `bikeshare_model_version`;
  - replay: `bikeshare_replay_*`, `bikeshare_monitor_wape`, `bikeshare_monitor_relative_wape`, `bikeshare_monitor_psi`, `bikeshare_monitor_breach_days`, `bikeshare_monitor_threshold`;
  - trainer: `bikeshare_retrain_running`, `bikeshare_retrain_jobs_total{outcome}`, `bikeshare_rollbacks_total`.
- **Alert rules** (`monitoring/prometheus/rules.yml`) fire when `breach_days` reaches the `sustain_days` threshold gauge, so thresholds exist only in `params.yaml`. promtool tests them.
- **Offline:**
  - `dvc repro` reruns changed stages.
  - `dvc repro calibrate` recomputes the threshold evidence.
  - The retraining job above can be run by hand.

## Files and recovery

| Where | What |
|---|---|
| Git | Source, `params.yaml`, `dvc.yaml` and `dvc.lock`, reports, monitoring config, docs |
| DVC remote (`../dvc-storage`) | Raw and processed data |
| `mlflow.db`, `mlartifacts/` | Runs, models and the registry (git-ignored) |
| `.env` | Grafana admin password (git-ignored) |

After a fresh clone:

```bash
dvc pull
cp .env.example .env
docker compose up -d --build
python3 scripts/smoke_test.py
```

If `mlflow.db` is empty, run `dvc repro train` and `python -m bikeshare.modeling.register` first.
