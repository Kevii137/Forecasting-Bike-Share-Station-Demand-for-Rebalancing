# Test plan, test cases and test report

## 1. Scope and objectives

Testing covers the whole system: data pipeline, features, models, the backend API, the frontend, monitoring, retraining, the promotion gate, rollback and the Docker deployment. It aims to show that:

1. each component behaves correctly, including at its edges (unit tests);
2. components work together: online and offline features agree, the API serves the registered model, retraining registers and promotes through MLflow (integration tests);
3. the deployed stack is up and wired correctly (smoke test);
4. the full monitoring and retraining loop works without manual steps (end-to-end tests);
5. the system meets the acceptance criteria in section 4.

## 2. Test levels and tools

| Level | Tool | Where | Count |
|---|---|---|---|
| Unit and integration, backend and pipeline | pytest | `tests/` | 60 |
| Unit, frontend (pages rendered against a fake backend) | pytest, Streamlit `AppTest` | `frontend/tests/` | 6 |
| Alert rules | promtool | `monitoring/prometheus/rules_test.yml` | 2 groups, 5 assertions |
| Alertmanager configuration and routing | amtool | `monitoring/alertmanager/alertmanager.yml` | config check, 2 routes |
| Deployed system (smoke) | `scripts/smoke_test.py` | the running Compose stack | 17 checks |
| End-to-end | replay, run by hand (section 5.4) | the running Compose stack | 4 scenarios |
| Static analysis | ruff (about 400 rules) | all Python | 1 |

## 3. Test environment

- One laptop: Linux, 14 GB of memory, Python 3.12, Java 17, Docker with Compose.
- Real data: Divvy trips from June 2025 to September 2026, restored with `dvc pull`. Two tests use it and skip themselves when it is absent.
- Synthetic data: most tests build small panels, feature tables and MLflow stores in temporary folders, so they run in seconds and touch no real state.
- Stand-ins: the frontend tests replace the backend with recorded responses; the trainer tests replace the backend and the job with fakes; the registry tests use a fake client or a temporary SQLite MLflow store.

## 4. Acceptance criteria

| ID | Criterion |
|---|---|
| AC1 | All unit and integration tests pass, backend and frontend, with no failures |
| AC2 | All smoke checks pass on a freshly started stack |
| AC3 | The model beats both baselines on test-month WAPE, overall and at every horizon |
| AC4 | Online features equal the offline (Spark) features exactly |
| AC5 | A sustained forecast-error alert leads, without manual steps, to a retraining job whose gate decision is recorded in MLflow, and the replay resumes afterwards |
| AC6 | A challenger that passes the gate is served by the backend; a rollback restores the previous version |
| AC7 | A forecast for an hour of an already-prepared day returns in under 1 s; a new day in under 60 s |
| AC8 | A retraining job finishes within the trainer's 6 GB memory limit and in under 20 minutes |
| AC9 | Static checks pass: ruff, promtool rule tests, amtool configuration check |
| AC10 | Every frontend page renders without errors, and an unreachable backend produces a clear message |

## 5. Test cases

### 5.1 Backend and pipeline (pytest, `tests/`)

| ID | Test | Verifies |
|---|---|---|
| **Environment** (`test_environment.py`) | | |
| TC-01 | `test_spark_apply_in_pandas` | Spark runs pandas functions per group, as the feature stage needs |
| TC-02 | `test_spark_preserves_hourly_timestamps_through_pandas` | Timestamps survive Spark–pandas transfer unchanged, including the daylight-saving day |
| TC-03 | `test_lightgbm_poisson_is_non_negative` | The Poisson objective never forecasts negative demand |
| TC-04 | `test_ray_tune_runs_parallel_trials` | Ray Tune runs trials in parallel |
| TC-05 | `test_mlflow_logs_to_sqlite` | MLflow logs runs to a SQLite store |
| **Data stages** (`test_spark_stages.py`, `test_stations.py`) | | |
| TC-06 | `test_spark_reads_timestamps_written_the_way_stations_are` | The station stage reads its own timestamps back correctly |
| TC-07 | `test_rules_reject_bad_trips_and_count_each_rule` | Trip validation rejects bad rows and counts each reason |
| TC-08 | `test_panel_fills_zeros_only_between_first_and_last_activity` | The panel adds zero-trip hours only while a station exists |
| TC-09 | `test_normalise_name` | Station names are normalised consistently |
| TC-10 | `test_nearby_names_sharing_an_id_are_renames` | A renamed station keeps one identity |
| TC-11 | `test_distant_names_sharing_an_id_stay_separate` | A reused ID at a different site gives a different station |
| TC-12 | `test_unrelated_stations_stay_separate` | Unrelated stations are never merged |
| **Features** (`test_features.py`) | | |
| TC-13 | `test_changing_data_after_origin_does_not_change_features` | No leakage: data after the forecast time cannot affect inputs |
| TC-14 | `test_target_is_demand_at_origin_plus_horizon` | Targets line up with the horizon |
| TC-15 | `test_seasonal_features_look_back_from_the_target_hour` | Same-hour history is taken relative to the target hour |
| TC-16 | `test_calendar_describes_target_hour_and_holidays` | Calendar and holiday flags describe the target hour |
| TC-17 | `test_unobserved_targets_are_missing_not_zero` | Unknown future demand is missing, not zero |
| TC-18 | `test_rejects_more_than_one_station` | The feature function refuses mixed-station input |
| **Modelling** (`test_modeling.py`) | | |
| TC-19 | `test_wape_is_total_error_over_total_demand` | WAPE definition |
| TC-20 | `test_poisson_deviance_is_zero_for_a_perfect_forecast` | Deviance definition |
| TC-21 | `test_poisson_deviance_handles_zero_forecasts_without_warnings` | Deviance is safe at zero forecasts |
| TC-22 | `test_volume_tiers_are_half_open` | Tier boundaries (5–10, 10–20, 20+) |
| TC-23 | `test_net_flow_sums_all_horizons_and_skips_incomplete_origins` | Net-flow error is computed over complete 3-hour windows only |
| TC-24 | `test_baselines_read_the_seasonal_features` | Baselines use the right history columns |
| TC-25 | `test_pyfunc_returns_non_negative_forecasts_for_both_targets` | The packaged model returns both forecasts, never negative |
| TC-26 | `test_pyfunc_rejects_missing_feature_columns` | The packaged model refuses incomplete input |
| TC-27 | `test_lineage_ignores_files_the_pipeline_rewrites` | Lineage tags flag changed code, not files the pipeline rewrites |
| **Serving** (`test_serving.py`) | | |
| TC-28 | `test_online_features_equal_build_features_on_full_history` | Online features equal the offline function's output |
| TC-29 | `test_origin_must_be_on_the_hour_and_inside_the_window` | Invalid forecast times are rejected |
| TC-30 | `test_routing_fallback_and_alerts` | Quiet and non-classic stations get the fallback and raise no alerts; alerts rank by pressure |
| TC-31 | `test_fallback_uses_shorter_averages_for_new_stations` | New stations fall back to shorter averages |
| TC-32 | `test_model_store_reloads_only_when_alias_moves` | The champion reloads only when the alias moves; a registry outage keeps the current model |
| TC-33 | `test_model_load_does_not_leave_bundled_code_on_the_path` | Loading a model does not shadow the service's code |
| TC-34 | `test_api_serves_alerts_station_detail_and_metrics` | `/alerts`, `/station` and `/metrics` respond correctly |
| TC-35 | `test_api_rejects_bad_origins_and_unknown_stations` | 422 and 404 responses |
| TC-36 | `test_online_features_match_offline_spark_features` | **Real data:** online inputs equal the Spark values bit for bit on three replay days, including 4 July |
| **Monitoring** (`test_monitoring.py`) | | |
| TC-37 | `test_psi_is_near_zero_for_the_reference_and_large_for_a_shift` | PSI behaves as expected |
| TC-38 | `test_zero_heavy_reference_keeps_zeros_in_a_bin_of_their_own` | Zero-heavy demand is binned correctly |
| TC-39 | `test_window_sums_days_before_dividing` | Rolling WAPE sums errors before dividing |
| TC-40 | `test_replay_scores_observed_hours_and_publishes_signals` | The replay scores forecasts against the panel and publishes the gauges |
| TC-41 | `test_new_champion_restarts_the_windows` | A new champion restarts the windows |
| TC-42 | `test_replay_api_starts_stops_and_rejects_overlap` | Replay endpoints, 409 on overlap, 422 on bad time |
| TC-43 | `test_live_day_record_equals_the_offline_calibration_record` | **Real data:** the live monitor computes exactly what the threshold calibration computed |
| **Retraining** (`test_retraining.py`) | | |
| TC-44 | `test_windows_end_at_as_of_and_do_not_overlap` | Fit, early-stopping and gate windows end at the simulated time and do not overlap |
| TC-45 | `test_load_rows_selects_by_target_hour_across_months` | Rows are selected by target hour across month partitions |
| TC-46 | `test_gate_passes_a_better_challenger_within_tolerance` | The gate passes a better challenger; tolerance is inclusive |
| TC-47 | `test_gate_names_every_failed_check` | A rejection lists every failed check |
| TC-48 | `test_baseline_rule_is_waived_only_when_the_champion_also_trails_it` | The owner's baseline rule (D9 amendment) |
| TC-49 | `test_gate_requires_strictly_beating_the_champion` | A tie with the champion is not enough |
| TC-50 | `test_promote_then_roll_back_restores_the_previous_champion` | Promotion and rollback move the alias correctly; a second rollback is refused |
| TC-51 | `test_promotion_refuses_when_the_champion_moved_during_training` | A stale gate result cannot promote |
| TC-52 | `test_retrain_alert_pauses_trains_reloads_and_resumes` | The trainer pauses, trains, reloads and resumes; drift alerts are ignored; the cooldown holds, then lapses |
| TC-53 | `test_a_failed_job_still_resumes_the_replay` | A crashed job (for example out of memory) still resumes the replay |
| TC-54 | `test_only_one_job_runs_and_manual_jobs_skip_the_cooldown` | One job at a time (409); manual jobs; rollback refused with no history |
| TC-55 | `test_pausing_the_replay_holds_the_clock` | Pause and resume of the simulated clock |
| TC-56 | `test_job_trains_gates_registers_and_promotes` | **Integration:** a full job against a temporary MLflow store trains, gates, logs, registers, promotes, serves and rolls back |

The 60 backend tests are TC-01 to TC-56 plus parametrised variants.

### 5.2 Frontend (pytest, `frontend/tests/`)

| ID | Test | Verifies |
|---|---|---|
| TC-F1 | `test_alerts_page_lists_both_directions` | Alert tables and counts |
| TC-F2 | `test_station_page_starts_with_the_top_alert` | Station picker order, the observed-after table |
| TC-F3 | `test_pipeline_console_shows_lineage_and_reloads` | Lineage display and the registry check button |
| TC-F4 | `test_pipeline_console_shows_retraining_jobs_and_gate_reasons` | Job in progress, rejection reasons, jobs table, Retrain disabled while busy, confirmed rollback |
| TC-F5 | `test_unreachable_backend_shows_an_error` | A clear message when the backend is down |
| TC-F6 | `test_monitoring_page_shows_signals_and_the_firing_alert` | Monitoring metrics, alert warning, PSI and slice tables |

### 5.3 Smoke test (`scripts/smoke_test.py`, running stack)

| ID | Check |
|---|---|
| SM-01 | MLflow server is healthy |
| SM-02 | Backend has loaded the champion model |
| SM-03 | Alerts are served |
| SM-04 | Station detail has one forecast per horizon |
| SM-05 | An origin outside the replay window is rejected |
| SM-06 | Prometheus metrics are exposed |
| SM-07 | Frontend is healthy |
| SM-08 | Replay monitor reports its thresholds |
| SM-09 | Prometheus scrapes the backend, trainer and Alertmanager |
| SM-10 | Prometheus has loaded the drift and error alert rules |
| SM-11 | Prometheus sends alerts to Alertmanager |
| SM-12 | Alertmanager is ready |
| SM-13 | Trainer reports its status and the champion |
| SM-14 | Trainer ignores an advisory drift alert |
| SM-15 | Trainer can read the git commit for lineage tags |
| SM-16 | Grafana serves the provisioned dashboard |
| SM-17 | The frontend container reaches the backend and trainer over the Compose network |

### 5.4 End-to-end scenarios (running stack, by hand)

| ID | Steps | Expected |
|---|---|---|
| E2E-1 | Start a replay from 1 March at full speed | Both alerts fire at simulated 11 March 23:00 (7 days to fill the window, then 5 breaching days) |
| E2E-2 | Let the error alert reach the trainer | Alertmanager calls the trainer; the replay pauses; a job trains, gates and registers a challenger; the replay resumes |
| E2E-3 | Let the alert repeat after the job | The trainer ignores repeats within 7 simulated days |
| E2E-4 | After a promotion, press *Roll back the champion*, then again | The backend serves the previous version; the second rollback is refused |

Before E2E-2, restart the trainer (`docker compose restart trainer`) if an earlier run used the same simulated time, because the cooldown is kept in its memory.

## 6. Test report (2026-11-14)

| Suite | Tests | Passed | Failed | Skipped |
|---|---|---|---|---|
| Backend and pipeline (pytest) | 60 | 60 | 0 | 0 |
| Frontend (pytest) | 6 | 6 | 0 | 0 |
| Prometheus rule tests (promtool) | 1 file | pass | 0 | |
| Alertmanager configuration (amtool) | 1 | pass | 0 | |
| Smoke test | 17 | 17 | 0 | 0 |
| End-to-end scenarios | 4 | 4 | 0 | 0 |
| Static analysis (ruff) | all files | pass | 0 | |
| **Test cases (pytest, smoke, end-to-end)** | **87** | **87** | **0** | **0** |

The backend suite took 95 s and the frontend suite 2 s.

### End-to-end results

- **First run (gate as originally written, 7-day early stopping):**
  - The error alert fired at simulated 11 March 23:00. Alertmanager called the trainer 6 s later, and the replay paused at 12 March 00:00.
  - Version 2 was trained in 5 minutes and rejected. It did not beat the historical average on drop-offs, and neither did the champion; with the unstable early stopping it also lost to the champion on pickups.
  - The replay resumed. A repeat alert 23 simulated hours later was ignored by the cooldown.
  - These findings led to two owner decisions: a challenger only has to beat a baseline the champion also beats, and early stopping uses 14 days instead of 7.
- **Second run (amended gate, 14-day early stopping):**
  - Version 3 was trained in 12.5 minutes and **promoted**: pickups WAPE 0.8379 against the champion's 0.8420, drop-offs 0.8337 against 0.8387. The drop-offs baseline check was waived because the champion also trailed the historical average.
  - The backend served version 3 at once. After the replay resumed, the monitor restarted its windows for version 3.
  - A rollback restored version 1, and a second rollback was refused.

### Acceptance criteria

| ID | Result | Evidence |
|---|---|---|
| AC1 | Met | 66 of 66 pytest tests passed |
| AC2 | Met | 17 of 17 smoke checks |
| AC3 | Met | February 2026 WAPE: pickups 0.817 against 0.842 and 0.991, drop-offs 0.819 against 0.826 and 0.963; better at every horizon (`reports/metrics.json`) |
| AC4 | Met | TC-28 and TC-36: identical on synthetic data and on three real replay days |
| AC5 | Met | E2E-2: MLflow runs `a95e8a63` (version 2, rejected) and `0216c3a8` (version 3, promoted) record the gate decision, trigger and simulated time |
| AC6 | Met | E2E-4: the backend served version 3, then version 1 after the rollback |
| AC7 | Met | A cached hour returned in 0.08 s, a new day in 21 s (`/alerts`, in Docker) |
| AC8 | Met | Jobs took 5 to 12.5 minutes. Peak memory was 4.9 GB in the capped dry run and 3.8 GB sampled every 5 s in Docker, under the 6 GB limit. |
| AC9 | Met | ruff, promtool and amtool all pass |
| AC10 | Met | TC-F1 to TC-F6 |

### Defects found during testing and fixed

| Defect | Found by | Fix |
|---|---|---|
| A one-hour shift in retraining time changed the challenger's quality (early stopping on 7 days) | E2E-2 against the dry run | Early stopping on 14 days (owner decision) |
| The gate rejected a challenger better than the champion when both trailed a baseline | Dry run | Baseline rule amended (owner decision) |
| MLflow returns version numbers as int locally and as text over HTTP, which could block a promotion | TC-56 | Versions normalised to text |
| DVC marked training out of date after test runs (bytecode was hashed) | `dvc status` | Bytecode ignored by DVC |
| A replay computing a new day blocked dispatcher requests for days already prepared | Load during the replay | The cache lock covers only the cache |
| The champion version showed 0 in Grafana after a restart | Dashboard review | Read from the loaded model at scrape time |

## 7. How to run

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests                    # backend and pipeline
cd frontend && ../.venv-frontend/bin/python -m pytest tests && cd ..  # frontend
docker run --rm -v "$PWD/monitoring/prometheus:/cfg:ro" --entrypoint promtool \
  prom/prometheus:v3.5.0 test rules /cfg/rules_test.yml               # alert rules
docker run --rm -v "$PWD/monitoring/alertmanager:/cfg:ro" --entrypoint amtool \
  prom/alertmanager:v0.28.1 check-config /cfg/alertmanager.yml       # Alertmanager
.venv/bin/ruff check .                                                # static analysis
docker compose up -d --build && python3 scripts/smoke_test.py         # smoke test
```
