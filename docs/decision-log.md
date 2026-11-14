# Decision log

Design decisions for the project, the alternatives considered, and why each was chosen. Decisions were made by the project owner. Newest revisions are recorded in place with a date.

## Evidence behind the decisions

Measured on 2026-10-31 from Divvy trip files (Aug 2025, Jan 2026, Aug 2026, with Apr 2024 and May to Jul 2025 used for station ID checks) and one snapshot of the live GBFS feed. Volumes are for Aug 2026 unless stated.

| Finding | Measurement |
|---|---|
| Trip files contain no inventory and no bike ID | 13-column schema |
| GBFS gives capacity and current bikes only, no history | 2,078 stations: 1,176 classic, 902 lightweight |
| Long-run inventory cannot be derived from trips | Month-end cumulative net flow: median 1.3x capacity, p90 4.1x |
| Short-run net flow carries signal | 6-hour net flow at or above 50% of capacity in 7.1% of station-hours (top stations) |
| Trips without a start station | 21.8%, all e-bikes, coordinates rounded to 1-2 decimals |
| Station ID format change | 0% `CHI` IDs in May 2025, 100% in June 2025 |
| Station sparsity | 1,574 active, 629 at 5+ trips/day, top 500 carry 91.4% of trips |
| Zero station-hours | 79% overall, 47% in the top 500 |
| Seasonal swing | 137,787 trips in Jan 2026, 868,191 in Aug 2026 |

## Decisions

### D1. Alert logic: demand pressure

- **Decision:** rank stations by forecast net flow over the horizon divided by capacity, labelled "likely draining" or "likely filling". The application never presents an occupancy figure.
- **Alternatives:** assumed starting inventory (result depends on an invented value); live-feed recording to validate against real empties.
- **Why:** historical inventory is not available. Live-feed validation was dropped because November trip data will not be published before the 2026-11-10 deadline, so recorded inventory could not be matched to trips in time.
- **Evaluation:** alerts are scored against realised net flow from trip data. They cannot be scored against real stockouts.
- **Known limit:** trips record satisfied demand, so an empty station shows low pickups. This cannot be corrected historically.

### D2. Station coverage: rule-based tiers

- **Decision:** classic stations with at least 5 trips/day over the trailing 28 days use the model. All others get the historical-average forecast, marked low confidence, and raise no alerts. Trips with no station are excluded and their share is tracked as a data-quality metric.
- **Alternatives:** model every station (mostly zeros, noisy alerts); fixed top N (ignores new stations).
- **Why:** every station is still served, as the project statement asks, without letting sparse stations produce noise.

### D3. Forecast structure: global model, 1 to 3 hours ahead

- **Decision:** one global LightGBM model per target (pickups, drop-offs) with hours-ahead as an input feature, using a Poisson objective. Horizons are 1, 2 and 3 hours.
- **Alternatives:** recursive one-step model (errors compound); direct model per horizon (12 models at a 6-hour horizon).
- **Why:** owner's choice. Two models are simpler to tune, package and explain.
- **Consequence:** rows are repeated once per horizon, and a shorter horizon gives dispatchers less lead time. All features must be computed from data available at forecast time.

### D4. Data window and evaluation: short window

- **Decision:** train Aug to Dec 2025, validate Jan 2026, test Feb 2026, replay Mar to Sep 2026. Ingest from Jun 2025 to provide history for lags and baselines. Splits are chronological.
- **Alternatives:** two-year training window (drift may not trigger); added cross-season backtests (more compute).
- **Why:** a model that has not seen warm-season surge gives a clear drift and retraining demonstration.
- **Consequence:** the short window is deliberate and must be disclosed in the report. February is a low-volume test month, so baselines are also scored through the replay months.

### D5. Features: no weather at first

- **Decision:** start with lag, rolling, calendar and station features. Add observed weather at forecast time as a tracked experiment, kept only if it clearly reduces error.
- **Alternatives:** archived weather forecasts (more integration work).
- **Why:** weather at the target hour would be leakage. The experiment is the first item to cut if time runs short.

### D6. Feature code path: one shared function

- **Decision:** one pandas feature function, run through Spark `applyInPandas` offline and called directly online. A parity test compares the two on sampled hours.
- **Alternatives:** precomputed replay features (not a real serving path); separate Spark and online implementations (risk of training-serving skew).
- **Why:** one implementation cannot diverge from itself.

### D7. Serving: model loaded inside FastAPI

- **Decision:** FastAPI loads the MLflow pyfunc in-process from the registry `champion` alias and reloads when the alias moves.
- **Alternatives:** a separate `mlflow models serve` container behind FastAPI.
- **Why:** owner's choice. One fewer service and no restart logic.
- **Consequence:** the rubric asks whether MLflow was used for APIfication, and this design does not clearly meet that line. Because the model is logged as a pyfunc, `mlflow models serve` remains possible without changing the model.

### D8. Frontend: Streamlit

- **Decision:** Streamlit in its own container, calling FastAPI through a configurable REST URL. The pipeline console is a second page.
- **Alternatives:** plain HTML/JS with Leaflet; React.
- **Why:** already familiar from Assignment 4, and the fastest route given the deadline.

### D9. Drift, retraining and promotion

- **Decision:** retraining fires on sustained drift (PSI on key features and predictions) or sustained error (rolling 7-day WAPE), with a cooldown, through a Prometheus alert and Alertmanager webhook.
- **Promotion gate:** on recent data neither model trained on, the challenger must beat the champion overall, must not get materially worse on any horizon or station tier, and must beat both baselines. Rollback moves the `champion` alias back.
- **Alternatives:** fixed schedule; drift only (seasonal change fires it when forecasts are fine); error only (reacts late).
- **Resolved in Milestone 5:** threshold values were calibrated from normal variation and approved on 2026-11-14 (Milestone 5, T4 and T5).

### D10. Runtime and storage

- **Decision:** Python 3.12 venv pinned by `requirements.lock`, with the same lockfile in containers. Spark runs in `local[*]` mode. The DVC remote is a local folder outside the repository, `../dvc-storage`.
- **Alternatives:** a pipeline-only container (slower iteration); the Assignment 3 Spark cluster (extra JVMs on one machine with no gain).

### D11. CI/CD extras deferred

- **Decision:** no GitHub Actions, `make ci` or Git hooks for now. The DVC pipeline remains, because data versioning depends on it and the rubric marks the DVC DAG.

## Implementation notes

Findings from Milestone 0 testing.

- **Spark workers:** Spark starts workers with `python3` from PATH, which is 3.14 here, so `get_spark` points them at the project interpreter.
- **Timestamps:** Spark runs in UTC so Divvy times stay wall-clock labels. A Chicago session zone would hit hours that do not exist on the spring daylight-saving day. Divvy times are confirmed as Chicago local: 2026-03-08 has no trips in the 02:00 hour.
- **Arrow:** enabled because the non-Arrow `toPandas()` path crashes on timestamps on this stack.
- **pandas:** pinned below 3 because PySpark 3.5 predates it. This pin was not the cause of the timestamp failure.

Findings from Milestone 1.

- **Station identity:** IDs are reused across sites and one site has several labels, so neither ID nor name works alone. Names are normalised, then merged only if they share an ID and sit within 100 m. This merged `Streeter Dr & Grand Ave` into `Navy Pier` and kept moved stations separate. 2,188 raw labels became 2,037 stations.
- **Timezone leak:** `F.lit()` on a Python datetime applies the machine's time zone, which let 6,840 July rows through on an IST machine. Date filtering now happens in pandas on plain labels.
- **Memory:** the first design ran out of memory on the 14 GB machine, and the kernel killed Spark and the editor twice. The features stage now filters inside the pandas function, so rows that would be dropped never return to the JVM, and Spark is capped at 6 cores and a 4 GB driver. Measured peak after the fix: 6.8 GB for the pipeline, with 1.3 GB of the machine still available.
- **Alert frequency:** net flow of at least half of capacity occurs in 0.78% of classic station-hours over 3 hours and 2.08% over 6 hours (16 months, all classic stations). With a 3-hour horizon, a ranked list is likely more useful than a fixed alert threshold. To be decided in Milestone 4.

Findings from Milestone 2.

- **Defaults chosen without a separate decision:**
  - Station ID is not a model input. The global model must generalise to new and renamed stations, so it sees location, capacity and recent volume instead.
  - MLflow runs as a local tracking server (SQLite with an artifact folder) so that M3 can use the registry.
  - Tuning uses random search over 12 configurations per target, on a 1.5M-row sample of the training data, with early stopping on January.
- **Result:** LightGBM beats both baselines overall, at every horizon and in peak hours (February test set, `reports/metrics.json`). The margin over the historical average is narrow: 3% on pickups WAPE, 1% on drop-offs and 3% on net flow MAE. The gain comes from the busiest stations (20+ trips/day) and peak hours, where rebalancing matters most.
- **Weak spot:** for 5 to 10 trips/day stations the model only ties the historical average, and on drop-offs it is 1.3% worse (WAPE 0.997 against 0.984). This affects the D9 promotion gate, which says a challenger must not be materially worse on any tier. Owner decision (2026-11-04): keep the model for all eligible stations, disclose the weakness, and set the gate's tolerance during M5 threshold calibration. Routing this tier to the historical average was considered and not chosen.
- **Sparse validation month:** January averages 0.42 trips per station-hour. Forecasting zero everywhere scores a WAPE of exactly 1.0. The model scores 0.993, and both baselines score worse than zero (1.22 and 1.29). WAPE rewards forecasts near the median, which is zero, while a Poisson model forecasts the mean. On sparse data, WAPE therefore cannot show a mean forecast is useful, so a zero forecast is a necessary reference point.
- **Tuning sensitivity:** all 24 trials fall within 0.5% of each other on validation WAPE. Ranking by Poisson deviance picks the same configuration for both targets. Hyperparameters are not the limiting factor.
- **Tuning metric, owner decision (2026-11-04):** trials are now ranked by validation Poisson deviance, not WAPE. Deviance is the training loss and a proper score for a mean forecast, and it still separates trials on sparse data. Both metrics chose the same configuration, so the model is unchanged.
- **Poisson deviance for baselines:** both baselines sometimes forecast exactly zero for an hour that had trips. The deviance then depends on the numerical floor of 1e-9, so baseline deviance (about 11) cannot be compared with the model's (0.94). Comparisons use WAPE, MAE and net flow MAE.
- **Memory:** tuning and training peaked at 4.9 GB inside the 9 GB cap, with no out-of-memory kills.

- **Reproducibility:** a full rerun of `tune` and `train` matched the first run exactly: identical test metrics, the same chosen configurations and stopping rounds, and a byte-identical model file.
- **Lineage flag:** `dvc repro` deletes stage outputs and rewrites `dvc.lock` while it runs, so every run was first tagged as uncommitted code. The dirty check now ignores `dvc.lock` and `reports/`, which the pipeline writes, and still flags changes to code or configuration.

## Milestones 3 and 4: defaults for owner review

The owner asked for M3 and M4 to be completed without stopping for each choice (2026-11-13). These were taken as defaults and can be changed.

- **S1. Replay instead of a live feed.** The service replays the hourly panel. A forecast time (origin) is chosen in the frontend, limited to the replay months, March to September 2026 (D4). Earlier origins are rejected, because forecasting them would reuse training, validation or test data.
  - *Alternative:* a live GBFS feed. It has no trip history to build features from (D1).
- **S2. Online features by day.** For each station, the shared feature function (D6) runs once over a 1,400-hour window ending at the end of the requested day, across 4 worker processes. The day's results are cached, holding up to 8 days. The window covers the longest lookback, the 8-week same-hour mean (1,343 hours).
  - *Measured:* a cold day takes 9 to 16 s natively and up to 34 s in Docker; a cached hour takes 0.2 s. On three replay days, including the 4 July holiday, all 29 model inputs equal the Spark values bit for bit.
  - *Alternatives:* precomputed features (rejected in D6); a vectorised rewrite of the feature function (a second implementation that could drift from the first).
- **S3. Registry and reload.** The model is registered as `bikeshare-demand`. `python -m bikeshare.modeling.register` sets the `champion` alias only when no champion exists, and promotion (M6) moves it after that. The backend checks the alias every 30 seconds and loads the new model by version number.
- **S4. Alert list.** Each direction lists up to 10 stations, ranked by 3-hour net flow as a share of docks. A station is listed only if that share is at least 10%. Only model-tier stations with a known capacity can be listed. This settles the M1 note on ranked lists against a fixed threshold. The settings are `serving.alert_limit` and `serving.min_pressure` in `params.yaml`.
- **S5. Fallback forecast (D2).** Low-confidence stations get the 8-week same-hour mean. If that history is missing, they get the 4-week mean, then the 1-week hourly mean, then zero.
- **S6. Container dependencies.** This deviates from D10's "same lockfile in containers":
  - The backend image installs only the serving packages, with versions constrained by `requirements.lock`. Shared packages have identical versions, and Spark, Ray and DVC are left out. The image is 1.3 GB, and the disk had 6 GB free.
  - The frontend has its own lockfile (`frontend/requirements.lock`), because it runs in a separate container and never imports backend code.
- **S7. MLflow in Compose.** The MLflow server runs as a Compose service on the existing `mlflow.db` and `mlartifacts/`, using the backend image. All ports bind to localhost.
- **S8. Frontend pages.** There are three pages, sharing one forecast-time picker in the sidebar:
  - Rebalancing alerts: two ranked tables and a map.
  - Station forecast: the last 24 hours, the forecast, and what happened next.
  - Pipeline console: model lineage, test results, data-quality counts and a registry check.
- **S9. Metrics endpoint early.** The backend already exposes request counts, latency, forecasts by source and the model version at `/metrics`, ready for Prometheus in M5.

Findings from Milestones 3 and 4.

- **Bundled model code shadowed the service code:** `mlflow.pyfunc.load_model` puts the model's bundled copy of `src/bikeshare` first on `sys.path`. Already-imported modules were unaffected, but feature workers started afterwards imported the training-time copy, which has no `serving` package, and crashed. The model store now restores `sys.path` after each load.
- **Input types:** the logged signature requires float64 for every input, as in training, so the service casts before predicting. Spark stores most features as float32, and the service casts its own values to float32 first, so parity stays exact.
- **MLflow host check:** the MLflow server rejects Host headers it does not know, as protection against DNS rebinding. The Compose service allows its own service name explicitly.
- **D7 risk still open:** the model is served in-process, not through `mlflow models serve`. Because it is a logged pyfunc, `mlflow models serve -m models:/bikeshare-demand@champion` would work without changes if the rubric needs it.

## Milestone 5: monitoring

D9 left the thresholds open for approval in this milestone. The owner approved the calibrated thresholds (T4) and the quiet-tier tolerance (T5) on 2026-11-14. T1–T3, T6 and T7 are implementation defaults. Whether drift alone should trigger retraining is still open (below).

- **T1. Replay monitor inside the backend.** A simulated clock moves one origin hour at a time through March to September 2026. At each hour the backend forecasts every active station through the same code path as `/alerts`. At the last hour of each day it scores the forecasts whose target hours have now happened, and updates a 7-day rolling window. It is started and stopped from the frontend's monitoring page or `POST /monitoring/replay/start`.
  - Errors are counted on the day of the target hour. Drift is counted on the day the inputs were seen. Only model-routed stations are monitored.
  - When the champion changes, the windows restart, so the new model is judged only on its own forecasts. A new model cannot raise an alert until a full window has passed, which also acts as the D9 cooldown.
  - *Alternatives:* a separate replay container that calls the API over HTTP (more traffic realism, but the forecasts would be scored outside the service); backfilling Prometheus with historical timestamps (not live monitoring).
- **T2. Error signal: model WAPE divided by the fallback's WAPE.** The fallback is the historical average the service would otherwise serve, scored on the same rows. A value above 1 means the model is no longer worth serving.
  - *Why not raw WAPE:* it falls as demand rises, from 0.85 in March to about 0.6 in summer, with the same model. A fixed WAPE threshold would track ridership, not model quality.
- **T3. Drift signal: PSI against the training window.** Seven columns are monitored: the 8-week same-hour means, 24-hour means, 28-day volume, and both forecasts. They are the strongest demand inputs by SHAP, plus the model's own output. The reference is a 500,000-row sample of the training window, binned at deciles. It is logged as `monitoring/reference_profile.json` in the model's MLflow run, so the service always compares against the profile of the model it is serving. M6 will log it at training time. Version 1's profile was added to its existing run.
- **T4. Thresholds.** These are calibrated with `dvc repro calibrate` (`reports/monitoring_calibration.json` and `.png`). The calibration reduces every day from August 2025 to September 2026 exactly as the live monitor does: a test checks that a replayed day equals the offline record.

  | Signal | Training window (Aug–Dec) | Jan–Feb (model judged fit) | Replay (Mar–Sep) |
  |---|---|---|---|
  | Largest 7-day PSI | median 0.20, max 2.49 | median 1.99, max 2.56 | median 0.20, max 1.04 |
  | Worse target's WAPE / fallback WAPE | in-sample, max 0.96 | median 0.95, max 1.03 | median 0.96, max 1.04 |

  - **Error: above 1.0 on 5 consecutive simulated days.** In the continuous offline series it fires once, on 5 March, and clears after 17 March, during a real weak spot where the model is up to 4% worse than the fallback on drop-offs. It raises no alert in January or February. A 3-day sustain raised two short false alerts in February.
  - **Drift: largest PSI above 0.5 on 5 consecutive days.** It fires on 5 March and clears after 8 April. Below 0.5, weather-driven weeks in summer start to fire; above 1.0, nothing in the replay does.
  - Thresholds live in `params.yaml` and reach Prometheus as `bikeshare_monitor_threshold`, so the rules have no copies of them. Alerts count consecutive breaching simulated days, not wall-clock time, so a fast or slow replay raises the same alerts.
- **T5. Quiet-tier gate tolerance (M2 owner decision).** A challenger may be at most 3% worse than the champion on any horizon or volume tier (`promotion.tier_tolerance`). For context, one model's 5–10 trips/day WAPE ratio moves a median 2.4% (pickups) to 2.8% (drop-offs) from one week to the next. A tighter tolerance would fail challengers on noise.
- **T6. Prometheus and Grafana.**
  - Images: `prom/prometheus:v3.5.0` and `grafana/grafana:12.1.0`, both pinned.
  - Prometheus scrapes every 5 s and keeps 7 days of data.
  - Grafana's datasource and dashboard are provisioned from files in Git, and editing them in the UI is disabled. Anonymous users get read-only access. The admin password comes from `.env`, which Git ignores; `.env.example` holds a placeholder.
- **T7. Alertmanager deferred to M6.** The two model alerts carry `action=retrain`. Alertmanager and the webhook that starts retraining belong with the retraining job.

**Owner decision needed: should drift trigger retraining on its own?** D9 retrains on sustained drift or sustained error. The calibration does not support drift as a trigger on this data:

- **The training window already covers the demand range.** Training was August to December, so it includes both late-summer and early-winter demand. Within it, a single week already reaches PSI 2.5 against the window's own profile.
- **The expected warm-season drift does not appear.** The replay never goes above 1.04, and summer 2026 is close to the reference (PSI near 0.02 on the 8-week means). D4's expected "warm-season surge" drift therefore does not occur.
- **High drift did not mean a worse model.** In January and February, PSI was around 2 while the model beat its fallback by 5% to 30%. A drift trigger would have retrained a model that was working.

Options:

- (a) Keep D9 as decided. In the replay both alerts fire on 5 March, so M6 behaves the same either way.
- (b) Make drift advisory (`action=investigate`) and retrain only on error. This is the recommendation, because it matches the evidence and the drift-only risk D9 already listed.

Findings from Milestone 5.

- **DVC staleness from bytecode:** `dvc.lock` hashed `src/bikeshare/modeling` including `__pycache__`, so any test run marked `train` as changed, and `dvc repro` would have retrained for nothing. `.dvcignore` now excludes bytecode. The lock was re-recorded with `dvc commit train`; only that directory hash changed (16 files to 9 source files).
- **Feature cache lock:** the feature store held one lock while computing a day (15 to 35 s), so a replay computing a new day blocked dispatcher requests for days already cached. The lock now covers only the cache, and requests for a day already being computed share that one computation.
- **End-to-end replay (Docker, 1 to 14 March):** both alerts fired in Prometheus on simulated 11 March, after 7 days to fill the window and 5 breaching days. The offline calibration says 5 March because its windows reach back into February. From 8 March onward, every live window value equals the offline calibration to four decimals.
- **Replay speed:** about 45 s per simulated day in Docker, with 4 feature workers using 4 cores, so March to September takes about 2.5 hours. More workers would be faster. The backend uses 2.3 GB of its 3 GB limit, so `serving.feature_workers` and `mem_limit` would have to rise together.
- **Pandas deprecation warnings:** tests show NumPy "generic timedelta unit" warnings from `Timestamp + Timedelta` arithmetic, including in M1's `features.py`. They predate M5 and do not affect results.
- **Latency buckets:** the default histogram stopped at 10 s, so a cold-day request (15 to 35 s) could not be measured. Buckets now reach 60 s.
