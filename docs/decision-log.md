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
- **Open:** threshold values are not set. They will be calibrated from normal variation and approved in Milestone 5.

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
