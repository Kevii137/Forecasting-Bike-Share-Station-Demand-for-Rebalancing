# AI disclosure

Required by the DA3408 Code of Conduct on fair and responsible use of AI. This file is updated as the project progresses.

## Tools used

Claude Code (Anthropic), running Claude Opus 5.5.

## How it was used

- Auditing the project materials and profiling Divvy data to inform design decisions.
- Explaining design trade-offs and recommending options. Each decision and its alternatives are recorded in `docs/decision-log.md`, and the project owner made the final choice.
- Generating boilerplate and environment setup, and debugging environment failures.
- Writing the Spark data pipeline and the modelling code (tuning, training, evaluation, MLflow logging) to the approach the owner agreed, and analysing the results.
- Writing the serving API, Streamlit frontend and Docker Compose deployment. The owner asked for these milestones to be completed without stopping for each choice, so the defaults taken are listed for review in `docs/decision-log.md` (S1 to S9).
- Writing the monitoring code (replay monitor, drift and error signals, Prometheus rules, Grafana dashboard) and running the threshold calibration. The owner approved the thresholds; the recommendation on drift-triggered retraining awaits the owner's decision (decision log, Milestone 5).

## Impact

The AI tool shaped the options considered and accelerated setup. The project owner chose the design, and in some cases chose differently from the AI recommendation (D3 and D7 in the decision log).
