# AI Disclosure

## Tool used

- **Claude Code (Anthropic), powered by Claude Opus 5.5.**

## How I used it

- I used Claude Code to review the project materials and help profile the Divvy dataset. I used these findings as input when considering the project design.
- I used it to explain design trade-offs and recommend possible options. I considered the alternatives and made the final decisions myself.
- I used it to generate boilerplate code and assist with environment setup, as well as to diagnose and debug environment-related failures.
- I used it for implementation assistance with the Spark data pipeline and the modelling workflow, including tuning, training, evaluation, and MLflow logging. I also used it to help interpret the results. I was responsible for reviewing the implementation and ensuring that the final workflow matched the decisions documented for the project.
- I used it to assist with the serving API, Streamlit frontend, and Docker Compose setup. For some milestones, I asked it to proceed using reasonable defaults rather than pausing for every individual choice. Those defaults were recorded for my review and are described in `docs/hld.md`.
- I used it to assist with the monitoring components, including the replay monitor, drift and error signals, Prometheus rules, and Grafana dashboard, and with threshold calibration. I made the decision to keep drift advisory and to use forecast error as the trigger for retraining; this is documented in `docs/hld.md`.
- I used it to assist with the retraining service, Alertmanager routing, promotion gate, and rollback implementation, based on the recommendations I accepted. I also used it while running and investigating the end-to-end retraining test. The relevant decisions are documented under Milestone 6.
- I used it to draft the architecture, HLD, LLD, test plan, and user manual in `docs/`, using the project's code, decision log, and test results as source material. These documents are drafts for my review, and I am responsible for checking that they accurately describe the implemented system.

## Impact on the project

Claude Code helped me explore alternatives, speed up routine development work, troubleshoot environment issues, and prepare project documentation. It also provided implementation suggestions and generated parts of the code, which I treated as assistance rather than as a substitute for understanding the system.