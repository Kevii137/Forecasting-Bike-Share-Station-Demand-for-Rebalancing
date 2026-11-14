"""Pipeline console: which model is serving, where it came from, retraining and data health."""
import os

import pandas as pd
import streamlit as st

import api

MLFLOW_UI_URL = os.environ.get("MLFLOW_UI_URL", "http://localhost:5000")
GRAFANA_URL = os.environ.get("GRAFANA_URL")

st.title("Pipeline console")

try:
    model = api.model()
    reports = api.pipeline()
except api.BackendError as error:
    st.error(str(error))
    st.stop()

st.subheader("Serving model")
tags = model["tags"]
columns = st.columns(3)
columns[0].metric(f"Version on alias '{model['alias']}'", model["version"])
columns[1].metric("Loaded (UTC)", model["loaded_at"][11:16],
                  help=f"Loaded at {model['loaded_at']}")
columns[2].metric("Trained from committed code",
                  "yes" if tags.get("git_dirty") == "False" else "no")
if model["last_refresh_error"]:
    st.warning(f"Last registry check failed, still serving version {model['version']}: "
               f"{model['last_refresh_error']}")

st.markdown(
    f"- Registered model: `{model['name']}`\n"
    f"- Training run: `{model['run_id']}`\n"
    f"- Git commit: `{tags.get('git_commit', 'unknown')[:12]}`\n"
    f"- Feature data version (DVC): `{tags.get('dvc_features_md5', 'unknown')}`\n"
    f"- Evaluated on: {tags.get('split_test', 'unknown')}"
    + (f"\n- Retrained after `{tags['trigger']}` at simulated {tags.get('as_of')}, "
       f"gate {tags.get('gate')}" if tags.get("trigger") else "")
)
links = [f"[MLflow experiments and registry]({MLFLOW_UI_URL})"]
if GRAFANA_URL:
    links.append(f"[Grafana monitoring]({GRAFANA_URL})")
st.markdown(" · ".join(links))

if st.button("Check the registry now", help="The backend also checks every 30 seconds."):
    try:
        outcome = api.reload_model()
    except api.BackendError as error:
        st.error(str(error))
    else:
        message = "Loaded a new champion" if outcome["reloaded"] else "Champion unchanged"
        st.success(f"{message}: version {outcome['version']}.")



@st.fragment(run_every=10)
def retraining() -> None:
    st.subheader("Retraining")
    st.caption("Alertmanager sends sustained forecast-error alerts to the retraining service. "
               "It pauses the replay, trains a challenger on the data observed so far, and "
               "promotes it only if it passes the gate on the last 14 days, which neither "
               "model trained on.")
    try:
        trainer = api.retraining()
        jobs = api.retraining_jobs()
    except api.BackendError as error:
        st.warning(str(error))
        return
    job = trainer["job"]
    if job:
        st.info(f"Retraining in progress ({job['phase']}): {job['trigger']} at simulated "
                f"{job['as_of']}, started {job['started_at'][11:16]} UTC. "
                "A job takes about 20 minutes; the replay waits for it.")
    latest = trainer["history"][0] if trainer["history"] else None
    if latest and latest.get("error"):
        st.error(f"The last job failed: {latest['error']}")
    elif latest and latest.get("gate") and not latest["gate"]["passed"]:
        st.warning("The last challenger was rejected by the gate:\n"
                   + "\n".join(f"- {reason}" for reason in latest["gate"]["failed"]))
    if jobs:
        table = pd.DataFrame([{
            "Started (UTC)": entry["started_at"][:16].replace("T", " "),
            "Simulated time": entry["as_of"], "Trigger": entry["trigger"],
            "Gate": entry["gate"], "Version": entry["version"],
            "Outcome": entry["outcome"] + (", rolled back" if entry.get("rolled_back_at") else ""),
            "Pickups WAPE, challenger": entry["challenger_pickups_wape"],
            "Pickups WAPE, champion": entry["champion_pickups_wape"],
            "Drop-offs WAPE, challenger": entry["challenger_dropoffs_wape"],
            "Drop-offs WAPE, champion": entry["champion_dropoffs_wape"],
        } for entry in jobs])
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            column: st.column_config.NumberColumn(format="%.4f")
            for column in table.columns if "WAPE" in column})
        st.caption("WAPE on the gate window. Every challenger is registered; only one that "
                   "passes the gate becomes champion.")
    else:
        st.info("No retraining runs yet.")

    left, right = st.columns(2)
    if left.button("Retrain now", disabled=trainer["running"],
                   help="Retrain at the replay's current simulated time, skipping the cooldown."):
        try:
            api.retrain_now()
        except api.BackendError as error:
            st.error(str(error))
        else:
            st.rerun(scope="fragment")
    with right.popover("Roll back the champion"):
        st.write("Point the champion alias back at the version the current champion "
                 "replaced. The backend loads it straight away.")
        if st.button("Confirm rollback"):
            try:
                moved = api.rollback()
            except api.BackendError as error:
                st.error(str(error))
            else:
                st.success(f"Champion moved from version {moved['from']} to {moved['to']}.")


retraining()

st.subheader("Test results, February 2026")
summary = (reports.get("metrics") or {}).get("summary")
if summary:
    names = {"lightgbm": "LightGBM model", "historical_average": "Historical average",
             "seasonal_naive": "Same hour last week"}
    st.dataframe(pd.DataFrame([
        {"Forecast": names.get(key, key), "Pickups WAPE": values["pickups_wape"],
         "Drop-offs WAPE": values["dropoffs_wape"], "Net flow MAE": values["net_flow_mae"]}
        for key, values in summary.items()
    ]), hide_index=True, width="stretch", column_config={
        column: st.column_config.NumberColumn(format="%.3f")
        for column in ("Pickups WAPE", "Drop-offs WAPE", "Net flow MAE")})
    st.caption("WAPE is total absolute error divided by total trips. Lower is better. "
               "These are the first model's results from the DVC pipeline.")
else:
    st.info("No evaluation report found.")

st.subheader("Data pipeline")
quality = reports.get("data_quality") or {}
stations = reports.get("stations") or {}
panel = reports.get("panel") or {}
features = reports.get("features") or {}
rows = [
    ("clean", "Trips kept", quality.get("rows_written"),
     f"{quality.get('rejected_share', 0):.1%} of {quality.get('rows_read', 0):,} rejected"),
    ("stations", "Stations", stations.get("stations"),
     f"{stations.get('trip_share_with_capacity', 0):.1%} of trips have a known capacity"),
    ("panel", "Station-hours", panel.get("rows"),
     f"{panel.get('zero_hour_share', 0):.0%} of hours have no trips"),
    ("features", "Feature rows", features.get("rows"),
     f"{len(features.get('by_month', {}))} months"),
]
st.dataframe(pd.DataFrame(rows, columns=["DVC stage", "Output", "Count", "Check"]),
             hide_index=True, width="stretch",
             column_config={"Count": st.column_config.NumberColumn(format="localized")})
