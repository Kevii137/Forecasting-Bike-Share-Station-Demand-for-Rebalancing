"""Model monitoring: replay months as simulated traffic and watch drift and error."""
import os
from datetime import datetime, time

import altair as alt
import pandas as pd
import streamlit as st

import api

GRAFANA_URL = os.environ.get("GRAFANA_URL")
REFRESH_SECONDS = 5

st.title("Model monitoring")
st.markdown(
    "The replay advances a simulated clock one hour at a time through March to September "
    "2026. At each hour the backend forecasts every station, as it does for dispatchers. "
    "At the end of each simulated day it scores forecasts against the trips that then "
    "happened and compares the week's inputs with the training data."
)

try:
    window = api.replay()
    state = api.monitoring()
except api.BackendError as error:
    st.error(str(error))
    st.stop()

first = pd.Timestamp(window["first_origin"])
last = pd.Timestamp(window["last_origin"])

with st.expander("Replay controls", expanded=not state["running"] and not state["history"]):
    with st.form("replay"):
        left, middle, right = st.columns(3)
        start_day = left.date_input("Start", value=first.date(), min_value=first.date(),
                                    max_value=last.date())
        end_day = middle.date_input("End", value=last.date(), min_value=first.date(),
                                    max_value=last.date())
        pace = right.selectbox("Pace", [0.0, 0.5, 2.0], format_func=lambda value: {
            0.0: "As fast as possible", 0.5: "Half a second per hour",
            2.0: "Two seconds per hour"}[value])
        st.caption("As fast as possible takes about 45 seconds per simulated day, "
                   "mostly to compute a new day's features. Starting again clears the windows.")
        if st.form_submit_button("Start replay", disabled=state["running"]):
            begin = pd.Timestamp(datetime.combine(start_day, time(0)))
            end = min(pd.Timestamp(datetime.combine(end_day, time(23))), last)
            try:
                api.start_replay(f"{begin:%Y-%m-%dT%H:%M}", f"{end:%Y-%m-%dT%H:%M}", pace)
            except api.BackendError as error:
                st.error(str(error))
            else:
                st.rerun()
    if state["running"] and st.button("Stop replay"):
        api.stop_replay()
        st.rerun()


def threshold_rule(value: float, label: str) -> alt.Chart:
    return alt.Chart(pd.DataFrame({"y": [value], "label": [label]})).mark_rule(
        color="#b2182b", strokeDash=[6, 4]).encode(y="y:Q", tooltip=["label:N", "y:Q"])


@st.fragment(run_every=REFRESH_SECONDS)
def live() -> None:
    try:
        state = api.monitoring()
    except api.BackendError as error:
        st.error(str(error))
        return
    thresholds = state["thresholds"]
    sustain = thresholds["sustain_days"]

    columns = st.columns([1, 1.4, 1, 1, 1])  # the simulated time needs the room
    phase = ("paused" if state.get("paused") else "running") if state["running"] else "stopped"
    columns[0].metric("Replay", phase, help="Retraining pauses the replay until the new model "
                      "is ready, so simulated time does not run ahead of it.")
    clock = pd.Timestamp(state["clock"]) if state["clock"] else None
    columns[1].metric("Simulated time", f"{clock:%d %b %H:%M}" if clock else "not started",
                      help=f"{clock:%A %d %B %Y, %H:%M} Chicago time" if clock else None)
    columns[2].metric("Days in window", f"{state['window_days']} of {state['window_size']}")
    columns[3].metric("Drift breach days", f"{state['breach_days']['drift']} of {sustain}",
                      help=f"Consecutive days with the largest PSI above {thresholds['max_psi']}")
    columns[4].metric("Error breach days", f"{state['breach_days']['error']} of {sustain}",
                      help="Consecutive days with model WAPE / historical-average WAPE above "
                           f"{thresholds['relative_wape']}")
    if state["error"]:
        st.error(f"The replay stopped: {state['error']}")
    if state.get("paused"):
        st.info("The replay is paused while a challenger is retrained (Pipeline console).")
    if not state["profile_loaded"] and state["model_version"]:
        st.warning("The champion has no reference profile, so drift is not measured.")
    messages = {"drift": ("Inputs have drifted from the training data",
                          "Prometheus raises an alert to investigate; drift alone does not retrain."),
                "error": ("The model is no better than the historical average",
                          "Prometheus raises a retraining alert.")}
    for signal, firing in state["alerting"].items():
        if firing:
            problem, consequence = messages[signal]
            st.warning(f"**Alert:** {problem} for {sustain} or more simulated days. "
                       f"{consequence}")

    history = pd.DataFrame(state["history"])
    if history.empty or not history["window_full"].any():
        st.info(f"Signals appear once {state['window_size']} simulated days have been scored.")
        return
    history = history[history["window_full"]].assign(day=lambda frame: pd.to_datetime(frame["day"]))

    left, right = st.columns(2)
    with left:
        st.subheader("Model error relative to the fallback")
        error = history.melt("day", ["pickups_relative_wape", "dropoffs_relative_wape"],
                             var_name="target", value_name="ratio")
        error["target"] = error["target"].str.removesuffix("_relative_wape")
        chart = alt.Chart(error).mark_line().encode(
            x=alt.X("day:T", title=None), y=alt.Y("ratio:Q", title="Model WAPE / baseline WAPE",
                                                  scale=alt.Scale(zero=False)),
            color=alt.Color("target:N", title=None), tooltip=["day:T", "target:N", "ratio:Q"])
        st.altair_chart(chart + threshold_rule(thresholds["relative_wape"], "threshold"),
                        width="stretch")
        st.caption("Below 1, the model beats the historical average it would fall back to.")
    with right:
        st.subheader("Drift against the training window")
        drift = history.dropna(subset=["max_psi"])
        chart = alt.Chart(drift).mark_line(color="#5e3c99").encode(
            x=alt.X("day:T", title=None),
            y=alt.Y("max_psi:Q", title="Largest PSI, last 7 days", scale=alt.Scale(type="log")),
            tooltip=["day:T", "max_psi:Q"])
        st.altair_chart(chart + threshold_rule(thresholds["max_psi"], "threshold"),
                        width="stretch")
        st.caption("Population stability index: 0 means the same distribution as in training.")

    latest = state["latest"]
    if latest:
        st.subheader(f"Latest window: {latest['first_day']} to {latest['last_day']}")
        errors = pd.DataFrame(latest["errors"])
        errors["Slice"] = errors.apply(
            lambda row: "All stations" if row["group"] == "all"
            else f"{row['label']} hour{'s' * (row['label'] != '1')} ahead"
            if row["group"] == "horizon" else f"{row['label']} trips/day", axis=1)
        rank = errors["group"].map({"all": 0, "horizon": 1, "tier": 2}) * 100 + errors[
            "label"].map(lambda label: {"all": 0, "5-10": 1, "10-20": 2, "20+": 3}.get(
                label, int(label) if label.isdigit() else 9))
        order = list(dict.fromkeys(errors.assign(rank=rank).sort_values("rank")["Slice"]))
        table = errors.pivot_table(index="Slice", columns="target", values="relative_wape")
        table = table.reindex(order)
        left, right = st.columns(2)
        left.dataframe(table.rename(columns=str.capitalize), width="stretch",
                       column_config={column: st.column_config.NumberColumn(format="%.3f")
                                      for column in ("Pickups", "Dropoffs")})
        left.caption("Model WAPE / historical-average WAPE by slice.")
        psi = pd.Series(latest["psi"], name="PSI").rename_axis("Column").reset_index()
        right.dataframe(psi, hide_index=True, width="stretch",
                        column_config={"PSI": st.column_config.NumberColumn(format="%.3f")})


live()
if GRAFANA_URL:
    st.markdown(f"[Open the Grafana dashboard]({GRAFANA_URL}) for the same signals with "
                "service latency, traffic and firing alerts.")
