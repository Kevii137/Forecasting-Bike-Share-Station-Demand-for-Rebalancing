"""One station: recent demand, the forecast, and what happened next in the replay."""
import altair as alt
import pandas as pd
import streamlit as st

import api

HISTORY_HOURS = 24
SERIES = {"pickups": "Bikes out (pickups)", "dropoffs": "Bikes in (drop-offs)"}

origin = st.session_state["origin"]
st.title("Station forecast")

try:
    with st.spinner("Loading stations..."):
        stations = pd.DataFrame(api.forecast(origin)["stations"])
        alerted = api.alerts(origin, 10)
except api.BackendError as error:
    st.error(str(error))
    st.stop()

# Offer the most pressing stations first, then the rest alphabetically.
ranked = [row["station_key"] for row in alerted["draining"] + alerted["filling"]]
others = sorted(set(stations["station_key"]) - set(ranked))
key = st.selectbox("Station", ranked + others, key="station",
                   help="Stations with alerts are listed first.")

try:
    detail = api.station(key, origin, HISTORY_HOURS)
except api.BackendError as error:
    st.error(str(error))
    st.stop()

info = detail["station"]
columns = st.columns(4)
columns[0].metric("Forecast net change", f"{info['net_flow']:+.1f} bikes",
                  help="Bikes in minus bikes out over the forecast hours.")
columns[1].metric("Docks", "unknown" if info["capacity"] is None else int(info["capacity"]))
columns[2].metric("Trips per day, last 28 days", f"{info['trips_per_day_28d']:.1f}")
columns[3].metric("Forecast source",
                  "Model" if info["confidence"] == "normal" else "Historical average")
if info["confidence"] == "low":
    st.warning("Low confidence: this station is too quiet or not a classic dock, so it gets "
               "a historical-average forecast and raises no alerts.")


def long_form(rows: list[dict], kind: str, time_column: str, suffix: str = "") -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["hour", "series", "trips", "kind"])
    frame = frame.rename(columns={time_column: "hour"})
    frame = frame.rename(columns={f"{demand}{suffix}": label for demand, label in SERIES.items()})
    melted = frame.melt(id_vars="hour", value_vars=list(SERIES.values()),
                        var_name="series", value_name="trips")
    return melted.assign(kind=kind, hour=pd.to_datetime(melted["hour"]))


observed = long_form(detail["history"], "Observed", "hour")
forecast = long_form(detail["forecast"], "Forecast", "target_hour", "_forecast")
after = long_form(detail["observed_after_origin"], "What happened (replay)", "hour")

# Join the forecast line to the last observed hour so it reads as a continuation.
last_hour = observed[observed["hour"] == observed["hour"].max()].assign(kind="Forecast")
forecast = pd.concat([last_hour, forecast])

colour = alt.Color("series:N", title=None,
                   scale=alt.Scale(domain=list(SERIES.values()), range=["#e66101", "#2166ac"]))
x = alt.X("hour:T", title=None, axis=alt.Axis(format="%H:%M"))
y = alt.Y("trips:Q", title="Trips per hour")
chart = (
    alt.Chart(observed).mark_line(point=True).encode(x=x, y=y, color=colour)
    + alt.Chart(forecast).mark_line(strokeDash=[6, 4], point=True).encode(x=x, y=y, color=colour)
    + alt.Chart(after).mark_point(shape="diamond", size=70, filled=False).encode(
        x=x, y=y, color=colour)
)
now = alt.Chart(pd.DataFrame({"hour": [pd.Timestamp(detail["origin"])]})).mark_rule(
    strokeDash=[2, 2], color="gray").encode(x="hour:T")
st.altair_chart((chart + now).properties(height=360), width="stretch")
st.caption("Solid: observed. Dashed: forecast. Hollow diamonds: what actually happened in "
           "the replayed data. The dotted line marks the forecast time.")

table = pd.DataFrame(detail["forecast"])
observed_next = pd.DataFrame(detail["observed_after_origin"])
st.dataframe(pd.DataFrame({
    "Hour": pd.to_datetime(table["target_hour"]).dt.strftime("%H:%M"),
    "Bikes out (forecast)": table["pickups_forecast"].round(1),
    "Bikes in (forecast)": table["dropoffs_forecast"].round(1),
    "Bikes out (actual)": observed_next["pickups"],
    "Bikes in (actual)": observed_next["dropoffs"],
}), hide_index=True, width="stretch", column_config={
    column: st.column_config.NumberColumn(format="%.1f")
    for column in ("Bikes out (forecast)", "Bikes in (forecast)")})
