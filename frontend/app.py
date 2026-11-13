"""Streamlit frontend: rebalancing alerts for dispatchers, plus a pipeline console."""
from datetime import datetime, time

import pandas as pd
import streamlit as st

import api

st.set_page_config(page_title="Bike-share rebalancing", page_icon=":material/pedal_bike:",
                   layout="wide")

try:
    window = api.replay()
except api.BackendError as error:
    st.error(f"{error} Check that the backend container is running.")
    st.stop()

first = pd.Timestamp(window["first_origin"])
last = pd.Timestamp(window["last_origin"])
default = pd.Timestamp(window["default_origin"])

with st.sidebar:
    st.subheader("Forecast time")
    day = st.date_input("Date", value=default.date(), min_value=first.date(),
                        max_value=last.date(), key="day")
    hour = st.selectbox("Last complete hour", range(24), index=default.hour, key="hour",
                        format_func=lambda value: f"{value:02d}:00")
    chosen = pd.Timestamp(datetime.combine(day, time(hour)))
    origin = min(max(chosen, first), last)
    if origin != chosen:
        st.caption(f"Data ends at {last:%d %b %H:%M}, so the forecast time is {origin:%H:%M}.")
    st.caption("Replaying Divvy trips from March to September 2026. "
               "Times are Chicago local time.")

st.session_state["origin"] = origin.strftime("%Y-%m-%dT%H:%M")

navigation = st.navigation([
    st.Page("views/alerts.py", title="Rebalancing alerts", icon=":material/warning:",
            default=True),
    st.Page("views/station.py", title="Station forecast", icon=":material/pedal_bike:"),
    st.Page("views/pipeline.py", title="Pipeline console", icon=":material/monitoring:"),
])
navigation.run()
