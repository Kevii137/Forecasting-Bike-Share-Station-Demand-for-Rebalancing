"""Dispatcher view: stations likely to run out of bikes or docks in the next hours."""
import pandas as pd
import pydeck as pdk
import streamlit as st

import api

DRAINING_COLOUR = [230, 97, 1]   # orange
FILLING_COLOUR = [33, 102, 172]  # blue
QUIET_COLOUR = [150, 150, 150]
LIMIT = 10

origin = st.session_state["origin"]
st.title("Rebalancing alerts")

try:
    with st.spinner("Forecasting every station. A new day takes about 15 seconds."):
        result = api.alerts(origin, LIMIT)
        stations = pd.DataFrame(api.forecast(origin)["stations"])
except api.BackendError as error:
    st.error(str(error))
    st.stop()

when = pd.Timestamp(result["origin"])
horizon = max(result["horizons"])
end = when + pd.Timedelta(hours=horizon)
st.markdown(f"Forecast at **{when:%A %d %B %Y, %H:%M}** for the next {horizon} hours, "
            f"until {end:%H:%M}. Model version {result['model_version']}.")

columns = st.columns(4)
columns[0].metric("Active stations", result["stations_forecast"])
columns[1].metric("Forecast by the model", result["stations_on_model"])
columns[2].metric("Likely draining", len(result["draining"]))
columns[3].metric("Likely filling", len(result["filling"]))


def alert_table(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return pd.DataFrame({
        "Station": frame["station_key"],
        "% of docks": (frame["pressure"] * 100).round(),
        "Net change": frame["net_flow"],
        "Bikes out": frame["pickups"],
        "Bikes in": frame["dropoffs"],
        "Docks": frame["capacity"].astype(int),
    })


number = st.column_config.NumberColumn(format="%.1f")
config = {"Bikes out": number, "Bikes in": number, "Net change": number,
          "% of docks": st.column_config.NumberColumn(
              format="%+d%%", help="Net change as a share of the station's docks")}
threshold = round(result["min_pressure"] * 100)

left, right = st.columns(2)
with left:
    st.subheader("Likely draining: send bikes")
    st.caption(f"More bikes leaving than arriving, by at least {threshold}% of docks.")
    table = alert_table(result["draining"])
    if table.empty:
        st.info("No station is forecast to drain this much.")
    else:
        st.dataframe(table, hide_index=True, column_config=config, width="stretch")
with right:
    st.subheader("Likely filling: free up docks")
    st.caption(f"More bikes arriving than leaving, by at least {threshold}% of docks.")
    table = alert_table(result["filling"])
    if table.empty:
        st.info("No station is forecast to fill this much.")
    else:
        st.dataframe(table, hide_index=True, column_config=config, width="stretch")

st.subheader("Map")
draining = {row["station_key"] for row in result["draining"]}
filling = {row["station_key"] for row in result["filling"]}
stations = stations.dropna(subset=["lat", "lng"]).copy()
stations["colour"] = [
    DRAINING_COLOUR if key in draining else FILLING_COLOUR if key in filling else QUIET_COLOUR
    for key in stations["station_key"]
]
stations["radius"] = [220 if key in draining | filling else 40 for key in stations["station_key"]]
stations["net_text"] = stations["net_flow"].map(lambda value: f"{value:+.1f}")
stations["confidence_text"] = stations["confidence"].map(
    {"normal": "model forecast", "low": "low confidence (historical average)"})
alerted = stations["station_key"].isin(draining | filling)
layer = pdk.Layer(
    "ScatterplotLayer",
    # Draw alerted stations last so they sit on top.
    data=pd.concat([stations[~alerted], stations[alerted]]),
    get_position=["lng", "lat"], get_fill_color="colour", get_radius="radius",
    radius_min_pixels=2, pickable=True, opacity=0.85,
)
view = pdk.ViewState(latitude=41.86, longitude=-87.64, zoom=10.8)
tooltip = {"text": "{station_key}\nNet change {net_text} bikes over the next hours\n"
                   "{confidence_text}"}
st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip,
                         map_provider="carto", map_style=pdk.map_styles.CARTO_LIGHT),
                height=520)
st.caption("Orange: likely draining. Blue: likely filling. Grey: no alert.")

with st.expander("How to read these alerts"):
    st.markdown(
        f"- **Net change** is forecast bikes arriving minus bikes leaving over the next "
        f"{horizon} hours. It is not a count of bikes at the station: Divvy publishes no "
        "history of station inventory, so the app cannot know how full a station is.\n"
        "- Stations are ranked by net change as a share of their docks, because a change of "
        "five bikes matters more at an 11-dock station than at a 40-dock one.\n"
        "- Only classic stations averaging at least 5 trips a day use the forecasting "
        "model and can raise alerts. Quieter and lightweight stations get a historical "
        "average, marked low confidence.\n"
        "- Trips record satisfied demand only. An empty station shows few pickups even when "
        "riders wanted bikes, so forecasts can understate demand at stations that often run "
        "empty."
    )
