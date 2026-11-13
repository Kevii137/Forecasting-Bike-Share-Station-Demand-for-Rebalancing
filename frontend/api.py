"""REST client for the backend. The frontend never imports backend code (Decision D8)."""
import os

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")
# A day the backend has not cached yet takes about 15 seconds to compute.
TIMEOUT_SECONDS = 120


class BackendError(RuntimeError):
    """The backend could not be reached or rejected the request."""


def _request(method: str, path: str, **params) -> dict | list:
    url = f"{BACKEND_URL}{path}"
    try:
        response = requests.request(method, url, timeout=TIMEOUT_SECONDS,
                                    params={k: v for k, v in params.items() if v is not None})
    except requests.RequestException as error:
        raise BackendError(f"Cannot reach the backend at {BACKEND_URL}.") from error
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise BackendError(f"The backend returned {response.status_code}: {detail}")
    return response.json()


@st.cache_data(ttl=300, show_spinner=False)
def replay() -> dict:
    return _request("GET", "/replay")


@st.cache_data(ttl=120, show_spinner=False)
def alerts(origin: str, limit: int) -> dict:
    return _request("GET", "/alerts", origin=origin, limit=limit)


@st.cache_data(ttl=120, show_spinner=False)
def forecast(origin: str) -> dict:
    return _request("GET", "/forecast", origin=origin)


@st.cache_data(ttl=120, show_spinner=False)
def station(key: str, origin: str, history_hours: int) -> dict:
    return _request("GET", "/station", key=key, origin=origin, history_hours=history_hours)


@st.cache_data(ttl=15, show_spinner=False)
def model() -> dict:
    return _request("GET", "/model")


@st.cache_data(ttl=60, show_spinner=False)
def pipeline() -> dict:
    return _request("GET", "/pipeline")


def reload_model() -> dict:
    result = _request("POST", "/model/reload")
    st.cache_data.clear()
    return result
