"""REST client for the backend and the trainer. The frontend never imports their code (D8)."""
import os

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")
TRAINER_URL = os.environ.get("TRAINER_URL", "http://localhost:8001").rstrip("/")
# A day the backend has not cached yet takes about 15 seconds to compute.
TIMEOUT_SECONDS = 120


class BackendError(RuntimeError):
    """A service could not be reached or rejected the request."""


def _request(method: str, path: str, base: str = BACKEND_URL, service: str = "backend",
             **params) -> dict | list:
    url = f"{base}{path}"
    try:
        response = requests.request(method, url, timeout=TIMEOUT_SECONDS,
                                    params={k: v for k, v in params.items() if v is not None})
    except requests.RequestException as error:
        raise BackendError(f"Cannot reach the {service} at {base}.") from error
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise BackendError(f"The {service} returned {response.status_code}: {detail}")
    return response.json()


def _trainer(method: str, path: str, **params) -> dict | list:
    return _request(method, path, base=TRAINER_URL, service="retraining service", **params)


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


def monitoring() -> dict:
    """Not cached: the replay moves on every few seconds."""
    return _request("GET", "/monitoring")


def start_replay(first: str, last: str, delay_seconds: float) -> dict:
    return _request("POST", "/monitoring/replay/start", first=first, last=last,
                    delay_seconds=delay_seconds)


def stop_replay() -> dict:
    return _request("POST", "/monitoring/replay/stop")


def retraining() -> dict:
    """The trainer's job in progress and recent results. Not cached: jobs change state."""
    return _trainer("GET", "/status")


def retraining_jobs() -> list[dict]:
    return _trainer("GET", "/jobs")


def retrain_now() -> dict:
    return _trainer("POST", "/retrain")


def rollback() -> dict:
    result = _trainer("POST", "/rollback")
    st.cache_data.clear()
    return result
