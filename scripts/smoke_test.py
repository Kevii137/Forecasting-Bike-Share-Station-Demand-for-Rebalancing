"""End-to-end check of the running Docker Compose stack, using only the standard library.

    docker compose up -d --build
    python3 scripts/smoke_test.py
"""
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

API = os.environ.get("API_URL", "http://localhost:8000")
UI = os.environ.get("UI_URL", "http://localhost:8501")
MLFLOW = os.environ.get("MLFLOW_URL", "http://localhost:5000")
ORIGIN = os.environ.get("SMOKE_ORIGIN", "2026-07-15T17:00")
failures = []


def get(url: str, timeout: float = 120) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


def check(name: str, passed: bool, detail: str = "") -> None:
    print(f"{'PASS' if passed else 'FAIL'}  {name}{f'  ({detail})' if detail else ''}")
    if not passed:
        failures.append(name)


status, _ = get(f"{MLFLOW}/health")
check("MLflow server is healthy", status == 200)

status, body = get(f"{API}/ready")
check("backend has loaded the champion model", status == 200, body)

status, body = get(f"{API}/alerts?origin={ORIGIN}")
alerts = json.loads(body) if status == 200 else {}
check("alerts are served", status == 200 and "draining" in alerts,
      f"{len(alerts.get('draining', []))} draining, {len(alerts.get('filling', []))} filling")

ranked = alerts.get("draining", []) + alerts.get("filling", [])
if ranked:
    key = urllib.parse.quote(ranked[0]["station_key"])
    status, body = get(f"{API}/station?key={key}&origin={ORIGIN}")
    forecast = json.loads(body).get("forecast", []) if status == 200 else []
    check("station detail has one forecast per horizon", len(forecast) == 3)

status, _ = get(f"{API}/alerts?origin=2020-01-01T00:00")
check("an origin outside the replay window is rejected", status == 422)

status, body = get(f"{API}/metrics")
check("Prometheus metrics are exposed", "bikeshare_requests_total" in body)

status, _ = get(f"{UI}/_stcore/health")
check("frontend is healthy", status == 200)

reach = subprocess.run(
    ["docker", "compose", "exec", "-T", "frontend", "python", "-c",
     "import urllib.request; print(urllib.request.urlopen('http://backend:8000/health').read())"],
    check=False, capture_output=True, text=True,
)
check("frontend container reaches the backend over the compose network",
      reach.returncode == 0, reach.stdout.strip() or reach.stderr.strip()[-120:])

print(f"\n{len(failures)} failed" if failures else "\nAll checks passed")
sys.exit(1 if failures else 0)
