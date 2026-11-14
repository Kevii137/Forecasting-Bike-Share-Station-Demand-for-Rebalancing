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
PROMETHEUS = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")
GRAFANA = os.environ.get("GRAFANA_URL", "http://localhost:3000")
TRAINER = os.environ.get("TRAINER_URL", "http://localhost:8001")
ALERTMANAGER = os.environ.get("ALERTMANAGER_URL", "http://localhost:9093")
ORIGIN = os.environ.get("SMOKE_ORIGIN", "2026-07-15T17:00")
failures = []


def get(url: str, timeout: float = 120) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


def post(url: str, payload: dict, timeout: float = 30) -> tuple[int, str]:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


def compose_exec(service: str, *command: str) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", "compose", "exec", "-T", service, *command],
                          check=False, capture_output=True, text=True)


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

status, body = get(f"{API}/monitoring")
monitoring = json.loads(body) if status == 200 else {}
check("replay monitor reports its thresholds", "max_psi" in monitoring.get("thresholds", {}),
      f"running={monitoring.get('running')}, clock={monitoring.get('clock')}")

status, body = get(f"{PROMETHEUS}/api/v1/targets")
targets = json.loads(body)["data"]["activeTargets"] if status == 200 else []
health = {t["labels"]["job"]: t["health"] for t in targets}
check("Prometheus scrapes the backend, trainer and Alertmanager",
      all(health.get(job) == "up" for job in ("backend", "trainer", "alertmanager")), str(health))

status, body = get(f"{PROMETHEUS}/api/v1/rules")
rules = ([rule["name"] for group in json.loads(body)["data"]["groups"] for rule in group["rules"]]
         if status == 200 else [])
check("Prometheus has loaded the drift and error alert rules",
      {"BikeshareInputDrift", "BikeshareForecastDegraded"} <= set(rules), f"{len(rules)} rules")

status, body = get(f"{PROMETHEUS}/api/v1/alertmanagers")
active = json.loads(body)["data"]["activeAlertmanagers"] if status == 200 else []
check("Prometheus sends alerts to Alertmanager", len(active) == 1, str(active))

status, _ = get(f"{ALERTMANAGER}/-/ready")
check("Alertmanager is ready", status == 200)

status, body = get(f"{TRAINER}/status")
trainer = json.loads(body) if status == 200 else {}
check("trainer reports its status and the champion", status == 200,
      f"running={trainer.get('running')}, champion={trainer.get('champion_version')}")

# A drift alert is advisory: the trainer must acknowledge it without starting a job.
status, body = post(f"{TRAINER}/alertmanager", {"alerts": [{
    "status": "firing", "labels": {"alertname": "BikeshareInputDrift", "action": "investigate"}}]})
check("trainer ignores an advisory drift alert",
      status == 200 and json.loads(body).get("decision") == "ignored", body[:120])

lineage = compose_exec("trainer", "git", "-C", "/repo", "rev-parse", "--short", "HEAD")
check("trainer can read the git commit for lineage tags", lineage.returncode == 0,
      lineage.stdout.strip() or lineage.stderr.strip()[-120:])

status, body = get(f"{GRAFANA}/api/dashboards/uid/bikeshare")
check("Grafana serves the provisioned dashboard", status == 200,
      json.loads(body)["dashboard"]["title"] if status == 200 else body[:80])

reach = compose_exec("frontend", "python", "-c",
                     "import urllib.request; [urllib.request.urlopen(f'http://{host}/health') "
                     "for host in ('backend:8000', 'trainer:8001')]; print('ok')")
check("frontend container reaches the backend and trainer over the compose network",
      reach.returncode == 0, reach.stdout.strip() or reach.stderr.strip()[-120:])

print(f"\n{len(failures)} failed" if failures else "\nAll checks passed")
sys.exit(1 if failures else 0)
