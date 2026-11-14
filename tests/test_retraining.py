"""Retraining: time windows, the promotion gate, rollback, the trainer service and a full job."""
import json
import math
import threading
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient
from helpers import RULES, ConstantModel, FakeClient
from prometheus_client import REGISTRY

from bikeshare.modeling.dataset import FEATURE_COLUMNS, TARGETS
from bikeshare.modeling.packaging import log_model
from bikeshare.modeling.register import CHAMPION, MODEL_NAME, register
from bikeshare.monitoring.replay import ReplayMonitor, Thresholds
from bikeshare.retraining import job, registry
from bikeshare.retraining.gate import compare
from bikeshare.retraining.service import Backend, Trainer, create_app, recorded_jobs
from bikeshare.retraining.windows import load_rows, plan
from bikeshare.serving.model_store import ModelStore

# Windows ------------------------------------------------------------------------------------


def test_windows_end_at_as_of_and_do_not_overlap():
    windows = plan(pd.Timestamp("2026-03-11T23:00"), "2025-08", gate_days=14,
                   early_stopping_days=7)
    assert windows.gate_start == pd.Timestamp("2026-02-26")
    assert windows.stop_start == pd.Timestamp("2026-02-19")
    assert windows.train_start == pd.Timestamp("2025-08-01")
    assert windows.describe()["gate"] == "2026-02-26 00:00 to 2026-03-11 23:00"
    with pytest.raises(ValueError, match="less than 28 days"):
        plan(pd.Timestamp("2025-09-01"), "2025-08", 14, 7)


def synthetic_features(root, days: int = 40, seed: int = 0):
    """Feature rows for three stations, partitioned by origin month like the Spark output."""
    rng = np.random.default_rng(seed)
    origins = pd.date_range("2026-01-01", periods=days * 24, freq="h")
    frames = []
    for station, rate in (("a", 1.5), ("b", 3.0), ("c", 6.0)):
        for horizon in (1, 2, 3):
            frame = pd.DataFrame({"station_key": station, "origin_hour": origins,
                                  "target_hour": origins + pd.Timedelta(hours=horizon)})
            for column in FEATURE_COLUMNS:
                frame[column] = rng.poisson(rate, len(frame)).astype("float32")
            frame["horizon"] = float(horizon)
            frame["trips_per_day_28d"] = {"a": 7.0, "b": 15.0, "c": 30.0}[station]
            for target in TARGETS.values():
                # Demand follows the last hour, which the model sees and the baselines
                # (same hour a week ago, 8-week mean) do not.
                demand = frame[f"{target.removesuffix('_target')}_lag_0"]
                frame[target] = rng.poisson(demand + 0.5).astype("float32")
            frames.append(frame)
    features = pd.concat(frames, ignore_index=True)
    features["month"] = features["origin_hour"].dt.strftime("%Y-%m")
    features.to_parquet(root, partition_cols=["month"])
    return features


def test_load_rows_selects_by_target_hour_across_months(tmp_path):
    synthetic_features(tmp_path / "features")
    rows = load_rows(pd.Timestamp("2026-02-01"), pd.Timestamp("2026-02-01T05:00"), keys=True,
                     path=str(tmp_path / "features"))
    assert rows["target_hour"].between("2026-02-01", "2026-02-01T05:00").all()
    # Targets at 00:00-02:00 on 1 February come from January origins.
    assert (rows["origin_hour"] < "2026-02-01").sum() == 3 * (1 + 2 + 3)
    assert len(rows) == 3 * 3 * 6


# Gate ---------------------------------------------------------------------------------------


def scores(overall: float, tiers: dict[str, float], horizons: dict[str, float] | None = None):
    horizons = horizons or {"1": overall, "2": overall, "3": overall}
    one = {"overall": {"wape": overall},
           "horizon": {label: {"wape": value} for label, value in horizons.items()},
           "tier": {label: {"wape": value} for label, value in tiers.items()}}
    return {"pickups": one, "dropoffs": one, "net_flow": {}}


CHAMPION_SCORES = scores(0.80, {"5-10": 1.00, "10-20": 0.80, "20+": 0.60})
BASELINES = {"historical_average": scores(0.82, {}), "seasonal_naive": scores(0.95, {})}


def test_gate_passes_a_better_challenger_within_tolerance():
    # 5-10 is exactly 3% worse: allowed, since the tolerance is inclusive.
    challenger = scores(0.78, {"5-10": 1.03, "10-20": 0.79, "20+": 0.58})
    decision = compare(challenger, CHAMPION_SCORES, BASELINES, tolerance=0.03)
    assert decision["passed"], decision["failed"]
    assert {check["check"] for check in decision["checks"]} == {
        "beats_champion", "within_tolerance", "beats_baseline"}


def test_gate_names_every_failed_check():
    challenger = scores(0.79, {"5-10": 1.05, "10-20": 0.80, "20+": math.nan})
    decision = compare(challenger, CHAMPION_SCORES, BASELINES, tolerance=0.03)
    assert not decision["passed"]
    failed = [(c["target"], c["check"], c["label"]) for c in decision["checks"] if not c["passed"]]
    assert failed == [(target, "within_tolerance", "5-10") for target in ("pickups", "dropoffs")]
    assert "tier 5-10" in decision["failed"][0] and "1.0500" in decision["failed"][0]


def test_baseline_rule_is_waived_only_when_the_champion_also_trails_it():
    # The March case: the champion trails the historical average, the challenger beats the
    # champion but not the average. Promoting it replaces the worse model.
    trailing = {**BASELINES, "historical_average": scores(0.785, {})}
    challenger = scores(0.79, {"5-10": 1.0, "10-20": 0.79, "20+": 0.59})
    decision = compare(challenger, CHAMPION_SCORES, trailing, tolerance=0.03)
    assert decision["passed"], decision["failed"]
    waived = [c for c in decision["checks"] if c["waived"]]
    assert {(c["reference"], c["limit"]) for c in waived} == {("historical_average", 0.80)}
    assert "only had to beat the champion" in decision["waived"][0]
    # When the champion beats the baseline, the check applies in full and is not waived.
    champion = scores(0.70, {"5-10": 1.00, "10-20": 0.80, "20+": 0.60})
    worse = scores(0.73, {"5-10": 1.0, "10-20": 0.80, "20+": 0.60})
    decision = compare(worse, champion, {"historical_average": scores(0.72, {})}, 0.03)
    baseline = decision["checks"][-1]
    assert baseline["check"] == "beats_baseline" and not baseline["waived"]
    assert not baseline["passed"] and baseline["limit"] == 0.72


def test_gate_requires_strictly_beating_the_champion():
    decision = compare(CHAMPION_SCORES, CHAMPION_SCORES, BASELINES, tolerance=0.03)
    assert [c["check"] for c in decision["checks"] if not c["passed"]] == ["beats_champion"] * 2


# Registry -----------------------------------------------------------------------------------


class FakeRegistry:
    def __init__(self, champion: str | None):
        self.alias = champion
        self.tags: dict[str, dict] = {}

    def get_model_version_by_alias(self, name, alias):
        if self.alias is None:
            raise mlflow.exceptions.MlflowException("no alias")
        return SimpleNamespace(version=self.alias)

    def get_model_version(self, name, version):
        return SimpleNamespace(tags=self.tags.get(version, {}))

    def set_model_version_tag(self, name, version, key, value):
        self.tags.setdefault(version, {})[key] = value

    def set_registered_model_alias(self, name, alias, version):
        self.alias = version


def test_promote_then_roll_back_restores_the_previous_champion():
    client = FakeRegistry("1")
    registry.promote(client, "2", expected_champion="1")
    assert client.alias == "2" and client.tags["2"][registry.PROMOTED_FROM] == "1"
    assert registry.rollback(client) == {"from": "2", "to": "1"}
    assert client.alias == "1" and "rolled_back_at" in client.tags["2"]
    with pytest.raises(registry.RegistryError, match="nothing to roll back"):
        registry.rollback(client)


def test_promotion_refuses_when_the_champion_moved_during_training():
    client = FakeRegistry("3")
    with pytest.raises(registry.RegistryError, match="changed from 1 to 3"):
        registry.promote(client, "4", expected_champion="1")
    assert client.alias == "3"


# Trainer service ----------------------------------------------------------------------------


class FakeBackend(Backend):
    def __init__(self, clock="2026-03-11T23:00", running=True):
        self.clock, self.running, self.paused = clock, running, False
        self.calls: list[str] = []

    def replay_status(self):
        return {"clock": self.clock, "running": self.running, "paused": self.paused}

    def pause(self):
        self.calls.append("pause")
        self.paused = True
        return self.replay_status()

    def resume(self):
        self.calls.append("resume")
        self.paused = False
        return self.replay_status()

    def reload_model(self):
        self.calls.append("reload")
        return {"reloaded": True, "version": "2"}


RETRAIN = {"alerts": [{"status": "firing", "labels": {"alertname": "BikeshareForecastDegraded",
                                                      "action": "retrain"}}]}
DRIFT = {"alerts": [{"status": "firing", "labels": {"alertname": "BikeshareInputDrift",
                                                    "action": "investigate"}}]}


def outcome_count(outcome: str) -> float:
    return REGISTRY.get_sample_value("bikeshare_retrain_jobs_total", {"outcome": outcome}) or 0.0


def test_retrain_alert_pauses_trains_reloads_and_resumes():
    backend, seen = FakeBackend(), []

    def runner(as_of, trigger):
        seen.append((as_of, trigger, list(backend.calls)))
        return {"outcome": "promoted", "version": "2"}

    trainer = Trainer(backend, runner, cooldown_days=7)
    assert trainer.on_alerts(DRIFT)["decision"] == "ignored"
    promoted = outcome_count("promoted")
    assert trainer.on_alerts(RETRAIN)["decision"] == "started"
    trainer.wait()
    assert seen == [(pd.Timestamp("2026-03-11T23:00"), "BikeshareForecastDegraded", ["pause"])]
    assert backend.calls == ["pause", "reload", "resume"]
    assert outcome_count("promoted") == promoted + 1
    assert trainer.status()["history"][0]["version"] == "2"

    # A repeat of the same alert within the cooldown is ignored; a week later it is not.
    assert "cooldown" in trainer.on_alerts(RETRAIN)["reason"]
    backend.clock = "2026-03-18T23:00"
    assert trainer.on_alerts(RETRAIN)["decision"] == "started"
    trainer.wait()


def test_a_failed_job_still_resumes_the_replay():
    backend = FakeBackend()

    def runner(as_of, trigger):
        raise RuntimeError("the retraining job exited with code 137")

    trainer = Trainer(backend, runner, cooldown_days=7)
    trainer.on_alerts(RETRAIN)
    trainer.wait()
    assert backend.calls == ["pause", "resume"]
    record = trainer.status()["history"][0]
    assert record["outcome"] == "failed" and "137" in record["error"]


def test_only_one_job_runs_and_manual_jobs_skip_the_cooldown():
    backend, release = FakeBackend(running=False), threading.Event()

    def runner(as_of, trigger):
        release.wait(10)
        return {"outcome": "rejected"}

    trainer = Trainer(backend, runner, cooldown_days=7)
    app = create_app(trainer, client=FakeRegistry("1"))
    with TestClient(app) as client:
        started = client.post("/retrain", params={"as_of": "2026-04-01T23:00"}).json()
        assert started["job"]["as_of"] == "2026-04-01T23:00"
        assert client.post("/retrain").status_code == 409
        assert "already running" in client.post("/alertmanager", json=RETRAIN).json()["reason"]
        release.set()
        trainer.wait()
        # The replay was not running, so nothing was paused.
        assert backend.calls == []
        assert client.post("/retrain").json()["decision"] == "started"
        trainer.wait()
        status = client.get("/status").json()
        assert [entry["outcome"] for entry in status["history"]] == ["rejected", "rejected"]
        assert status["champion_version"] == "1"
        assert client.post("/rollback").status_code == 409
        assert "bikeshare_retrain_running 0.0" in client.get("/metrics").text


# Replay pause -------------------------------------------------------------------------------


def test_pausing_the_replay_holds_the_clock(store, data_dir, monkeypatch):
    monkeypatch.setattr(mlflow.pyfunc, "load_model", lambda uri: ConstantModel())
    models = ModelStore("demand", "champion", FakeClient())
    models.refresh()
    monitor = ReplayMonitor(store, models, RULES, str(data_dir / "panel"),
                            Thresholds(1.0, 1.0, 3), window_days=2, tier_bounds=[5, 10, 20],
                            load_profile=lambda run: None)
    first = store.first_origin
    monitor.start(first, first + pd.Timedelta(hours=47), delay_seconds=0.02)
    monitor.pause()
    held = monitor.status()
    assert held["paused"] and held["running"]
    assert monitor.status()["clock"] == held["clock"]
    monitor.resume()
    monitor.stop()
    assert not monitor.status()["paused"]
    assert pd.Timestamp(monitor.status()["clock"]) >= pd.Timestamp(held["clock"])
    with pytest.raises(RuntimeError, match="not running"):
        monitor.pause()


# A full job against a temporary MLflow registry ---------------------------------------------


@pytest.fixture
def tracking(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("bikeshare-demand")
    yield mlflow.MlflowClient(uri)
    mlflow.set_tracking_uri(None)


def test_job_trains_gates_registers_and_promotes(tmp_path, tracking):
    features = synthetic_features(tmp_path / "features")
    params = yaml.safe_load(Path("params.yaml").read_text())
    params["split"]["train"] = ["2026-01", "2026-01"]
    params["promotion"]["gate_days"] = 3
    params["retraining"].update(early_stopping_days=2, threads=2)
    params["monitoring"]["profile_sample_rows"] = 2_000
    best = {demand: {"config": {"learning_rate": 0.1, "num_leaves": 7, "min_data_in_leaf": 20,
                                "feature_fraction": 1.0, "bagging_fraction": 1.0,
                                "lambda_l2": 0.0}} for demand in TARGETS}
    (tmp_path / "best.json").write_text(json.dumps(best))

    # The champion forecasts 1 trip everywhere, which the challenger should beat.
    sample = features.head(200)
    flat = {demand: lgb.train({"objective": "poisson", "verbosity": -1, "learning_rate": 1e-9},
                              lgb.Dataset(sample[FEATURE_COLUMNS], np.ones(len(sample))),
                              num_boost_round=1) for demand in TARGETS}
    with mlflow.start_run() as run:
        log_model(flat, sample, tmp_path / "champion")
    register(run.info.run_id, tracking)
    tracking.set_registered_model_alias(MODEL_NAME, CHAMPION, "1")

    result = job.run(pd.Timestamp("2026-02-05T23:00"), "BikeshareForecastDegraded",
                     params=params, features_path=str(tmp_path / "features"),
                     best_params_path=tmp_path / "best.json", max_rounds=30, client=tracking)
    assert result["gate"]["passed"], result["gate"]["failed"]
    assert result["outcome"] == "promoted" and result["version"] == "2"
    assert result["windows"]["gate"] == "2026-02-03 00:00 to 2026-02-05 23:00"
    assert registry.champion_version(tracking) == "2"
    version = tracking.get_model_version(MODEL_NAME, "2")
    assert version.tags[registry.PROMOTED_FROM] == "1" and version.tags["gate"] == "passed"
    run = tracking.get_run(result["run_id"])
    assert run.data.tags["outcome"] == "promoted" and run.data.tags["stage"] == "retrain"
    assert "gate_challenger_pickups_wape" in run.data.metrics
    artifacts = {item.path for item in tracking.list_artifacts(result["run_id"])}
    assert {"gate.json", "monitoring"} <= artifacts
    profile = json.loads(Path(mlflow.artifacts.download_artifacts(
        run_id=result["run_id"], artifact_path="monitoring/reference_profile.json")).read_text())
    assert profile["window"].startswith("2026-01-01") and "pickups_forecast" in profile["columns"]
    # The promoted model loads and forecasts like any champion.
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{CHAMPION}")
    assert len(model.predict(sample[FEATURE_COLUMNS].astype("float64"))) == len(sample)

    assert recorded_jobs(tracking)[0]["rolled_back_at"] is None
    rolled = registry.rollback(tracking)
    assert rolled == {"from": "2", "to": "1"}
    [listed] = recorded_jobs(tracking)
    assert listed["outcome"] == "promoted" and listed["version"] == "2"
    assert listed["rolled_back_at"] is not None and listed["challenger_pickups_wape"] > 0
