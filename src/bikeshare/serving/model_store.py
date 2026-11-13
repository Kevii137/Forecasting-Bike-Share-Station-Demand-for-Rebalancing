"""Hold the champion model in memory and swap it when the registry alias moves (D7)."""
import logging
import sys
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime

import mlflow
import mlflow.pyfunc  # explicit: mlflow exposes pyfunc through a lazy proxy
from mlflow import MlflowClient

log = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    model: object
    version: str
    run_id: str
    tags: dict[str, str]
    loaded_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))


def _load_keeping_path(uri: str):
    """Load a pyfunc without letting its bundled code shadow the service's own package.

    MLflow puts the model's copy of src/bikeshare (a snapshot from training time) first on
    sys.path. Modules already imported are unaffected, but processes started later, such
    as the feature workers, would import the snapshot instead.
    """
    saved = list(sys.path)
    try:
        return mlflow.pyfunc.load_model(uri)
    finally:
        sys.path[:] = saved


class ModelStore:
    """Load models:/<name>@<alias> and reload it when the alias points to a new version."""

    def __init__(self, name: str, alias: str, client: MlflowClient | None = None):
        self.name = name
        self.alias = alias
        self.client = client or MlflowClient()
        self.current: LoadedModel | None = None
        self.last_error: str | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def refresh(self) -> bool:
        """Load the alias target if it changed. Returns True when a new model was loaded."""
        with self._lock:
            try:
                target = self.client.get_model_version_by_alias(self.name, self.alias)
                if self.current and self.current.version == target.version:
                    self.last_error = None
                    return False
                # Load by version number, so a move during loading cannot mix versions.
                model = _load_keeping_path(f"models:/{self.name}/{target.version}")
            except Exception as error:  # noqa: BLE001 - keep serving the current model on any failure
                self.last_error = f"{type(error).__name__}: {error}"
                log.warning("Model refresh failed: %s", self.last_error)
                return False
            previous = self.current.version if self.current else None
            self.current = LoadedModel(model, target.version, target.run_id, dict(target.tags))
            self.last_error = None
            log.info("Loaded %s version %s (was %s)", self.name, target.version, previous)
            return True

    def poll(self, interval_seconds: float) -> threading.Thread:
        """Refresh in a background thread until stop() is called."""
        def loop() -> None:
            while not self._stop.wait(interval_seconds):
                self.refresh()

        thread = threading.Thread(target=loop, name="model-poll", daemon=True)
        thread.start()
        return thread

    def stop(self) -> None:
        self._stop.set()
