"""Move the champion alias: promotion after the gate, and rollback to the previous champion."""
from datetime import UTC, datetime

import mlflow
from mlflow import MlflowClient

from bikeshare.modeling.register import CHAMPION, MODEL_NAME

PROMOTED_FROM = "promoted_from"


class RegistryError(RuntimeError):
    """The registry is not in a state that allows the change."""


def champion_version(client: MlflowClient) -> str | None:
    """As a string: MLflow returns an int from a local store and a string over HTTP."""
    try:
        return str(client.get_model_version_by_alias(MODEL_NAME, CHAMPION).version)
    except mlflow.exceptions.MlflowException:
        return None


def promote(client: MlflowClient, version: str, expected_champion: str | None) -> None:
    """Make version the champion, unless the champion moved since the gate compared them."""
    current = champion_version(client)
    if current != expected_champion:
        raise RegistryError(f"the champion changed from {expected_champion} to {current} "
                            "while retraining, so the gate result no longer applies")
    if current is not None:
        client.set_model_version_tag(MODEL_NAME, version, PROMOTED_FROM, current)
    client.set_registered_model_alias(MODEL_NAME, CHAMPION, version)


def rollback(client: MlflowClient) -> dict[str, str]:
    """Point the champion alias back at the version the current champion replaced."""
    current = champion_version(client)
    if current is None:
        raise RegistryError("there is no champion to roll back")
    previous = client.get_model_version(MODEL_NAME, current).tags.get(PROMOTED_FROM)
    if not previous:
        raise RegistryError(f"version {current} was not promoted over another version, "
                            "so there is nothing to roll back to")
    client.set_registered_model_alias(MODEL_NAME, CHAMPION, previous)
    client.set_model_version_tag(MODEL_NAME, current, "rolled_back_at",
                                 datetime.now(UTC).isoformat(timespec="seconds"))
    return {"from": current, "to": previous}
