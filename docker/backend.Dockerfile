# Backend API, MLflow tracking server and retraining service. One dependency layer, three
# uses: `backend` (the default, last stage) serves the API and runs the MLflow server;
# `trainer` runs the retraining service on the checked-out repository.
FROM python:3.12.16-slim-bookworm AS base

# LightGBM needs the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.lock docker/backend-requirements.txt ./
RUN pip install --no-cache-dir -r backend-requirements.txt -c requirements.lock

# Writable by any user: SQLite creates its journal next to the database file.
RUN mkdir -m 1777 /mlflow
ENV PYTHONUNBUFFERED=1 MLFLOW_DISABLE_AGENT_HINT=1 HOST=0.0.0.0


FROM base AS trainer
# Git records the commit and whether the code had uncommitted changes (lineage tags).
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
# The repository is mounted read-only at /repo, so jobs run the checked-out code on the
# DVC-tracked features, as `dvc repro` does. Nothing is written into it.
WORKDIR /repo
ENV PYTHONPATH=/repo/src PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 HOME=/tmp PORT=8001
USER 1000:1000
EXPOSE 8001
CMD ["python", "-m", "bikeshare.retraining.service"]


FROM base AS backend
COPY params.yaml ./
COPY src ./src
ENV PYTHONPATH=/app/src PORT=8000
USER 1000:1000
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"
CMD ["python", "-m", "bikeshare.serving.app"]
