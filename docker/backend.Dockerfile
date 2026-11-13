# Backend API and MLflow tracking server (one image, two compose services).
FROM python:3.12.16-slim-bookworm

# LightGBM needs the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.lock docker/backend-requirements.txt ./
RUN pip install --no-cache-dir -r backend-requirements.txt -c requirements.lock

COPY params.yaml ./
COPY src ./src

# Writable by any user: SQLite creates its journal next to the database file.
RUN mkdir -m 1777 /mlflow
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1 MLFLOW_DISABLE_AGENT_HINT=1 HOST=0.0.0.0 PORT=8000
USER 1000:1000
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"
CMD ["python", "-m", "bikeshare.serving.app"]
