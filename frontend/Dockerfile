# Streamlit frontend (Decision D8). Talks to the backend over REST at BACKEND_URL.
FROM python:3.12.16-slim-bookworm

WORKDIR /app
COPY frontend/requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

COPY frontend/app.py frontend/api.py ./
COPY frontend/views ./views
COPY frontend/.streamlit ./.streamlit

ENV PYTHONUNBUFFERED=1 BACKEND_URL=http://backend:8000
USER 1000:1000
EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
