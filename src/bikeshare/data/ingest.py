"""Download Divvy monthly trip files and one snapshot of station information.

Spark cannot read .zip archives, so each month is recompressed to .csv.gz.
"""
import gzip
import io
import json
import logging
import shutil
import time
import zipfile
from pathlib import Path

import requests

from bikeshare.config import load_params, month_range

log = logging.getLogger(__name__)

TRIP_URL = "https://divvy-tripdata.s3.amazonaws.com/{month}-divvy-tripdata.zip"
STATION_INFO_URL = "https://gbfs.lyft.com/gbfs/1.1/chi/en/station_information.json"
RAW_DIR = Path("data/raw")
MAX_ATTEMPTS = 3
TIMEOUT_SECONDS = 120


def fetch(url: str) -> bytes:
    """Download a URL, retrying on network and server failures."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.get(url, timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.content
        except requests.RequestException as error:
            if attempt == MAX_ATTEMPTS:
                raise
            log.warning("Attempt %d failed for %s: %s", attempt, url, error)
            time.sleep(2**attempt)
    raise RuntimeError(f"unreachable: no attempts made for {url}")


def recompress_month(archive: bytes, destination: Path) -> None:
    """Write the single CSV inside a monthly archive to a gzip file."""
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        members = [
            name
            for name in bundle.namelist()
            if name.endswith(".csv") and not name.startswith("__MACOSX")
        ]
        if len(members) != 1:
            raise ValueError(f"expected one CSV for {destination.name}, found {members}")
        with bundle.open(members[0]) as source, gzip.open(destination, "wb") as target:
            shutil.copyfileobj(source, target)


def ingest_trips(months: list[str]) -> None:
    trip_dir = RAW_DIR / "trips"
    trip_dir.mkdir(parents=True, exist_ok=True)
    for month in months:
        compact = month.replace("-", "")
        recompress_month(fetch(TRIP_URL.format(month=compact)), trip_dir / f"{month}.csv.gz")
        log.info("Ingested %s", month)


def ingest_station_snapshot() -> None:
    """Save today's station list. Capacity is a snapshot, not a historical record."""
    snapshot = json.loads(fetch(STATION_INFO_URL))
    (RAW_DIR / "station_information.json").write_text(json.dumps(snapshot))
    log.info("Saved %d stations", len(snapshot["data"]["stations"]))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    params = load_params()["data"]
    ingest_trips(month_range(params["first_month"], params["last_month"]))
    ingest_station_snapshot()


if __name__ == "__main__":
    main()
