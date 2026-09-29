"""Download monthly ANAC CIG zips and extract the CSV.

The server sometimes cuts long downloads short and its firewall can answer
HTTP 200 with an HTML error page, so every download is validated as a real,
complete zip (CRC check) before it is accepted, and retried otherwise.

Usage: python -m tender_scout.download --year 2025 --months 01 02 03
"""

import argparse
import http.client
import logging
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

from tender_scout.config import ANAC_URL, DATA_RAW

log = logging.getLogger(__name__)
USER_AGENT = "tender-scout/0.1 (open-data research project)"


class DownloadError(Exception):
    pass


def validate_zip(path: Path, member: str) -> None:
    """Raise DownloadError unless path is a complete zip containing member."""
    if not zipfile.is_zipfile(path):
        raise DownloadError(f"{path.name} is not a zip (firewall page or truncated)")
    with zipfile.ZipFile(path) as zf:
        if member not in zf.namelist():
            raise DownloadError(f"{member} not in {path.name}: {zf.namelist()}")
        bad = zf.testzip()
        if bad is not None:
            raise DownloadError(f"CRC error in {bad}")


def download_month(year: str, month: str, dest: Path = DATA_RAW,
                   attempts: int = 6, wait: int = 10) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    csv_name = f"cig_csv_{year}_{month}.csv"
    csv_path = dest / csv_name
    if csv_path.exists():
        log.info("%s already present, skipping", csv_name)
        return csv_path

    url = ANAC_URL.format(year=year, month=month)
    zip_path = dest / f"cig_csv_{year}_{month}.zip"
    tmp = zip_path.with_suffix(".part")
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=600) as resp, open(tmp, "wb") as f:
                shutil.copyfileobj(resp, f)
            validate_zip(tmp, csv_name)
            tmp.replace(zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extract(csv_name, dest)
            log.info("%s ok (%.1f MB zip)", csv_name, zip_path.stat().st_size / 1e6)
            return csv_path
        except (OSError, http.client.HTTPException, DownloadError) as e:
            log.warning("%s attempt %d/%d failed: %s", csv_name, attempt, attempts, e)
            time.sleep(wait)
    tmp.unlink(missing_ok=True)
    raise DownloadError(f"giving up on {url} after {attempts} attempts")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", default="2025")
    parser.add_argument("--months", nargs="+", default=["01", "02", "03"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for month in args.months:
        download_month(args.year, month)


if __name__ == "__main__":
    main()
