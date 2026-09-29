"""Paths, data source and shared settings."""

import os
from pathlib import Path

# TENDER_SCOUT_ROOT lets tests point the pipeline at another folder.
ROOT = Path(os.environ.get("TENDER_SCOUT_ROOT", Path(__file__).resolve().parent.parent))
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

LOTS_PATH = DATA_INTERIM / "lots.parquet"
EVAL_SAMPLE_PATH = DATA_INTERIM / "eval_sample.csv"
LABELS_PATH = REPORTS / "labels.json"
PRICES_PATH = ROOT / "prices.json"

# ANAC open data, CIG dataset, one zip per month (CC BY-SA 4.0).
ANAC_URL = (
    "https://dati.anticorruzione.it/opendata/download/dataset/"
    "cig-{year}/filesystem/cig_csv_{year}_{month}.zip"
)

# Columns read from the 61 in the ANAC CSV. Outcome columns (ESITO, dates of
# award) are left out on purpose: they are known only after publication.
# descrizione_cpv is read only to name the labels, never used as a feature.
COLUMNS = [
    "cig",
    "numero_gara",
    "oggetto_gara",
    "oggetto_lotto",
    "oggetto_principale_contratto",
    "n_lotti_componenti",
    "importo_lotto",
    "data_pubblicazione",
    "cod_cpv",
    "descrizione_cpv",
    "flag_prevalente",
]

# Time-based split: learn from the past, test on the future.
DEFAULT_SPLIT = {"2025_01": "train", "2025_02": "val", "2025_03": "test"}

AWS_REGION = "eu-central-1"
SEED = 42
EVAL_SAMPLE_SIZE = 1000
