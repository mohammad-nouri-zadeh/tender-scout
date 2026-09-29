"""Load the prepared lots and the fixed evaluation sample."""

import json

import pandas as pd

from tender_scout.config import EVAL_SAMPLE_PATH, LABELS_PATH, LOTS_PATH


def load_lots() -> pd.DataFrame:
    if not LOTS_PATH.exists():
        raise SystemExit(f"{LOTS_PATH} missing: run `python -m tender_scout.prepare` first")
    return pd.read_parquet(LOTS_PATH)


def load_eval_sample(lots: pd.DataFrame) -> pd.DataFrame:
    cigs = pd.read_csv(EVAL_SAMPLE_PATH, dtype=str)["cig"]
    return lots[lots["cig"].isin(cigs)].sort_values("cig").reset_index(drop=True)


def load_labels() -> dict[str, str]:
    return json.loads(LABELS_PATH.read_text())
