"""Turn raw monthly ANAC CSVs into one clean table of lots with a label.

Decisions (each one checked on Q1 2025 data, see README):
- One row per lot (CIG). A lot can list several CPV codes; ANAC marks the main
  one with flag_prevalente == "1". We keep only that row.
- Label = CPV division = first 2 digits of cod_cpv (e.g. "45112000-5" -> "45").
  Lots whose code is 99999999 ("CPV not available") have no label and are dropped.
- Input text = oggetto_lotto, plus oggetto_gara when it adds something.
- Split by publication month (file), never at random.

Usage: python -m tender_scout.prepare
       python -m tender_scout.prepare --split 2025_01=train 2025_02=val 2025_03=test
"""

import argparse
import json
import logging
import re

import pandas as pd

from tender_scout.config import (
    COLUMNS,
    DATA_RAW,
    DEFAULT_SPLIT,
    EVAL_SAMPLE_PATH,
    EVAL_SAMPLE_SIZE,
    LABELS_PATH,
    LOTS_PATH,
    REPORTS,
    SEED,
)

log = logging.getLogger(__name__)
# ANAC writes CPV codes both with and without the check digit ("45112000-5", "45112000").
CPV_PATTERN = re.compile(r"^\d{8}(-\d)?$")
# "Cpv prevalente non disponibile": the lot has no known label.
MISSING_CPV = "99999999"


def load_month(key: str) -> pd.DataFrame:
    """Read one month; dtype=str keeps codes like '09...' and '046015' intact."""
    path = DATA_RAW / f"cig_csv_{key}.csv"
    df = pd.read_csv(path, sep=";", dtype=str, usecols=COLUMNS)
    df["file_month"] = key
    return df


def build_text(df: pd.DataFrame) -> pd.Series:
    lotto = df["oggetto_lotto"].fillna("").str.strip()
    gara = df["oggetto_gara"].fillna("").str.strip()
    text = lotto.where(gara.eq(lotto) | gara.eq(""), lotto + " | " + gara)
    return text.str.replace(r"\s+", " ", regex=True).str.strip()


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep one row per lot with a valid prevalent CPV; return (lots, log)."""
    stats = {
        "raw_rows": len(raw),
        "raw_cigs": int(raw["cig"].nunique()),
        "rows_per_cig_gt1": int(raw["cig"].duplicated().sum()),
    }

    lots = raw[raw["flag_prevalente"] == "1"].copy()
    lost = sorted(set(raw["cig"]) - set(lots["cig"]))
    stats["cigs_without_prevalent_cpv"] = len(lost)
    stats["cigs_without_prevalent_cpv_examples"] = lost[:10]

    dup = lots["cig"].duplicated()
    stats["cigs_with_several_prevalent_rows"] = int(dup.sum())
    lots = lots[~dup]

    code = lots["cod_cpv"].fillna("")
    missing = code.str.startswith(MISSING_CPV)
    stats["cpv_not_available_dropped"] = int(missing.sum())
    valid = code.str.match(CPV_PATTERN) & ~missing
    stats["invalid_cpv_dropped"] = int((~valid & ~missing).sum())
    lots = lots[valid].copy()

    lots["cpv2"] = lots["cod_cpv"].str[:2]
    lots["text"] = build_text(lots)
    empty = lots["text"].eq("")
    stats["empty_text_dropped"] = int(empty.sum())
    lots = lots[~empty].copy()
    stats["text_shorter_than_5_chars_kept"] = int(lots["text"].str.len().lt(5).sum())

    stats["lots"] = len(lots)
    stats["classes"] = int(lots["cpv2"].nunique())
    return lots, stats


def label_names(raw: pd.DataFrame) -> dict[str, str]:
    """Name each division from the data itself.

    Prefer the description of the division-level code (XX000000); otherwise use
    the most frequent description seen in that division.
    """
    d = raw[["cod_cpv", "descrizione_cpv"]].dropna()
    d = d[d["cod_cpv"].str.match(CPV_PATTERN) & ~d["cod_cpv"].str.startswith(MISSING_CPV)]
    d = d.assign(cpv2=d["cod_cpv"].str[:2])
    names = {}
    for cpv2, grp in d.groupby("cpv2"):
        top = grp[grp["cod_cpv"].str[2:8] == "000000"]["descrizione_cpv"]
        source = top if len(top) else grp["descrizione_cpv"]
        names[cpv2] = source.mode().iloc[0]
    return dict(sorted(names.items()))


def assign_split(lots: pd.DataFrame, split: dict[str, str]) -> pd.DataFrame:
    lots = lots.copy()
    lots["split"] = lots["file_month"].map(split)
    return lots


def eval_sample(lots: pd.DataFrame, n: int = EVAL_SAMPLE_SIZE, seed: int = SEED) -> pd.Series:
    """Fixed random sample of test lots; every method is scored on the same one."""
    test = lots[lots["split"] == "test"]
    return test.sample(n=min(n, len(test)), random_state=seed)["cig"].sort_values()


def parse_split(items: list[str] | None) -> dict[str, str]:
    if not items:
        return dict(DEFAULT_SPLIT)
    split = dict(item.split("=", 1) for item in items)
    bad = set(split.values()) - {"train", "val", "test"}
    if bad:
        raise SystemExit(f"unknown split names: {bad}")
    return split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", nargs="+", help="MONTH=train|val|test, e.g. 2025_01=train")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    split = parse_split(args.split)

    raw = pd.concat([load_month(k) for k in split], ignore_index=True)
    lots, stats = clean(raw)
    lots = assign_split(lots, split)
    stats["rows_per_split"] = lots["split"].value_counts().to_dict()
    stats["lots_per_tender_months_gt1"] = int(
        (lots.groupby("numero_gara")["file_month"].nunique() > 1).sum()
    )

    keep = ["cig", "numero_gara", "file_month", "split", "text", "cpv2",
            "oggetto_principale_contratto", "importo_lotto", "data_pubblicazione"]
    LOTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    lots[keep].to_parquet(LOTS_PATH, index=False)
    eval_sample(lots).to_csv(EVAL_SAMPLE_PATH, index=False)
    LABELS_PATH.write_text(json.dumps(label_names(raw), ensure_ascii=False, indent=2))
    (REPORTS / "cleaning_log.json").write_text(json.dumps(stats, indent=2))
    log.info("cleaning log: %s", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
