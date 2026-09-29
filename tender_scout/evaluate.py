"""Metrics shared by every method, so the comparison is like for like."""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


def normalise(text: pd.Series) -> pd.Series:
    return text.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def unseen_mask(train_text: pd.Series, test_text: pd.Series) -> pd.Series:
    """True for test lots whose exact text never appeared in training.

    About 13% of lots repeat an earlier text (recurring purchases). Scoring on
    unseen texts only shows how well a model generalises beyond memory.
    """
    seen = set(normalise(train_text))
    return ~normalise(test_text).isin(seen)


def scores(y_true, y_pred) -> dict:
    return {
        "n": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
    }


def per_class(y_true, y_pred, names: dict[str, str] | None = None) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    rows = {k: v for k, v in rep.items() if k not in ("accuracy", "macro avg", "weighted avg")}
    df = pd.DataFrame(rows).T.rename_axis("cpv2").reset_index()
    df["support"] = df["support"].astype(int)
    if names:
        df.insert(1, "name", df["cpv2"].map(names))
    return df.sort_values("f1-score")


def top_confusions(y_true, y_pred, k: int = 20) -> pd.DataFrame:
    pairs = pd.DataFrame({"true": list(y_true), "pred": list(y_pred)})
    wrong = pairs[pairs["true"] != pairs["pred"]]
    return wrong.value_counts().head(k).rename("count").reset_index()


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
