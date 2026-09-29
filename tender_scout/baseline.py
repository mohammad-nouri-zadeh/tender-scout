"""Classic ML: TF-IDF (words + character n-grams) and a linear SVM.

Why a linear SVM and not logistic regression: at this scale (about 230,000
lots, 46 classes, 250,000+ features) scikit-learn's LogisticRegression took
over 10 minutes per fit in a benchmark, LinearSVC about 2 minutes, with a
better macro-F1. Both are linear models on the same features.

1. Majority-class baseline (always predict the biggest division).
2. Tune C on validation (train -> val), pick the best macro-F1.
3. Refit on train + val, score once on test: full test, unseen-text test
   and the shared 1,000-lot evaluation sample.

Usage: python -m tender_scout.baseline            (tunes C; about 10-20 minutes)
       python -m tender_scout.baseline --c 0.3    (skip tuning)
"""

import argparse
import logging
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from tender_scout.config import MODELS, REPORTS
from tender_scout.data import load_eval_sample, load_labels, load_lots
from tender_scout.evaluate import per_class, scores, top_confusions, unseen_mask, write_json

log = logging.getLogger(__name__)
C_GRID = [0.1, 0.3, 1.0]


def make_features(min_df: int = 2) -> FeatureUnion:
    # Word n-grams capture phrases ("manutenzione verde"); character n-grams
    # are robust to Italian inflections, typos and truncated words ("MANUT.").
    words = TfidfVectorizer(strip_accents="unicode", ngram_range=(1, 2), min_df=min_df,
                            max_features=200_000, sublinear_tf=True)
    chars = TfidfVectorizer(strip_accents="unicode", analyzer="char_wb", ngram_range=(3, 5),
                            min_df=min_df, max_features=200_000, sublinear_tf=True)
    return FeatureUnion([("words", words), ("chars", chars)])


def make_pipeline(c: float = 0.3, min_df: int = 2) -> Pipeline:
    return Pipeline([("features", make_features(min_df)), ("clf", LinearSVC(C=c))])


def tune(train: pd.DataFrame, val: pd.DataFrame, grid: list[float]) -> list[dict]:
    """Fit TF-IDF once on train, then one SVM per C; score each on val."""
    features = make_features()
    x_train = features.fit_transform(train["text"])
    x_val = features.transform(val["text"])
    out = []
    for c in grid:
        t0 = time.perf_counter()
        clf = LinearSVC(C=c).fit(x_train, train["cpv2"])
        s = scores(val["cpv2"], clf.predict(x_val))
        out.append({"C": c, "val": s, "fit_seconds": round(time.perf_counter() - t0, 1)})
        log.info("C=%s val %s", c, s)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c", type=float, help="skip tuning and use this C")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    lots = load_lots()
    names = load_labels()
    train, val, test = (lots[lots["split"] == s] for s in ("train", "val", "test"))
    sample = load_eval_sample(lots)
    log.info("train %d | val %d | test %d | eval sample %d", len(train), len(val), len(test), len(sample))

    majority = train["cpv2"].mode().iloc[0]
    results = {"majority_class": majority,
               "majority_test": scores(test["cpv2"], np.full(len(test), majority))}

    tuning = []
    if args.c is None:
        tuning = tune(train, val, C_GRID)
        best_c = max(tuning, key=lambda r: r["val"]["macro_f1"])["C"]
    else:
        best_c = args.c
    results["tuning"] = tuning
    results["C"] = best_c

    train_val = pd.concat([train, val])
    t0 = time.perf_counter()
    model = make_pipeline(best_c).fit(train_val["text"], train_val["cpv2"])
    results["fit_seconds"] = round(time.perf_counter() - t0, 1)

    t0 = time.perf_counter()
    pred = model.predict(test["text"])
    results["predict_ms_per_1000"] = round((time.perf_counter() - t0) / len(test) * 1e6, 2)

    unseen = unseen_mask(train_val["text"], test["text"]).to_numpy()
    results["test"] = scores(test["cpv2"], pred)
    results["test_unseen_text"] = scores(test["cpv2"][unseen], pred[unseen])

    sample_pred = model.predict(sample["text"])
    results["eval_sample"] = scores(sample["cpv2"], sample_pred)

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS / "baseline.joblib")
    write_json(REPORTS / "baseline_metrics.json", results)
    per_class(test["cpv2"], pred, names).to_csv(REPORTS / "baseline_per_class.csv", index=False)
    top_confusions(test["cpv2"], pred).to_csv(REPORTS / "baseline_confusions.csv", index=False)
    pd.DataFrame({"cig": sample["cig"], "true": sample["cpv2"], "pred": sample_pred}).to_csv(
        REPORTS / "pred_baseline.csv", index=False)
    log.info("test %s | unseen %s | sample %s",
             results["test"], results["test_unseen_text"], results["eval_sample"])


if __name__ == "__main__":
    main()
