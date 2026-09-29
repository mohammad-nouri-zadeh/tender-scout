"""Bedrock embeddings + logistic regression.

Embeds a random sample of training lots and the shared evaluation sample with
Amazon Titan Text Embeddings v2, trains logistic regression on the vectors and
scores it on the evaluation sample. Every embedding is cached on disk.

Usage: python -m tender_scout.bedrock_embed --n-train 20000
"""

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from tender_scout import bedrock
from tender_scout.baseline import make_pipeline
from tender_scout.config import DATA_INTERIM, REPORTS, SEED
from tender_scout.data import load_eval_sample, load_labels, load_lots
from tender_scout.evaluate import per_class, scores, write_json

log = logging.getLogger(__name__)
DEFAULT_MODEL = "amazon.titan-embed-text-v2:0"


def embed_one(client, model_id: str, text: str, dims: int) -> dict:
    body = json.dumps({"inputText": text, "dimensions": dims, "normalize": True})
    resp = bedrock.with_backoff(lambda: client.invoke_model(
        modelId=model_id, body=body, contentType="application/json", accept="application/json"))
    out = json.loads(resp["body"].read())
    return {"embedding": out["embedding"], "tokens": out.get("inputTextTokenCount", 0)}


def embed_texts(client, texts: list[str], model_id: str, dims: int,
                cache: bedrock.JsonlCache, workers: int = 8) -> tuple[np.ndarray, int]:
    """Return (matrix, tokens paid in this run). Cached texts cost nothing."""
    todo = sorted({t for t in texts if cache.get(bedrock.text_key(t)) is None})
    log.info("%d texts, %d to embed", len(set(texts)), len(todo))
    tokens = 0

    def work(text):
        return text, embed_one(client, model_id, text, dims)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (text, res) in enumerate(pool.map(work, todo), 1):
            cache.put({"key": bedrock.text_key(text), **res})
            tokens += res["tokens"]
            if i % 1000 == 0:
                log.info("embedded %d/%d", i, len(todo))
    matrix = np.array([cache.get(bedrock.text_key(t))["embedding"] for t in texts], dtype=np.float32)
    return matrix, tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dims", type=int, default=512, choices=[256, 512, 1024])
    parser.add_argument("--n-train", type=int, default=20000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    lots = load_lots()
    train = lots[lots["split"].isin(["train", "val"])]
    train = train.sample(n=min(args.n_train, len(train)), random_state=SEED)
    sample = load_eval_sample(lots)

    client = bedrock.client()
    cache = bedrock.JsonlCache(DATA_INTERIM / f"emb_{bedrock.safe_name(args.model)}_{args.dims}.jsonl")
    t0 = time.perf_counter()
    x_train, tok_train = embed_texts(client, train["text"].tolist(), args.model, args.dims, cache, args.workers)
    x_eval, tok_eval = embed_texts(client, sample["text"].tolist(), args.model, args.dims, cache, args.workers)
    embed_seconds = time.perf_counter() - t0

    clf = LogisticRegression(C=10.0, max_iter=2000).fit(x_train, train["cpv2"])
    pred = clf.predict(x_eval)

    # Control: the classic model trained on the very same lots, so the gap is not just training size.
    control = make_pipeline().fit(train["text"], train["cpv2"])
    control_pred = control.predict(sample["text"])

    # Tokens for the eval sample alone give the inference cost per 1,000 lots.
    eval_tokens = sum(cache.get(bedrock.text_key(t))["tokens"] for t in sample["text"])
    per_1000 = eval_tokens / len(sample) * 1000
    results = {
        "model": args.model,
        "dims": args.dims,
        "n_train": len(train),
        "eval_sample": scores(sample["cpv2"], pred),
        "tfidf_same_train_eval_sample": scores(sample["cpv2"], control_pred),
        "tokens_paid_this_run": tok_train + tok_eval,
        "tokens_per_1000_lots": round(per_1000),
        "cost_usd_per_1000_lots": bedrock.cost_usd(args.model, round(per_1000)),
        "embed_seconds_this_run": round(embed_seconds, 1),
    }
    write_json(REPORTS / "embed_metrics.json", results)
    per_class(sample["cpv2"], pred, load_labels()).to_csv(REPORTS / "embed_per_class.csv", index=False)
    pd.DataFrame({"cig": sample["cig"], "true": sample["cpv2"], "pred": pred}).to_csv(
        REPORTS / "pred_embed.csv", index=False)
    log.info("%s", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
