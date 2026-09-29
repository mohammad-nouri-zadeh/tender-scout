"""Zero-shot LLM classification on Bedrock (Converse API).

The model sees the list of CPV divisions (named from the data, in Italian) and
one lot description, and must answer with the 2-digit code only. Runs on the
shared 1,000-lot evaluation sample; answers are cached per model.

Model IDs differ by region; in eu-central-1 many models are called through an
EU cross-region inference profile ("eu." prefix). Check the Bedrock console
(Model catalog / Cross-region inference) for the exact IDs enabled on your
account.

Usage: python -m tender_scout.bedrock_llm --model eu.amazon.nova-lite-v1:0
       python -m tender_scout.bedrock_llm --model <id> --limit 50   (cheap trial)
"""

import argparse
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from tender_scout import bedrock
from tender_scout.config import DATA_INTERIM, REPORTS
from tender_scout.data import load_eval_sample, load_labels, load_lots
from tender_scout.evaluate import per_class, scores, write_json

log = logging.getLogger(__name__)
DEFAULT_MODEL = "eu.amazon.nova-lite-v1:0"
INVALID = "??"


def system_prompt(labels: dict[str, str]) -> str:
    lines = "\n".join(f"{code}: {name}" for code, name in labels.items())
    return (
        "You classify Italian public procurement lots into CPV divisions.\n"
        "Answer with the 2-digit division code only, nothing else.\n\n"
        f"Divisions:\n{lines}"
    )


def parse_code(answer: str, valid: set[str]) -> str:
    """First 2-digit number in the answer that is a known division."""
    for match in re.findall(r"(?<!\d)(\d{2})(?!\d)", answer or ""):
        if match in valid:
            return match
    return INVALID


def answer_text(resp: dict) -> str:
    # Reasoning models return a reasoning block before the text block.
    for block in resp["output"]["message"]["content"]:
        if "text" in block:
            return block["text"]
    return ""


def classify_one(client, model_id: str, system: str, text: str, max_tokens: int) -> dict:
    resp = bedrock.with_backoff(lambda: client.converse(
        modelId=model_id,
        system=[{"text": system}],
        messages=[{"role": "user", "content": [{"text": f"Lotto: {text}\nCodice:"}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
    ))
    usage = resp.get("usage", {})
    return {
        "answer": answer_text(resp),
        "input_tokens": usage.get("inputTokens", 0),
        "output_tokens": usage.get("outputTokens", 0),
        "latency_ms": resp.get("metrics", {}).get("latencyMs"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, help="only the first N lots of the sample")
    parser.add_argument("--max-tokens", type=int, default=10,
                        help="raise to ~1000 for reasoning models such as gpt-oss")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    labels = load_labels()
    system = system_prompt(labels)
    sample = load_eval_sample(load_lots())
    if args.limit:
        sample = sample.head(args.limit)

    client = bedrock.client()
    cache = bedrock.JsonlCache(DATA_INTERIM / f"llm_{bedrock.safe_name(args.model)}.jsonl")
    todo = [c for c in sample["cig"] if cache.get(c) is None]
    text_by_cig = dict(zip(sample["cig"], sample["text"], strict=True))
    log.info("%d lots, %d to classify with %s", len(sample), len(todo), args.model)

    def work(cig):
        return cig, classify_one(client, args.model, system, text_by_cig[cig], args.max_tokens)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (cig, res) in enumerate(pool.map(work, todo), 1):
            cache.put({"key": cig, **res})
            if i % 100 == 0:
                log.info("%d/%d", i, len(todo))
    run_seconds = time.perf_counter() - t0

    rows = [cache.get(c) for c in sample["cig"]]
    pred = [parse_code(r["answer"], set(labels)) for r in rows]
    n = len(rows)
    tin = sum(r["input_tokens"] for r in rows)
    tout = sum(r["output_tokens"] for r in rows)
    latencies = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    results = {
        "model": args.model,
        "eval_sample": scores(sample["cpv2"], pred),
        "invalid_answers": pred.count(INVALID),
        "input_tokens_per_1000_lots": round(tin / n * 1000),
        "output_tokens_per_1000_lots": round(tout / n * 1000),
        "cost_usd_per_1000_lots": bedrock.cost_usd(args.model, round(tin / n * 1000), round(tout / n * 1000)),
        "median_latency_ms": float(pd.Series(latencies).median()) if latencies else None,
        "run_seconds": round(run_seconds, 1),
    }
    name = bedrock.safe_name(args.model)
    write_json(REPORTS / f"llm_{name}_metrics.json", results)
    per_class(sample["cpv2"], pred, labels).to_csv(REPORTS / f"llm_{name}_per_class.csv", index=False)
    pd.DataFrame({"cig": sample["cig"], "true": sample["cpv2"], "pred": pred,
                  "answer": [r["answer"] for r in rows]}).to_csv(REPORTS / f"pred_llm_{name}.csv", index=False)
    log.info("%s", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
