"""Collect every method's metrics into one table: reports/comparison.md.

All methods are scored on the same fixed 1,000-lot sample of test lots.

Usage: python -m tender_scout.compare
"""

import json

import pandas as pd

from tender_scout.config import REPORTS


def load(path):
    return json.loads(path.read_text())


def rows() -> list[dict]:
    out = []
    base = REPORTS / "baseline_metrics.json"
    if base.exists():
        m = load(base)
        out.append({"method": "Majority class (always " + m["majority_class"] + ")",
                    "macro_f1": m["majority_test"]["macro_f1"], "accuracy": m["majority_test"]["accuracy"],
                    "usd_per_1000": 0.0, "note": "full test set"})
        full, unseen = m["test"]["macro_f1"], m["test_unseen_text"]["macro_f1"]
        out.append({"method": f"TF-IDF + linear SVM (C={m['C']})",
                    "macro_f1": m["eval_sample"]["macro_f1"], "accuracy": m["eval_sample"]["accuracy"],
                    "usd_per_1000": 0.0,
                    "note": f"full test macro-F1 {full}, unseen text {unseen}"})
    emb = REPORTS / "embed_metrics.json"
    if emb.exists():
        m = load(emb)
        out.append({"method": f"Embeddings {m['model']} ({m['dims']}d) + LogReg",
                    "macro_f1": m["eval_sample"]["macro_f1"], "accuracy": m["eval_sample"]["accuracy"],
                    "usd_per_1000": m["cost_usd_per_1000_lots"],
                    "note": f"trained on {m['n_train']} lots; {m['tokens_per_1000_lots']} tokens/1000"})
        if "tfidf_same_train_eval_sample" in m:
            c = m["tfidf_same_train_eval_sample"]
            out.append({"method": "TF-IDF + linear SVM, same training lots as the embeddings",
                        "macro_f1": c["macro_f1"], "accuracy": c["accuracy"], "usd_per_1000": 0.0,
                        "note": f"control: trained on the same {m['n_train']} lots"})
    for path in sorted(REPORTS.glob("llm_*_metrics.json")):
        m = load(path)
        out.append({"method": f"Zero-shot LLM {m['model']}",
                    "macro_f1": m["eval_sample"]["macro_f1"], "accuracy": m["eval_sample"]["accuracy"],
                    "usd_per_1000": m["cost_usd_per_1000_lots"],
                    "note": f"n={m['eval_sample']['n']}, {m['invalid_answers']} invalid, "
                            f"median {m['median_latency_ms']} ms"})
    return out


def main() -> None:
    table = pd.DataFrame(rows())
    if table.empty:
        raise SystemExit("no metrics yet: run baseline / bedrock_embed / bedrock_llm first")
    table["usd_per_1000"] = table["usd_per_1000"].map(lambda v: "n/a (fill prices.json)" if v is None else v)
    md = "# Comparison (same 1,000 test lots, March 2025)\n\n" + table.to_markdown(index=False) + "\n"
    (REPORTS / "comparison.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
