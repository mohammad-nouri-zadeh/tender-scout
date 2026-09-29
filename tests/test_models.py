import io
import json

import pandas as pd
import pytest
from botocore.exceptions import ClientError

from tender_scout import bedrock
from tender_scout.baseline import make_pipeline
from tender_scout.bedrock_embed import embed_texts
from tender_scout.bedrock_llm import answer_text, classify_one, parse_code, system_prompt
from tender_scout.evaluate import scores, unseen_mask


def test_pipeline_learns_separable_classes():
    text = ["fornitura siringhe", "lavori strade asfalto", "servizio riscossione tributi"] * 20
    y = ["33", "45", "79"] * 20
    model = make_pipeline(c=1.0, min_df=1).fit(text, y)
    assert list(model.predict(["siringhe sterili", "asfalto strade"])) == ["33", "45"]


def test_unseen_mask_ignores_case_and_spaces():
    train = pd.Series(["Fornitura  GAS"])
    test = pd.Series(["fornitura gas", "fornitura acqua"])
    assert unseen_mask(train, test).tolist() == [False, True]


def test_macro_f1_punishes_majority_guess():
    s = scores(["33"] * 8 + ["45", "79"], ["33"] * 10)
    assert s["accuracy"] == 0.8
    assert s["macro_f1"] < 0.4


def test_parse_code():
    valid = {"09", "33", "45"}
    assert parse_code("33", valid) == "33"
    assert parse_code("Codice: 45.", valid) == "45"
    assert parse_code("45112000", valid) == "??"  # an 8-digit code is not an answer
    assert parse_code("99", valid) == "??"
    assert parse_code("", valid) == "??"


def test_system_prompt_lists_labels():
    p = system_prompt({"09": "PETROLIO", "33": "APPARECCHIATURE MEDICHE"})
    assert "09: PETROLIO" in p and "33: APPARECCHIATURE MEDICHE" in p


class FakeConverse:
    def converse(self, **kwargs):
        self.kwargs = kwargs
        return {
            "output": {"message": {"content": [{"reasoningContent": {}}, {"text": "33"}]}},
            "usage": {"inputTokens": 900, "outputTokens": 2},
            "metrics": {"latencyMs": 250},
        }


def test_classify_one_reads_text_after_reasoning_block():
    client = FakeConverse()
    res = classify_one(client, "m", "sys", "SIRINGHE", 10)
    assert res == {"answer": "33", "input_tokens": 900, "output_tokens": 2, "latency_ms": 250}
    assert client.kwargs["inferenceConfig"]["temperature"] == 0.0
    assert answer_text({"output": {"message": {"content": []}}}) == ""


class FakeEmbed:
    calls = 0

    def invoke_model(self, modelId, body, **kwargs):
        self.calls += 1
        n = len(json.loads(body)["inputText"])
        return {"body": io.BytesIO(json.dumps({"embedding": [float(n), 1.0],
                                               "inputTextTokenCount": 3}).encode())}


def test_embed_cache_avoids_paying_twice(tmp_path):
    client = FakeEmbed()
    cache = bedrock.JsonlCache(tmp_path / "c.jsonl")
    x, tokens = embed_texts(client, ["ab", "abc", "ab"], "m", 2, cache, workers=2)
    assert x.shape == (3, 2) and tokens == 6 and client.calls == 2
    cache2 = bedrock.JsonlCache(tmp_path / "c.jsonl")  # reload from disk
    _, tokens2 = embed_texts(client, ["ab", "abc"], "m", 2, cache2)
    assert tokens2 == 0 and client.calls == 2


def test_cost_uses_prices_file(tmp_path, monkeypatch):
    prices = tmp_path / "prices.json"
    prices.write_text(json.dumps({"m": {"input_per_1k": 0.5, "output_per_1k": 2.0}}))
    monkeypatch.setattr(bedrock, "PRICES_PATH", prices)
    assert bedrock.cost_usd("m", 1000, 500) == 1.5
    assert bedrock.cost_usd("other", 1000) is None


def test_backoff_waits_out_throttling_but_not_other_errors():
    def error(code):
        return ClientError({"Error": {"Code": code, "Message": "x"}}, "Converse")

    calls, waits = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise error("ThrottlingException")
        return "ok"

    assert bedrock.with_backoff(flaky, base=1, sleep=waits.append) == "ok"
    assert waits == [1, 2]

    def denied():
        raise error("AccessDeniedException")

    with pytest.raises(ClientError):
        bedrock.with_backoff(denied, sleep=waits.append)
    assert waits == [1, 2]  # no retry on a real error
