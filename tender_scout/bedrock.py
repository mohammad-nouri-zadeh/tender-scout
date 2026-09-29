"""Amazon Bedrock helpers: client, on-disk cache and cost from token usage.

Prices are not hard-coded: copy prices.example.json to prices.json and fill
in the current per-1,000-token prices for your region from
https://aws.amazon.com/bedrock/pricing/ . Without it, reports show tokens only.
"""

import hashlib
import json
import time
from pathlib import Path

from tender_scout.config import AWS_REGION, PRICES_PATH

THROTTLE_CODES = {"ThrottlingException", "ServiceUnavailableException", "ModelNotReadyException"}


def client(region: str = AWS_REGION):
    import boto3
    from botocore.config import Config

    cfg = Config(retries={"max_attempts": 10, "mode": "adaptive"}, read_timeout=120)
    return boto3.client("bedrock-runtime", region_name=region, config=cfg)


def with_backoff(call, attempts: int = 8, base: float = 5.0, cap: float = 120.0, sleep=time.sleep):
    """Run call(), waiting longer each time Bedrock still throttles after botocore's own retries.

    New accounts get low on-demand quotas (a 1,000-lot run died at lot 517 without this):
    a long run should slow down, not crash.
    """
    from botocore.exceptions import ClientError

    for attempt in range(attempts):
        try:
            return call()
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") not in THROTTLE_CODES or attempt == attempts - 1:
                raise
            sleep(min(cap, base * 2**attempt))


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class JsonlCache:
    """Append-only cache so an interrupted run resumes without paying twice."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.data[row["key"]] = row

    def get(self, key: str) -> dict | None:
        return self.data.get(key)

    def put(self, row: dict) -> None:
        self.data[row["key"]] = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def cost_usd(model_id: str, input_tokens: int, output_tokens: int = 0) -> float | None:
    if not PRICES_PATH.exists():
        return None
    prices = json.loads(PRICES_PATH.read_text()).get(model_id)
    if not prices or prices.get("input_per_1k") is None:
        return None
    out = prices.get("output_per_1k") or 0.0
    return round(input_tokens / 1000 * prices["input_per_1k"] + output_tokens / 1000 * out, 6)


def safe_name(model_id: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in model_id)
