"""JSONL telemetry with structural redaction — the house pattern.

One row per ask(): lane telemetry, route decision, latencies, cost. Never
the prompt text, never the final answer, never key values — content stays
with the caller; the log carries sha256 + counts so rows stay joinable
(input_sha256 joins a feedback row back to its ask, same as sys1).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

DEFAULT_ROOT = (
    Path(os.environ["BLEND1_LOG_DIR"])
    if os.environ.get("BLEND1_LOG_DIR")
    else Path.home() / ".local" / "share" / "blend1" / "logs"
)

# Keys that must never reach the log even if a lane puts them in telemetry.
REDACT_KEYS = {"api_key", "authorization", "token", "prompt", "text", "final_answer", "messages"}


def _redact(obj):
    if isinstance(obj, dict):
        return {k: ("<redacted>" if k in REDACT_KEYS else _redact(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    return obj


def log_record(payload: dict) -> None:
    payload = _redact(payload)
    day = time.strftime("%Y/%m/%d")
    path = DEFAULT_ROOT / day / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")
