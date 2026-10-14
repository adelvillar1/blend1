"""The BlendResult contract — one shape regardless of which lanes answered.

Contract rule (house pattern): result keys are only ever ADDED. Renaming or
removing a key below is a breaking change and needs a plan.

Every lane reports itself: `route`, `semantic`, `tabular`, `generated`,
`validation`, `voice`, `web`, `escalation`, `score` each carry their own
telemetry and their own error. A lane that
could not answer never breaks the blend — it fails open with its reason
recorded, and the orchestrator composes the best answer from the lanes that
did answer (mirroring sys1/sdm1 fail-open).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

RESULT_KEYS = (
    "prompt_sha256",
    "prompt_chars",
    "path",
    "route",
    "semantic",
    "tabular",
    "generated",
    "validation",
    "voice",
    "web",
    "escalation",
    "score",
    "final_answer",
    "total_latency_ms",
    "escalated_answer",
)


def _lane(status: str, **kw: Any) -> dict:
    return {"status": status, **kw}


@dataclass
class BlendResult:
    prompt_sha256: str
    prompt_chars: int
    path: str  # "direct" | "synthesized" | "decompose"
    route: dict = field(default_factory=dict)
    semantic: dict = field(default_factory=dict)
    tabular: dict = field(default_factory=dict)
    generated: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    voice: dict = field(default_factory=dict)
    web: dict = field(default_factory=dict)
    escalation: dict = field(default_factory=dict)
    score: dict = field(default_factory=dict)
    final_answer: str = ""
    total_latency_ms: int = 0
    escalated_answer: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_sha256": self.prompt_sha256,
            "prompt_chars": self.prompt_chars,
            "path": self.path,
            "route": dict(self.route),
            "semantic": dict(self.semantic),
            "tabular": dict(self.tabular),
            "generated": dict(self.generated),
            "validation": dict(self.validation),
            "voice": dict(self.voice),
            "web": dict(self.web),
            "escalation": dict(self.escalation),
            "score": dict(self.score),
            "final_answer": self.final_answer,
            "total_latency_ms": self.total_latency_ms,
            "escalated_answer": self.escalated_answer,
        }

    @property
    def escalated(self) -> bool:
        """True when any lane failed open — the caller should know the answer
        rests on fewer lanes than requested. A validation gate that FAILED
        the answer (even after retry) also escalates: the answer ships, but
        the caller deserves to know it was rated unsatisfactory.

        Note the meaning is exactly "a lane failed open": the escalation leg's
        SHIPPED answer is flagged by the separate `escalated_answer` key (the
        leg succeeding is not a lane failure)."""
        lanes_errored = any(
            lane.get("status") == "error"
            for lane in (self.route, self.semantic, self.tabular, self.generated,
                         self.validation, self.voice, self.web, self.escalation,
                         self.score)
            if lane
        )
        gate_failed = bool(
            self.validation
            and self.validation.get("status") == "ok"
            and not self.validation.get("passed", True)
        )
        return lanes_errored or gate_failed


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
