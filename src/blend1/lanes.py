"""The five lanes. Each lane is one function that returns a lane report and
never raises: a lane that cannot answer fails open with its reason recorded
(house pattern — mirroring sys1's provider fail-open and sdm1's backend
fail-open). The orchestrator composes the answer from the lanes that did
answer.

Lanes:
    route     — sys1 + Liquid d1: does this ask need generative synthesis?
    semantic  — sem1 + local EmbeddingGemma-2 (llama-server :8901): prompt
                embedding + corpus neighbors
    tabular   — sdm1 + Prior Labs TabPFN (hosted): exact predictions
    generate  — Liquid LFM2-24B-A2B via OpenRouter: the only paid text leg
    voice     — gen1 speak (optional; the media render of the final answer)
"""

from __future__ import annotations

import contextlib
import json
import os
import urllib.error
import urllib.request

# ── route lane: sys1 + Liquid d1 ─────────────────────────────────────────────

ROUTE_TASK_ID = "blend_route"
ROUTE_FLOOR = 0.70  # needs_generation noul floor; below → direct path


def _route_task():
    from sys1.types import Task, make_choice, make_noul

    return Task(
        id=ROUTE_TASK_ID,
        description="Blend router: does this ask need generative synthesis?",
        heads=[
            make_noul(
                "Does answering this request require open-ended generative "
                "synthesis, beyond returning a fact, a computed metric, or a "
                "classification?",
                id="needs_generation",
                floor=ROUTE_FLOOR,
            ),
            make_choice(
                "What shape should the answer take?",
                labels=["direct", "synthesized", "decompose"],
                id="answer_shape",
                descriptions={
                    "direct": "A fact, metric, or classification another lane already computed.",
                    "synthesized": "Open-ended prose only the generative lane can produce.",
                    "decompose": "The request bundles several sub-asks; split before answering.",
                },
            ),
        ],
    )


def route_lane(prompt: str) -> dict:
    """Ask Liquid d1 (via sys1) whether the 24B leg is required."""
    try:
        import sys1
    except Exception as e:  # pragma: no cover - sys1 is a hard dep
        return {"status": "error", "error": f"sys1 import failed: {e}"}
    try:
        r = sys1.classify("liquid", _route_task(), prompt)
        ng = r.head("needs_generation", "liquid") or {}
        shape = r.head("answer_shape", "liquid") or {}
        if not r.providers_used:
            return {
                "status": "error",
                "error": r.reason or "liquid provider unavailable",
                "provider": "liquid",
            }
        return {
            "status": "ok",
            "provider": "liquid",
            "needs_generation": ng.get("noul"),
            "answer_shape": shape.get("label"),
            "shape_probabilities": shape.get("probabilities"),
            "verdict": r.verdict,
            "latency_ms": r.latency_ms,
            "telemetry": r.telemetry.get("liquid", {}),
        }
    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {e}", "provider": "liquid"}


# ── semantic lane: sem1 + local EmbeddingGemma-2 ─────────────────────────────


def semantic_lane(prompt: str, *, retrieve: bool = True) -> dict:
    """Embed the prompt; optionally retrieve corpus neighbors."""
    try:
        import sem1
    except Exception as e:  # pragma: no cover - sem1 is a hard dep
        return {"status": "error", "error": f"sem1 import failed: {e}"}
    try:
        # A list, always: sem1's contract is a list of items (a bare str is
        # now also accepted upstream, but explicit is the blend's style).
        res = sem1.embed([prompt])
        out = {
            "status": "ok",
            "provider": res.provider,
            "dims": res.dims,
            "latency_ms": res.telemetry.get("latency_ms"),
            "vector_head": [round(x, 4) for x in res.vectors[0][:8]],
        }
        if retrieve:
            from .corpus import retrieve as corpus_retrieve

            neighbors = corpus_retrieve(res.vectors[0])
            out["neighbors"] = neighbors
        return out
    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {e}", "provider": "llama-server"}


# ── tabular lane: sdm1 + Prior Labs TabPFN (hosted) ──────────────────────────


def tabular_lane(table: list[dict], *, target: str, task_type: str = "classification") -> dict:
    """Score the unlabeled rows of `table` with the tabular foundation model.

    sdm1 fails open by contract: an unconfigured backend comes back as
    verdict="error" with the missing-key reason, never as a wrong number.
    """
    try:
        import sdm1
    except Exception as e:  # pragma: no cover - sdm1 is a hard dep
        return {"status": "error", "error": f"sdm1 import failed: {e}"}
    try:
        r = sdm1.classify(table, target=target, task_type=task_type, model="tabpfn-hosted")
        if r.verdict == "error":
            return {
                "status": "error",
                "error": r.error or r.reason,
                "model": "tabpfn-hosted",
                "latency_ms": r.latency_ms,
            }
        return {
            "status": "ok",
            "model": "tabpfn-hosted",
            "verdict": r.verdict,
            "predictions": [p.as_dict() for p in r.predictions],
            "n_context_rows": r.n_context_rows,
            "n_query_rows": r.n_query_rows,
            "latency_ms": r.latency_ms,
            "telemetry": r.telemetry,
        }
    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {e}", "model": "tabpfn-hosted"}


# ── generate lane: Liquid LFM2.5-24B-A2B via Liquid's own API ────────────────

# OpenAI-compatible chat lane on Liquid's API (probed live 2026-10-13:
# GET /v1/chat/models lists lfm2.5-24b-a2b; POST /openai/v1/chat/completions
# answers — note /v1/chat/completions 404s, the openai prefix is the lane).
# Same LIQUID_API_KEY as the d1 decision lane: one vendor, one key.
GENERATE_BASE = "https://api.liquid.ai/openai/v1"
GENERATE_MODEL = "lfm2.5-24b-a2b"
GENERATE_TIMEOUT_S = 180
# lfm2.5 is a reasoning model: reasoning tokens consume the completion
# budget (a 60-token cap returned content=null, finish_reason=length), so
# the lane asks for room and treats an empty content as a failed leg.
GENERATE_MAX_TOKENS = 2048

S2_SYSTEM_TEMPLATE = (
    "You are the generative lane of a blended model. A tabular foundation "
    "model (TabPFN) and a semantic embedding index have already computed the "
    "facts below. Use them exactly as given; never invent, round, or alter "
    "numbers. If the facts are insufficient to answer, say so plainly.\n"
    "TABULAR FACTS:\n{tabular_facts}\n"
    "SEMANTIC CONTEXT (nearest corpus entries):\n{semantic_facts}"
)


def _liquid_chat(messages: list[dict], model: str, api_key: str, timeout: float) -> dict:
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": GENERATE_MAX_TOKENS,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{GENERATE_BASE}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def generate_lane(prompt: str, *, tabular_facts: str = "(none)", semantic_facts: str = "(none)") -> dict:
    """The one paid text leg: LFM2.5-24B-A2B on Liquid's own API.

    Never called on the direct path — the router gates this lane.
    """
    api_key = os.environ.get("LIQUID_API_KEY")
    if not api_key:
        return {
            "status": "error",
            "error": "generate lane is not configured: set LIQUID_API_KEY "
            "(the same key the d1 lane uses) in ~/.config/blend1/env",
            "model": GENERATE_MODEL,
        }
    messages = [
        {"role": "system", "content": S2_SYSTEM_TEMPLATE.format(
            tabular_facts=tabular_facts, semantic_facts=semantic_facts)},
        {"role": "user", "content": prompt},
    ]
    try:
        data = _liquid_chat(messages, GENERATE_MODEL, api_key, GENERATE_TIMEOUT_S)
    except urllib.error.HTTPError as e:
        body = ""
        with contextlib.suppress(Exception):
            body = e.read().decode("utf-8", "replace")[:200]
        if e.code in (401, 403):
            return {
                "status": "error",
                "error": f"Liquid API rejected the key (HTTP {e.code}): {body}",
                "model": GENERATE_MODEL,
            }
        return {"status": "error", "error": f"Liquid API HTTP {e.code}: {body}", "model": GENERATE_MODEL}
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        return {"status": "error", "error": f"Liquid API unreachable: {e}", "model": GENERATE_MODEL}

    choice = (data.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content") or ""
    usage = data.get("usage") or {}
    if not text:
        # A reasoning model that spent its whole budget thinking returns
        # content=null with finish_reason=length — that is a failed leg,
        # not an answer.
        return {
            "status": "error",
            "error": f"lfm2.5 returned no content (finish_reason="
            f"{choice.get('finish_reason')}, reasoning_tokens="
            f"{(usage.get('completion_tokens_details') or {}).get('reasoning_tokens')})",
            "model": GENERATE_MODEL,
            "usage": usage,
        }
    return {
        "status": "ok",
        "model": data.get("model") or GENERATE_MODEL,
        "text": text,
        "usage": usage,
        "cost_usd": usage.get("cost"),
        "tokens_per_second": (data.get("metrics") or {}).get("tokens_per_second"),
    }


# ── voice lane: gen1 speak (optional) ────────────────────────────────────────


def voice_lane(text: str) -> dict:
    """Render the final answer to audio via gen1's cascade. Opt-in: the
    caller asks for it, because generation is not free and not idempotent."""
    try:
        import gen1
    except Exception as e:
        return {
            "status": "error",
            "error": f"voice lane needs the gen1 extra (pip install 'blend1[voice]'): {e}",
        }
    try:
        result = gen1.speak(text)
        return {
            "status": "ok",
            "provider": result.provider,
            "duration_seconds": result.duration_seconds,
            "format": result.format,
            "audio_bytes": len(result.audio or b""),
        }
    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}


# ── doctor ───────────────────────────────────────────────────────────────────


def lane_report() -> dict:
    """Per-lane availability snapshot for `blend1 doctor` — reasons named."""
    report: dict[str, dict] = {}

    try:
        import sys1

        h = sys1.health_report().get("liquid", {})
        report["route"] = {
            "available": bool(h.get("available")),
            "reason": h.get("reason"),
            "provider": "liquid (d1) via sys1",
        }
    except Exception as e:
        report["route"] = {"available": False, "reason": f"sys1 health failed: {e}"}

    try:
        import sem1

        p = sem1.probe("llama-server")
        report["semantic"] = {
            "available": bool(p.get("reachable")),
            "reason": f"{p.get('dims')}d, {p.get('latency_ms')}ms" if p.get("reachable") else "endpoint unreachable",
            "provider": "llama-server (EmbeddingGemma-2)",
        }
    except Exception as e:
        report["semantic"] = {"available": False, "reason": f"sem1 probe failed: {e}"}

    try:
        import sdm1

        h = sdm1.health_report().get("tabpfn-hosted", {})
        report["tabular"] = {
            "available": bool(h.get("available")),
            "reason": h.get("reason"),
            "provider": "tabpfn-hosted (Prior Labs)",
        }
    except Exception as e:
        report["tabular"] = {"available": False, "reason": f"sdm1 health failed: {e}"}

    report["generate"] = {
        "available": bool(os.environ.get("LIQUID_API_KEY")),
        "reason": "LIQUID_API_KEY set (Liquid's own API — same key as the d1 lane)"
        if os.environ.get("LIQUID_API_KEY")
        else "missing LIQUID_API_KEY (set it in ~/.config/blend1/env)",
        "provider": f"{GENERATE_MODEL} via api.liquid.ai",
    }

    has_voice_keys = bool(os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("STEPFUN_API_KEY"))
    report["voice"] = {
        "available": has_voice_keys,
        "reason": "dashscope/stepfun key present" if has_voice_keys
        else "missing DASHSCOPE_API_KEY / STEPFUN_API_KEY (gen1 lane)",
        "provider": "gen1 speak cascade",
    }
    return report
