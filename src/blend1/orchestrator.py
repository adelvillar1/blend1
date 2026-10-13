"""The orchestrator: one ask(), five lanes, one answer.

Flow (all stdlib; lanes run concurrently — they are network-bound):

    prompt ──┬── route lane    (sys1 + Liquid d1: needs_generation?)
             ├── semantic lane (sem1 + EmbeddingGemma-2: embed + neighbors)
             └── tabular lane  (sdm1 + TabPFN: exact predictions)   [if table]
                                  │
                            merge facts
                                  │
              needs_generation? ─┼─ no  → direct answer (no 24B call)
                                  └─ yes → generate lane (LFM2-24B-A2B)
                                              │
                                        final answer (+ optional voice)

Every lane fails open: a lane that cannot answer records its reason and the
blend composes the best answer from the lanes that did answer. The result
names which lanes answered (BlendResult.escalated).
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from . import lanes, telemetry
from .types import BlendResult, prompt_sha256

DIRECT_SHAPE = "direct"


def _tabular_facts(tabular: dict, target: str) -> str:
    """Render tabular predictions as exact facts for the S2 system prompt."""
    if tabular.get("status") != "ok":
        return "(tabular lane unavailable)"
    lines = []
    for p in tabular.get("predictions", []):
        label = p.get("label")
        probs = p.get("probabilities") or {}
        conf = p.get("confidence")
        top = f"{label}"
        if probs:
            top += f" (p={max(probs.values()):.3f})" if probs else ""
        if conf is not None:
            top += f", confidence={conf:.3f}"
        lines.append(f"- {target} = {top}")
    n_ctx = tabular.get("n_context_rows")
    n_q = tabular.get("n_query_rows")
    lines.append(f"(TabPFN, {n_ctx} context rows, {n_q} query rows)")
    return "\n".join(lines) if lines else "(no predictions)"


def _semantic_facts(semantic: dict) -> str:
    if semantic.get("status") != "ok" or not semantic.get("neighbors"):
        return "(no semantic context)"
    return "\n".join(
        f"- [{n['key']}] (cos {n['score']}): {n['text']}" for n in semantic["neighbors"]
    )


def _validate_state(prompt: str, givens: str, answer: str) -> str:
    """The state the critic reads: the ask, the givens, the answer."""
    return (
        f"QUESTION: {prompt}\n"
        f"FACTS (computed by other lanes — the answer may use these and no others):\n{givens}\n"
        f"ANSWER: {answer}"
    )


def _retry_feedback(validation: dict) -> str:
    """Render the gate's verdict as feedback for the retry's system prompt."""
    problems = []
    atq = validation.get("answers_the_question")
    if atq is not None and atq < lanes.VALIDATE_FLOOR:
        problems.append(
            f"your answer drifted from the question that was asked "
            f"(rated {atq:.2f} on addressing it) — answer the actual ask"
        )
    contra = validation.get("contradicts_facts")
    if contra is not None and contra >= (1.0 - lanes.VALIDATE_FLOOR):
        problems.append(
            f"your answer contradicted the given facts "
            f"(rated {contra:.2f} on contradicting them) — restate them exactly"
        )
    if validation.get("number_check") == "flagged":
        problems.append(
            "your answer cited numbers that are not in the given facts: "
            + ", ".join(validation.get("flagged_numbers") or [])
            + " — use only the facts as given"
        )
    if not problems:
        problems.append("the previous attempt was rated unsatisfactory")
    return "A validation gate rejected your previous answer: " + "; ".join(problems) + "."


def _fallback_answer(
    route: dict, semantic: dict, tabular: dict, target: str | None, generated: dict
) -> str:
    """The no-prose path: compose from computed facts only. Covers both the
    router's direct choice AND a synthesized ask whose generative lane
    failed open — the facts are still exact, the prose is simply absent."""
    parts = []
    if generated.get("status") == "error":
        parts.append(f"Generative lane unavailable: {generated.get('error')}")
    if tabular.get("status") == "ok":
        parts.append(_tabular_facts(tabular, target or "target"))
    elif tabular:
        parts.append(f"Tabular lane unavailable: {tabular.get('error')}")
    if semantic.get("status") == "ok" and semantic.get("neighbors"):
        parts.append("Closest context: " + "; ".join(
            f"{n['key']} (cos {n['score']})" for n in semantic["neighbors"]
        ))
    if not parts:
        parts.append("No lane produced facts; the router chose the direct path.")
    return "\n".join(parts)


def ask(
    prompt: str,
    *,
    table: list[dict] | None = None,
    target: str | None = None,
    task_type: str = "classification",
    voice: bool = False,
    retrieve: bool = True,
    log: bool = True,
    _lane_overrides: dict | None = None,
) -> BlendResult:
    """One blended ask.

    Args:
        prompt: the user's text.
        table: optional tabular rows (dicts). Rows with a non-null `target`
            are labeled context; rows without it are the query rows sdm1
            scores. When absent, the tabular lane is skipped.
        target: the column TabPFN predicts (required when `table` is given).
        task_type: sdm1 task type ("classification" | "regression").
        voice: also render the final answer with gen1 speak (spends money).
        retrieve: retrieve semantic corpus neighbors (default True).
        log: append the redacted JSONL row (default True).
        _lane_overrides: test seam — {lane_name: callable} replaces a lane.
    """
    t0 = time.monotonic()
    ov = _lane_overrides or {}
    route_fn = ov.get("route", lanes.route_lane)
    sem_fn = ov.get("semantic", lanes.semantic_lane)
    tab_fn = ov.get("tabular", lanes.tabular_lane)
    gen_fn = ov.get("generate", lanes.generate_lane)
    voice_fn = ov.get("voice", lanes.voice_lane)

    # ── parallel lanes ──
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            "route": pool.submit(route_fn, prompt),
            "semantic": pool.submit(sem_fn, prompt, retrieve=retrieve),
        }
        if table is not None:
            if not target:
                raise ValueError("target is required when a table is given")
            futures["tabular"] = pool.submit(tab_fn, table, target=target, task_type=task_type)
        reports = {name: fut.result() for name, fut in futures.items()}

    route, semantic = reports["route"], reports["semantic"]
    tabular = reports.get("tabular", {})

    # ── merge + route ──
    ng = route.get("needs_generation") if route.get("status") == "ok" else None
    shape = route.get("answer_shape") if route.get("status") == "ok" else None
    # Router unavailable → fail toward synthesis (quality over cost), the
    # error rides in the result either way.
    needs_generation = True if ng is None else ng >= lanes.ROUTE_FLOOR
    path = "decompose" if shape == "decompose" else ("synthesized" if needs_generation else DIRECT_SHAPE)

    generated: dict = {}
    validation: dict = {}
    if needs_generation:
        # The givens the number check allows = the prompt + the computed
        # facts. The prompt matters: an asker's own numbers (customer ids,
        # dates) are legitimately quotable — the check exists to catch
        # numbers invented by the MODEL, not numbers the human supplied.
        givens = prompt + "\n" + _tabular_facts(tabular, target or "target") + "\n" + _semantic_facts(semantic)
        generated = gen_fn(prompt, tabular_facts=_tabular_facts(tabular, target or "target"),
                           semantic_facts=_semantic_facts(semantic))

        # ── validation gate: GLiDE rates the answer before it ships ──
        if generated.get("status") == "ok":
            validate_fn = ov.get("validate", lanes.validate_lane)
            state = _validate_state(prompt, givens, generated["text"])
            validation = validate_fn(state, givens=givens)

            if validation.get("status") == "ok" and validation.get("passed") is False:
                # One bounded retry: the verdict rides back into the generate
                # lane as feedback. Worst case ≈ 2x generate cost, failures only.
                # NOTE: `passed is False` — an inconclusive outcome (passed
                # None, no critic answered) ships ungated, it does not retry.
                feedback = _retry_feedback(validation)
                retry = gen_fn(prompt, tabular_facts=_tabular_facts(tabular, target or "target"),
                               semantic_facts=_semantic_facts(semantic), feedback=feedback)
                retry_validation = {}
                if retry.get("status") == "ok":
                    retry_validation = validate_fn(
                        _validate_state(prompt, givens, retry["text"]), givens=givens
                    )
                if retry.get("status") == "ok" and retry_validation.get("passed"):
                    generated = retry
                    validation = {**retry_validation, "retried": True}
                else:
                    # The retry didn't clear the gate either: the ORIGINAL
                    # answer ships, flagged — the gate never blocks.
                    validation = {
                        **validation,
                        "retried": True,
                        "retry_passed": False,
                        "retry_error": retry.get("error"),
                    }
            if validation.get("status") != "ok":
                # A down critic ships the answer ungated (fail-open) — the
                # reason rides in the validation report, not the answer.
                validation = {"status": "error", "error": validation.get("error")}

    if needs_generation and generated.get("status") == "ok":
        final_answer = generated["text"]
    else:
        # Direct path — or the generative lane failed open: compose from facts.
        final_answer = _fallback_answer(route, semantic, tabular, target, generated)

    voice_report: dict = {}
    if voice:
        voice_report = voice_fn(final_answer)

    result = BlendResult(
        prompt_sha256=prompt_sha256(prompt),
        prompt_chars=len(prompt),
        path=path,
        route=route,
        semantic=semantic,
        tabular=tabular,
        generated=generated,
        validation=validation,
        voice=voice_report,
        final_answer=final_answer,
        total_latency_ms=int((time.monotonic() - t0) * 1000),
    )

    if log:
        telemetry.log_record({
            "op": "ask",
            "prompt_sha256": result.prompt_sha256,
            "prompt_chars": result.prompt_chars,
            "path": result.path,
            "route": {k: v for k, v in route.items() if k != "telemetry"},
            "semantic": {k: v for k, v in semantic.items() if k not in ("vector_head", "neighbors")},
            "semantic_neighbor_keys": [n["key"] for n in semantic.get("neighbors", [])],
            "tabular": {k: v for k, v in tabular.items() if k not in ("predictions", "telemetry")},
            "generated": {k: v for k, v in generated.items() if k != "text"},
            "validation": {k: v for k, v in validation.items() if k != "telemetry"},
            "voice": voice_report,
            "answer_sha256": prompt_sha256(final_answer),
            "answer_chars": len(final_answer),
            "total_latency_ms": result.total_latency_ms,
            "escalated": result.escalated,
        })
    return result
