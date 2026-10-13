"""Offline tests: the orchestrator's merge/routing/fail-open logic with stub
lanes (the house stub-provider pattern — no network, no keys).

Live lane behavior is proven by scripts/smoke_live.py, not here.
"""

from __future__ import annotations

import pytest

from blend1.orchestrator import ask


def _ok_route(ng: float, shape: str = "synthesized"):
    def _lane(prompt):
        return {"status": "ok", "needs_generation": ng, "answer_shape": shape, "latency_ms": 10}

    return _lane


def _ok_semantic(prompt, *, retrieve=True):
    return {
        "status": "ok",
        "dims": 768,
        "latency_ms": 5,
        "neighbors": [{"key": "doc-1", "score": 0.91, "text": "billing errors cluster around invoice dates"}] if retrieve else [],
    }


def _ok_tabular(table, *, target, task_type="classification"):
    return {
        "status": "ok",
        "model": "tabpfn-hosted",
        "verdict": "pass",
        "predictions": [{"row_index": 0, "label": "1", "probabilities": {"0": 0.158, "1": 0.842}, "confidence": 0.842}],
        "n_context_rows": 6,
        "n_query_rows": 1,
        "latency_ms": 100,
    }


def _err_tabular(table, *, target, task_type="classification"):
    return {"status": "error", "error": "tabpfn-hosted backend is not configured: set TABPFN_API_KEY", "model": "tabpfn-hosted"}


def _ok_generate(prompt, *, tabular_facts="", semantic_facts=""):
    assert "0.842" in tabular_facts, "the S2 prompt must carry the exact TabPFN number"
    assert "doc-1" in semantic_facts, "the S2 prompt must carry the semantic neighbors"
    return {"status": "ok", "model": "liquid/lfm-2.5-24b-a2b", "text": "Dear customer 4471, ...", "usage": {"total_tokens": 412}}


def _err_generate(prompt, *, tabular_facts="", semantic_facts=""):
    return {"status": "error", "error": "generate lane is not configured: set LIQUID_API_KEY (the same key the d1 lane uses) in ~/.config/blend1/env", "model": "lfm2.5-24b-a2b"}


def _echo_generate(prompt, *, tabular_facts="", semantic_facts=""):
    return {"status": "ok", "model": "liquid/lfm-2.5-24b-a2b", "text": "echo: " + prompt, "usage": {"total_tokens": 10}}


TABLE = [
    {"logins": 2, "tickets": 5, "churned": 1},
    {"logins": 10, "tickets": 0, "churned": 0},
    {"logins": 1, "tickets": 6, "churned": None},
]


def test_synthesized_path_calls_generator_with_merged_facts():
    r = ask(
        "write a save email for 4471",
        table=TABLE,
        target="churned",
        _lane_overrides={"route": _ok_route(0.99), "semantic": _ok_semantic, "tabular": _ok_tabular, "generate": _ok_generate},
    )
    assert r.path == "synthesized"
    assert r.generated["status"] == "ok"
    assert r.final_answer == "Dear customer 4471, ..."
    assert not r.escalated


def test_direct_path_skips_generator():
    def _boom(prompt, **kw):
        raise AssertionError("generator must not run on the direct path")

    r = ask(
        "what is the churn probability for 4471?",
        table=TABLE,
        target="churned",
        _lane_overrides={"route": _ok_route(0.02, "direct"), "semantic": _ok_semantic, "tabular": _ok_tabular, "generate": _boom},
    )
    assert r.path == "direct"
    assert r.generated == {}
    assert "0.842" in r.final_answer  # the exact TabPFN fact, no LLM in the loop


def test_generator_failure_fails_open_to_facts():
    r = ask(
        "write a save email",
        table=TABLE,
        target="churned",
        _lane_overrides={"route": _ok_route(0.99), "semantic": _ok_semantic, "tabular": _ok_tabular, "generate": _err_generate},
    )
    assert r.path == "synthesized"
    assert r.generated["status"] == "error"
    assert "OPENROUTER_API_KEY" in r.final_answer or "0.842" in r.final_answer
    assert r.escalated


def test_tabular_failure_fails_open():
    r = ask(
        "what is the churn probability?",
        table=TABLE,
        target="churned",
        _lane_overrides={"route": _ok_route(0.02, "direct"), "semantic": _ok_semantic, "tabular": _err_tabular, "generate": _err_generate},
    )
    assert r.tabular["status"] == "error"
    assert "TABPFN_API_KEY" in r.final_answer
    assert r.escalated


def test_router_unavailable_fails_toward_synthesis():
    def _err_route(prompt):
        return {"status": "error", "error": "liquid provider unavailable"}

    r = ask(
        "hello?",
        _lane_overrides={"route": _err_route, "semantic": _ok_semantic, "generate": _echo_generate},
    )
    assert r.path == "synthesized"  # quality over cost when the router is down
    assert r.escalated


def test_decompose_shape_recorded():
    r = ask(
        "compare churn across all segments and draft the board update",
        _lane_overrides={"route": _ok_route(0.95, "decompose"), "semantic": _ok_semantic, "generate": _echo_generate},
    )
    assert r.path == "decompose"


def test_result_shape_is_json_safe():
    r = ask(
        "hello",
        _lane_overrides={"route": _ok_route(0.02, "direct"), "semantic": _ok_semantic},
    )
    d = r.as_dict()
    assert isinstance(d, dict)
    assert d["prompt_chars"] == 5
    assert d["path"] == "direct"


def test_target_required_with_table():
    with pytest.raises(ValueError):
        ask("hello", table=TABLE, _lane_overrides={"route": _ok_route(0.02)})
