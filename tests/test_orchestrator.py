"""Offline tests: the orchestrator's merge/routing/fail-open logic with stub
lanes (the house stub-provider pattern — no network, no keys).

Live lane behavior is proven by scripts/smoke_live.py, not here.
"""

from __future__ import annotations

import pytest

from blend1.orchestrator import ask


def _ok_route(ng: float, shape: str = "synthesized", needs_live_facts: float = 0.0):
    def _lane(prompt):
        return {"status": "ok", "needs_generation": ng, "answer_shape": shape,
                "needs_live_facts": needs_live_facts, "latency_ms": 10}

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


def _ok_generate(prompt, *, tabular_facts="", semantic_facts="", web_facts="", feedback=""):
    assert "0.842" in tabular_facts, "the S2 prompt must carry the exact TabPFN number"
    assert "doc-1" in semantic_facts, "the S2 prompt must carry the semantic neighbors"
    return {"status": "ok", "model": "lfm2.5-24b-a2b", "text": "Dear customer 4471, ...", "usage": {"total_tokens": 412}}


def _err_generate(prompt, *, tabular_facts="", semantic_facts="", web_facts=""):
    return {"status": "error", "error": "generate lane is not configured: set LIQUID_API_KEY (the same key the d1 lane uses) in ~/.config/blend1/env", "model": "lfm2.5-24b-a2b"}


def _echo_generate(prompt, *, tabular_facts="", semantic_facts="", web_facts="", feedback=""):
    return {"status": "ok", "model": "lfm2.5-24b-a2b", "text": "echo: " + prompt, "usage": {"total_tokens": 10}}


def _ok_web(prompt):
    return {
        "status": "ok",
        "provider": "tinyfish",
        "results": [{
            "url": "https://maplibre.org/docs",
            "title": "3D Terrain - MapLibre",
            "text": "terrain source accepts quantized-mesh; hillshade is the default",
        }],
        "n_results": 1,
        "latency_ms": 800,
    }


def _err_web(prompt):
    return {
        "status": "error",
        "error": "web lane is not configured: set TINYFISH_API_KEY in ~/.config/blend1/env",
    }


# ── validation-gate stubs ──

def _gate_pass(state, *, givens=""):
    return {"status": "ok", "passed": True, "outcome": "passed", "provider": "glide",
            "answers_the_question": 0.91, "contradicts_facts": 0.04, "number_check": "ok"}


def _gate_fail(state, *, givens=""):
    return {"status": "ok", "passed": False, "outcome": "failed", "provider": "glide",
            "answers_the_question": 0.21, "contradicts_facts": 0.62, "number_check": "ok"}


def _gate_inconclusive(state, *, givens=""):
    return {"status": "ok", "passed": None, "outcome": "inconclusive", "provider": "glide",
            "answers_the_question": 0.63, "contradicts_facts": 0.31,
            "escalation_error": "no escalation critic could answer"}


def _gate_error(state, *, givens=""):
    return {"status": "error", "error": "glide provider unavailable"}


def _tracking_generate(calls):
    def _gen(prompt, *, tabular_facts="", semantic_facts="", web_facts="", feedback=""):
        calls.append(feedback)
        return {"status": "ok", "model": "lfm2.5-24b-a2b",
                "text": "first attempt" if not feedback else "corrected attempt"}
    return _gen


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
        _lane_overrides={"route": _ok_route(0.99), "semantic": _ok_semantic, "tabular": _ok_tabular,
                         "generate": _ok_generate, "validate": _gate_pass},
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
        _lane_overrides={"route": _err_route, "semantic": _ok_semantic,
                         "generate": _echo_generate, "validate": _gate_pass},
    )
    assert r.path == "synthesized"  # quality over cost when the router is down
    assert r.escalated


def test_decompose_shape_recorded():
    r = ask(
        "compare churn across all segments and draft the board update",
        _lane_overrides={"route": _ok_route(0.95, "decompose"), "semantic": _ok_semantic,
                         "generate": _echo_generate, "validate": _gate_pass},
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


# ── validation gate ──

BASE_OV = {"route": _ok_route(0.99), "semantic": _ok_semantic}


def test_gate_pass_ships_without_retry():
    calls: list[str] = []
    r = ask("write a save email", _lane_overrides={
        **BASE_OV, "generate": _tracking_generate(calls), "validate": _gate_pass,
    })
    assert calls == [""], "a passing gate must not retry"
    assert r.validation["outcome"] == "passed"
    assert r.final_answer == "first attempt"
    assert not r.escalated


def test_gate_fail_retries_once_with_feedback():
    calls: list[str] = []
    r = ask("write a save email", _lane_overrides={
        **BASE_OV, "generate": _tracking_generate(calls), "validate": _gate_fail,
    })
    assert len(calls) == 2, "exactly one retry"
    assert "drifted from the question" in calls[1], "the feedback names the failed head"
    # the retry passes the gate too (stub validate always fails? no — the retry
    # is gated by the same stub, so the original ships flagged)
    assert r.validation["retried"] is True
    assert r.validation["retry_passed"] is False
    assert r.final_answer == "first attempt", "the original ships when the retry also fails"
    assert r.escalated


def test_gate_fail_retry_success_ships_corrected():
    calls: list[str] = []
    verdicts = iter([_gate_fail, _gate_pass])  # first fails, retry passes

    def _validate(state, *, givens=""):
        return next(verdicts)(state, givens=givens)

    r = ask("write a save email", _lane_overrides={
        **BASE_OV, "generate": _tracking_generate(calls), "validate": _validate,
    })
    assert r.final_answer == "corrected attempt"
    assert r.validation["retried"] is True
    assert r.validation["passed"] is True


def test_inconclusive_ships_ungated_without_retry():
    calls: list[str] = []
    r = ask("write a save email", _lane_overrides={
        **BASE_OV, "generate": _tracking_generate(calls), "validate": _gate_inconclusive,
    })
    assert calls == [""], "an inconclusive gate must not retry"
    assert r.validation["outcome"] == "inconclusive"
    assert r.validation["passed"] is None
    assert r.escalated, "the caller must know the answer was never cleared"


def test_gate_down_ships_ungated():
    calls: list[str] = []
    r = ask("write a save email", _lane_overrides={
        **BASE_OV, "generate": _tracking_generate(calls), "validate": _gate_error,
    })
    assert calls == [""]
    assert r.validation["status"] == "error"
    assert r.final_answer == "first attempt"
    assert r.escalated


def test_number_check_units():
    from blend1.lanes import metric_numbers, number_check

    # canonical form: percentages become probabilities
    assert metric_numbers("risk of 92.4% and spend of 150") == {"0.924", "150"}
    # list markers and small ordinals are formatting, not metrics
    assert metric_numbers("1. Reach out\n2. Check billing\n3. Offer a discount") == set()
    assert metric_numbers("churn within 60 days") == {"60"}

    # the fact renders as a probability; the answer writes a percentage —
    # same number, and the check must not flag it
    ok = number_check("Churn risk: 92.4%", "churned = 1 (p=0.924)")
    assert ok["ok"] and ok["flagged"] == []
    bad = number_check("Churn risk: 94.1%", "churned = 1 (p=0.924)")
    assert not bad["ok"] and bad["flagged"] == ["0.941"]


def test_escalate_lane_reports_missing_keys(monkeypatch):
    from blend1 import lanes

    for var in ("TOKEN_PLAN_API_KEY", "OLLAMA_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    esc = lanes.escalate_lane("QUESTION: x\nANSWER: y")
    assert esc["status"] == "error"
    assert "TOKEN_PLAN_API_KEY" in esc["error"]
    assert "OLLAMA_API_KEY" in esc["error"]

# ── web lane ──

def test_web_lane_injected_when_router_fires():
    captured: dict = {}
    givens_seen: dict = {}

    def _capture_generate(prompt, *, tabular_facts="", semantic_facts="", web_facts="", feedback=""):
        captured["web_facts"] = web_facts
        return {"status": "ok", "model": "lfm2.5-24b-a2b",
                "text": "The terrain API accepts a quantized-mesh source.", "usage": {"total_tokens": 10}}

    def _capture_validate(state, *, givens=""):
        givens_seen["givens"] = givens
        return {"status": "ok", "passed": True, "outcome": "passed", "provider": "glide",
                "answers_the_question": 0.91, "contradicts_facts": 0.04, "number_check": "ok"}

    r = ask(
        "What is the current MapLibre GL JS 3D terrain API?",
        _lane_overrides={
            "route": _ok_route(0.95, needs_live_facts=0.9),
            "semantic": _ok_semantic,
            "web": _ok_web,
            "generate": _capture_generate,
            "validate": _capture_validate,
        },
    )
    assert r.web["status"] == "ok" and r.web["n_results"] == 1
    assert "https://maplibre.org/docs" in captured["web_facts"], "the url reaches the S2 prompt"
    assert "quantized-mesh" in captured["web_facts"], "the excerpt reaches the S2 prompt"
    assert "https://maplibre.org/docs" in givens_seen["givens"], "the url rides in the validator givens"


def test_web_lane_not_invoked_when_head_low():
    calls: list[str] = []

    def _tracking_web(prompt):
        calls.append(prompt)
        return _ok_web(prompt)

    r = ask(
        "what is the churn probability for 4471?",
        _lane_overrides={
            "route": _ok_route(0.02, "direct", needs_live_facts=0.02),
            "semantic": _ok_semantic,
            "web": _tracking_web,
        },
    )
    assert calls == [], "web lane must not run when needs_live_facts is below the floor"
    assert r.web == {}


def test_web_lane_not_invoked_when_route_errors():
    calls: list[str] = []

    def _tracking_web(prompt):
        calls.append(prompt)
        return _ok_web(prompt)

    def _err_route(prompt):
        return {"status": "error", "error": "liquid provider unavailable"}

    r = ask(
        "What is the current MapLibre GL JS 3D terrain API?",
        _lane_overrides={"route": _err_route, "semantic": _ok_semantic,
                         "web": _tracking_web, "generate": _echo_generate, "validate": _gate_pass},
    )
    assert calls == [], "web lane must not run when the route errored"
    assert r.route["status"] == "error"
    assert r.web == {}


def test_live_override_forces_and_suppresses():
    calls: list[str] = []

    def _tracking_web(prompt):
        calls.append(prompt)
        return _ok_web(prompt)

    # live=True forces the lane on even with a low head
    r = ask(
        "what is the churn probability?",
        live=True,
        _lane_overrides={
            "route": _ok_route(0.02, "direct", needs_live_facts=0.02),
            "semantic": _ok_semantic, "web": _tracking_web,
        },
    )
    assert len(calls) == 1, "live=True forces the web lane regardless of the head"
    assert r.web["status"] == "ok"

    # live=False suppresses it even with a high head
    calls.clear()
    r2 = ask(
        "What is the current MapLibre GL JS 3D terrain API?",
        live=False,
        _lane_overrides={
            "route": _ok_route(0.95, needs_live_facts=0.9),
            "semantic": _ok_semantic, "web": _tracking_web,
            "generate": _echo_generate, "validate": _gate_pass,
        },
    )
    assert calls == [], "live=False suppresses the web lane regardless of the head"
    assert r2.web == {}


def test_number_check_allows_web_numbers():
    from blend1.lanes import number_check

    web_extract = "the terrain source accepts quantized-mesh tiles; the default is zoom 14"
    ok = number_check("The terrain API defaults to zoom 14 for hillshade", web_extract)
    assert ok["ok"], "a number present in the web extract must pass the check"
    bad = number_check("The terrain API defaults to zoom 17 for hillshade", web_extract)
    assert not bad["ok"] and "17" in bad["flagged"], "a number in no given is flagged"


def test_web_lane_error_fails_open():
    r = ask(
        "What is the current MapLibre GL JS 3D terrain API?",
        live=True,
        _lane_overrides={
            "route": _ok_route(0.95, needs_live_facts=0.9),
            "semantic": _ok_semantic, "web": _err_web,
            "generate": _echo_generate, "validate": _gate_pass,
        },
    )
    assert r.web["status"] == "error"
    assert "TINYFISH_API_KEY" in r.web["error"]
    assert r.escalated, "a web-lane error escalates"
    assert r.final_answer == "echo: What is the current MapLibre GL JS 3D terrain API?", (
        "the ask still answers when the web lane fails open"
    )


def test_route_task_has_three_heads():
    from blend1.lanes import _route_task

    task = _route_task()
    assert len(task.heads) == 3
    assert "needs_live_facts" in task.head_ids()


def test_web_lane_internals_with_fake_sdk(monkeypatch):
    """web_lane itself — the orchestrator tests above stub the lane, so this
    one exercises the lane's own logic against a fake tinyfish module
    (search→urls, highlights join, the beta fallback, all-failed, partial
    failure, missing key, missing extra). No network, ever."""
    import sys
    import types as pytypes

    from blend1 import lanes

    class PermissionDeniedError(Exception):
        pass

    class _Snip:
        def __init__(self, text):
            self.text = text

    class _SearchHit:
        def __init__(self, url):
            self.url = url

    class _SearchResp:
        def __init__(self, urls):
            self.results = [_SearchHit(u) for u in urls]

    class _FetchResult:
        def __init__(self, url, title, text=None, highlights=None):
            self.url, self.title = url, title
            self.text, self.highlights = text, highlights

    class _FetchErr:
        def __init__(self, url, error):
            self.url, self.error = url, error

    class _FetchResp:
        def __init__(self, results, errors):
            self.results, self.errors = results, errors

    def _install(fetch_impl):
        """fetch_impl(highlights_requested: bool) -> _FetchResp (may raise)."""
        mod = pytypes.ModuleType("tinyfish")
        mod.__dict__["PermissionDeniedError"] = PermissionDeniedError
        seen = {}

        class _Search:
            def query(self, q, *, purpose=None):
                seen["query"] = q
                return _SearchResp(["https://a.test/docs", "https://b.test/docs"])

        class _Fetch:
            def get_contents(self, urls, **kw):
                seen["highlights"] = "highlights" in kw
                return fetch_impl("highlights" in kw)

        class _Client:
            def __init__(self, api_key=None):
                seen["key"] = bool(api_key)

            search = _Search()
            fetch = _Fetch()

        mod.__dict__["TinyFish"] = _Client
        monkeypatch.setitem(sys.modules, "tinyfish", mod)
        return seen

    monkeypatch.setenv("TINYFISH_API_KEY", "test-key")

    # happy path: highlights are requested and joined per source
    seen = _install(lambda hl: _FetchResp(
        [_FetchResult("https://a.test/docs", "A", highlights=[_Snip("alpha 14"), _Snip("beta")]),
         _FetchResult("https://b.test/docs", "B", highlights=[_Snip("gamma")])], []))
    rep = lanes.web_lane("what is the a.test API?")
    assert rep["status"] == "ok" and rep["n_results"] == 2
    assert seen["highlights"] is True, "the lane requests highlights first"
    assert rep["results"][0]["text"] == "alpha 14\nbeta"
    assert rep["results"][0]["title"] == "A"
    assert rep["errors"] == []

    # beta denial: falls back to full text, capped to the excerpt budget
    def _fallback(hl):
        if hl:
            raise PermissionDeniedError("highlights is beta")
        return _FetchResp([_FetchResult("https://a.test/docs", None, text="x" * 5000)], [])

    _install(_fallback)
    rep = lanes.web_lane("what is the a.test API?")
    assert rep["status"] == "ok"
    assert len(rep["results"][0]["text"]) == lanes.WEB_EXCERPT_CHARS, "fallback text is capped"
    assert rep["results"][0]["title"] == "(untitled)", "a None title is named, not rendered"

    # every url failed → error naming the per-url reasons
    _install(lambda hl: _FetchResp(
        [], [_FetchErr("https://a.test/docs", "404"), _FetchErr("https://b.test/docs", "timeout")]))
    rep = lanes.web_lane("what is the a.test API?")
    assert rep["status"] == "error"
    assert "a.test/docs: 404" in rep["error"] and "b.test/docs: timeout" in rep["error"]

    # partial failure rides along — never silent
    _install(lambda hl: _FetchResp(
        [_FetchResult("https://a.test/docs", "A", highlights=[_Snip("alpha")])],
        [_FetchErr("https://b.test/docs", "timeout")]))
    rep = lanes.web_lane("what is the a.test API?")
    assert rep["status"] == "ok" and rep["n_results"] == 1
    assert rep["errors"] == [{"url": "https://b.test/docs", "error": "timeout"}]

    # missing key → error naming it (checked before any client is built)
    monkeypatch.delenv("TINYFISH_API_KEY")
    rep = lanes.web_lane("anything")
    assert rep["status"] == "error" and "TINYFISH_API_KEY" in rep["error"]

    # missing extra → error naming the extra (None in sys.modules makes the
    # import raise — the broad guard must catch it, not propagate)
    monkeypatch.setitem(sys.modules, "tinyfish", None)
    rep = lanes.web_lane("anything")
    assert rep["status"] == "error" and "tinyfish extra" in rep["error"]

