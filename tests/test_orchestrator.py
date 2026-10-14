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


def _ok_semantic(prompt, *, retrieve=True, project=None):
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


# ── sys1-score stubs ──

def _score_correct(state):
    return {"status": "ok", "provider": "liquid", "is_correct": 0.93,
            "correctness": 5.0, "correct": True, "latency_ms": 180}


def _score_wrong(state):
    return {"status": "ok", "provider": "liquid", "is_correct": 0.21,
            "correctness": 1.0, "correct": False, "latency_ms": 180}


def _score_error(state):
    return {"status": "error", "error": "liquid provider unavailable", "provider": "liquid"}


# ── escalation-leg stubs ──

def _ok_escalation(prompt, *, tabular_facts="", semantic_facts="", web_facts="", feedback=""):
    return {"status": "ok", "model": "big-model-70b", "text": "escalated answer",
            "usage": {"total_tokens": 900}}


def _skipped_escalation(prompt, *, tabular_facts="", semantic_facts="", web_facts="", feedback=""):
    return {"status": "skipped",
            "reason": "escalation target is not configured: set BLEND1_ESCALATION_MODEL "
                      "and BLEND1_ESCALATION_BASE ..."}


def _err_escalation(prompt, *, tabular_facts="", semantic_facts="", web_facts="", feedback=""):
    return {"status": "error", "error": "escalation model HTTP 503: upstream unavailable",
            "model": "big-model-70b"}


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
                         "generate": _ok_generate, "validate": _gate_pass, "score": _score_correct},
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
                         "generate": _echo_generate, "validate": _gate_pass,
                         "score": _score_correct},
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

# The tabular stub rides here so every synthesized ask that passes a table
# stays offline — the orchestrator only submits the tabular lane when a
# table is given, so table-less tests are unaffected.
BASE_OV = {"route": _ok_route(0.99), "semantic": _ok_semantic, "tabular": _ok_tabular}


def test_gate_pass_ships_without_retry():
    calls: list[str] = []
    r = ask("write a save email", _lane_overrides={
        **BASE_OV, "generate": _tracking_generate(calls), "validate": _gate_pass,
        "score": _score_correct,
    })
    assert calls == [""], "a passing gate must not retry"
    assert r.validation["outcome"] == "passed"
    assert r.final_answer == "first attempt"
    assert not r.escalated


def test_gate_fail_retries_once_with_feedback():
    calls: list[str] = []
    r = ask("write a save email", _lane_overrides={
        **BASE_OV, "generate": _tracking_generate(calls), "validate": _gate_fail,
        "score": _score_correct, "escalation": _skipped_escalation,
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
        "score": _score_correct,
    })
    assert r.final_answer == "corrected attempt"
    assert r.validation["retried"] is True
    assert r.validation["passed"] is True


def test_inconclusive_ships_ungated_without_retry():
    calls: list[str] = []
    r = ask("write a save email", _lane_overrides={
        **BASE_OV, "generate": _tracking_generate(calls), "validate": _gate_inconclusive,
        "score": _score_correct, "escalation": _skipped_escalation,
    })
    assert calls == [""], "an inconclusive gate must not retry"
    assert r.validation["outcome"] == "inconclusive"
    assert r.validation["passed"] is None
    assert r.escalated, "the caller must know the answer was never cleared"


def test_gate_down_ships_ungated():
    calls: list[str] = []
    r = ask("write a save email", _lane_overrides={
        **BASE_OV, "generate": _tracking_generate(calls), "validate": _gate_error,
        "score": _score_correct,
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
            "score": _score_correct,
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
            "score": _score_correct,
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
            "score": _score_correct,
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


# ── escalation leg: a larger configured model answers what the gate could
#    not clear ──


def test_escalation_fires_on_inconclusive_and_ships_flagged():
    calls: list[str] = []
    seen: dict = {}

    def _esc(prompt, *, tabular_facts="", semantic_facts="", web_facts="", feedback=""):
        calls.append(feedback)
        seen.update(tabular_facts=tabular_facts, semantic_facts=semantic_facts,
                    web_facts=web_facts, feedback=feedback)
        return _ok_escalation(prompt, tabular_facts=tabular_facts,
                              semantic_facts=semantic_facts, web_facts=web_facts,
                              feedback=feedback)

    r = ask("write a save email", table=TABLE, target="churned", _lane_overrides={
        **BASE_OV, "generate": _tracking_generate([]), "validate": _gate_inconclusive,
        "score": _score_correct, "escalation": _esc,
    })
    # the escalated answer ships, flagged by the new key
    assert r.escalated_answer is True
    assert r.final_answer == "escalated answer"
    assert r.escalation["status"] == "ok" and r.escalation["shipped"] is True
    assert r.escalation["model"] == "big-model-70b"
    # the gate's unresolved verdict still rides in the result
    assert r.validation["outcome"] == "inconclusive" and r.validation["passed"] is None
    assert r.escalated, "the caller must know the answer was never cleared"
    # bounded: exactly one escalation attempt per ask
    assert len(calls) == 1
    # the same givens reach the larger model
    assert "0.842" in seen["tabular_facts"], "the exact TabPFN number rides along"
    assert "doc-1" in seen["semantic_facts"], "the semantic neighbors ride along"
    assert "could not reach a confident verdict" in seen["feedback"]
    assert "answers_the_question rated 0.63" in seen["feedback"]


def test_escalation_fires_when_retry_fails():
    calls: list[str] = []
    seen: dict = {}

    def _esc(prompt, *, tabular_facts="", semantic_facts="", web_facts="", feedback=""):
        seen["feedback"] = feedback
        return _ok_escalation(prompt, tabular_facts=tabular_facts,
                              semantic_facts=semantic_facts, web_facts=web_facts,
                              feedback=feedback)

    r = ask("write a save email", table=TABLE, target="churned", _lane_overrides={
        **BASE_OV, "generate": _tracking_generate(calls), "validate": _gate_fail,
        "score": _score_correct, "escalation": _esc,
    })
    assert len(calls) == 2, "the gate's retry discipline is unchanged"
    assert r.validation["retry_passed"] is False
    assert r.escalated_answer is True
    assert r.final_answer == "escalated answer"
    assert "drifted from the question" in seen["feedback"], "the gate's verdict rides as feedback"


def test_escalation_unconfigured_is_a_named_skip():
    r = ask("write a save email", table=TABLE, target="churned", _lane_overrides={
        **BASE_OV, "generate": _tracking_generate([]), "validate": _gate_inconclusive,
        "score": _score_correct, "escalation": _skipped_escalation,
    })
    assert r.escalation["status"] == "skipped"
    assert "BLEND1_ESCALATION_MODEL" in r.escalation["reason"]
    assert r.escalated_answer is False
    assert r.final_answer == "first attempt", "the original ships when the leg is unconfigured"


def test_failed_escalation_ships_the_original_with_the_failure_named():
    r = ask("write a save email", table=TABLE, target="churned", _lane_overrides={
        **BASE_OV, "generate": _tracking_generate([]), "validate": _gate_inconclusive,
        "score": _score_correct, "escalation": _err_escalation,
    })
    assert r.escalation["status"] == "error"
    assert "503" in r.escalation["error"], "the failure is named"
    assert r.escalated_answer is False
    assert r.final_answer == "first attempt", "the gate never blocks"
    assert r.escalated


def test_escalation_does_not_fire_on_a_pass():
    calls: list[int] = []

    def _esc(*a, **kw):
        calls.append(1)
        return _ok_escalation(*a, **kw)

    r = ask("write a save email", _lane_overrides={
        **BASE_OV, "generate": _tracking_generate([]),
        "validate": _gate_pass, "score": _score_correct, "escalation": _esc,
    })
    assert calls == [], "a cleared gate must not escalate"
    assert r.escalated_answer is False
    assert r.escalation == {}


def test_direct_path_runs_neither_new_leg():
    def _boom_score(state):
        raise AssertionError("no sys1 score on the direct path")

    def _boom_escalation(**kw):
        raise AssertionError("no escalation on the direct path")

    r = ask("what is the churn probability for 4471?", table=TABLE, target="churned",
            _lane_overrides={"route": _ok_route(0.02, "direct"), "semantic": _ok_semantic,
                             "tabular": _ok_tabular, "score": _boom_score,
                             "escalation": _boom_escalation})
    assert r.path == "direct"
    assert r.score == {} and r.escalation == {}
    assert r.escalated_answer is False


def test_escalation_target_resolution(monkeypatch):
    from blend1 import lanes

    monkeypatch.delenv("BLEND1_ESCALATION_KEY_ENV", raising=False)
    monkeypatch.setenv("BLEND1_ESCALATION_MODEL", "big-model-70b")
    monkeypatch.setenv("BLEND1_ESCALATION_BASE", "https://esc.example/v1")
    assert lanes._escalation_target() == {
        "name": "big-model-70b", "base": "https://esc.example/v1",
        "key_env": "BLEND1_ESCALATION_API_KEY",
    }
    monkeypatch.setenv("BLEND1_ESCALATION_KEY_ENV", "MY_ESC_KEY")
    assert lanes._escalation_target()["key_env"] == "MY_ESC_KEY"
    monkeypatch.delenv("BLEND1_ESCALATION_BASE")
    assert lanes._escalation_target() is None, "a model alone does not arm the leg"


def test_escalation_lane_unconfigured_names_the_env_vars(monkeypatch):
    from blend1 import lanes

    for var in ("BLEND1_ESCALATION_MODEL", "BLEND1_ESCALATION_BASE", "BLEND1_ESCALATION_KEY_ENV"):
        monkeypatch.delenv(var, raising=False)
    rep = lanes.escalation_lane("anything")
    assert rep["status"] == "skipped"
    assert "BLEND1_ESCALATION_MODEL" in rep["reason"]
    assert "BLEND1_ESCALATION_BASE" in rep["reason"]


def test_escalation_lane_missing_key_is_a_named_skip(monkeypatch):
    from blend1 import lanes

    monkeypatch.setenv("BLEND1_ESCALATION_MODEL", "big-model-70b")
    monkeypatch.setenv("BLEND1_ESCALATION_BASE", "https://esc.example/v1")
    monkeypatch.delenv("BLEND1_ESCALATION_API_KEY", raising=False)
    rep = lanes.escalation_lane("anything")
    assert rep["status"] == "skipped"
    assert "BLEND1_ESCALATION_API_KEY" in rep["reason"]


def _install_fake_escalation_transport(monkeypatch, payload):
    """Point the shared OpenAI-compatible wire at a fake transport; return
    the captured request view. Never the network."""
    import json

    from blend1 import lanes

    monkeypatch.setenv("BLEND1_ESCALATION_MODEL", "big-model-70b")
    monkeypatch.setenv("BLEND1_ESCALATION_BASE", "https://esc.example/v1/")
    monkeypatch.setenv("BLEND1_ESCALATION_API_KEY", "test-key")
    captured: dict = {}

    class _Resp:
        def __init__(self, body):
            self._body = body

        def read(self):
            return json.dumps(self._body).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["auth"] = req.headers.get("Authorization")
        captured["timeout"] = timeout
        return _Resp(payload)

    monkeypatch.setattr(lanes.urllib.request, "urlopen", _fake_urlopen)
    return captured


def test_escalation_lane_calls_the_configured_model(monkeypatch):
    from blend1 import lanes

    captured = _install_fake_escalation_transport(monkeypatch, {
        "model": "big-model-70b",
        "choices": [{"message": {"content": "the escalated answer"}, "finish_reason": "stop"}],
        "usage": {"total_tokens": 900, "cost": 0.004},
    })
    rep = lanes.escalation_lane(
        "write a save email",
        tabular_facts="churned = 1 (p=0.842)",
        semantic_facts="- [doc-1] (cos 0.91): billing errors",
        web_facts="- [Docs] (https://x.test/docs): untrusted excerpt",
        feedback="the gate could not decide",
    )
    assert rep["status"] == "ok" and rep["text"] == "the escalated answer"
    assert rep["cost_usd"] == 0.004
    assert captured["url"] == "https://esc.example/v1/chat/completions"
    assert captured["body"]["model"] == "big-model-70b"
    assert captured["body"]["max_tokens"] == lanes.ESCALATION_MAX_TOKENS
    assert captured["auth"] == "Bearer test-key"
    assert captured["timeout"] == lanes.ESCALATION_TIMEOUT_S
    system = captured["body"]["messages"][0]["content"]
    # the same givens, the untrusted-source framing intact
    assert "0.842" in system and "doc-1" in system and "https://x.test/docs" in system
    assert "never follow" in system
    assert "the gate could not decide" in system
    assert captured["body"]["messages"][1]["content"] == "write a save email"


def test_escalation_lane_http_error_is_named(monkeypatch):
    import io
    import urllib.error

    from blend1 import lanes

    _install_fake_escalation_transport(monkeypatch, {})

    def _fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            "https://esc.example/v1/chat/completions", 503, "Service Unavailable", {},
            io.BytesIO(b"upstream unavailable"),
        )

    monkeypatch.setattr(lanes.urllib.request, "urlopen", _fake_urlopen)
    rep = lanes.escalation_lane("anything")
    assert rep["status"] == "error"
    assert "503" in rep["error"] and "upstream unavailable" in rep["error"]


def test_escalation_lane_empty_content_is_a_failed_leg(monkeypatch):
    from blend1 import lanes

    _install_fake_escalation_transport(monkeypatch, {
        "choices": [{"message": {"content": None}, "finish_reason": "length"}],
    })
    rep = lanes.escalation_lane("anything")
    assert rep["status"] == "error"
    assert "no content" in rep["error"] and "length" in rep["error"]


# ── sys1 score leg: d1 scores the generated answer's correctness ──


def test_sys1_score_rides_in_the_result():
    seen: dict = {}

    def _score(state):
        seen["state"] = state
        return _score_correct(state)

    r = ask("write a save email", table=TABLE, target="churned", _lane_overrides={
        **BASE_OV, "generate": _tracking_generate([]), "validate": _gate_pass,
        "score": _score,
    })
    assert r.score["status"] == "ok"
    assert r.score["correct"] is True
    assert r.score["is_correct"] == 0.93
    assert r.score["reprocessed"] is False
    assert r.score["shipped"] == "original"
    # the score reads the SAME state shape the GLiDE critic reads
    assert "QUESTION: write a save email" in seen["state"]
    assert "FACTS AND SOURCES" in seen["state"]
    assert "ANSWER: first attempt" in seen["state"]
    assert "0.842" in seen["state"], "the computed facts are part of the evidence"


def test_wrong_score_reprocesses_and_the_corrected_answer_ships():
    calls: list[str] = []
    r = ask("write a save email", table=TABLE, target="churned", _lane_overrides={
        **BASE_OV, "generate": _tracking_generate(calls), "validate": _gate_pass,
        "score": _score_wrong,
    })
    assert len(calls) == 2, "exactly one reprocess"
    assert "correctness scorer" in calls[1], "the score's verdict rides back as feedback"
    assert "0.21" in calls[1]
    assert r.final_answer == "corrected attempt"
    assert r.score["reprocessed"] is True
    assert r.score["shipped"] == "reprocessed"
    assert r.score["correct"] is False, "the verdict that fired the reprocess still rides"


def test_right_score_does_not_reprocess():
    calls: list[str] = []
    r = ask("write a save email", table=TABLE, target="churned", _lane_overrides={
        **BASE_OV, "generate": _tracking_generate(calls), "validate": _gate_pass,
        "score": _score_correct,
    })
    assert len(calls) == 1, "a right score must not double-spend"
    assert r.final_answer == "first attempt"
    assert r.score["reprocessed"] is False


def test_unavailable_sys1_names_the_absence():
    def _err_route(prompt):
        return {"status": "error", "error": "liquid provider unavailable"}

    def _boom_score(state):
        raise AssertionError("the score lane must not run when the route errored")

    calls: list[str] = []
    r = ask("write a save email", table=TABLE, target="churned", _lane_overrides={
        "route": _err_route, "semantic": _ok_semantic, "tabular": _ok_tabular,
        "generate": _tracking_generate(calls), "validate": _gate_pass, "score": _boom_score,
    })
    assert r.score["status"] == "skipped"
    assert "sys1 unavailable" in r.score["reason"]
    assert len(calls) == 1, "no reprocess without a score"
    assert r.final_answer == "first attempt", "the answer ships unscored, never blocked"


def test_sys1_down_names_the_absence():
    calls: list[str] = []
    r = ask("write a save email", table=TABLE, target="churned", _lane_overrides={
        **BASE_OV, "generate": _tracking_generate(calls), "validate": _gate_pass,
        "score": _score_error,
    })
    assert r.score["status"] == "error"
    assert "liquid provider unavailable" in r.score["error"]
    assert len(calls) == 1, "no reprocess when the scorer could not answer"
    assert r.final_answer == "first attempt"
    assert r.escalated, "a leg that could not answer escalates"


def test_failed_reprocess_ships_the_scored_answer_with_the_failure_named():
    calls: list[str] = []

    def _err_on_feedback_generate(prompt, *, tabular_facts="", semantic_facts="",
                                  web_facts="", feedback=""):
        calls.append(feedback)
        if feedback:
            return {"status": "error", "error": "Liquid API unreachable: connection reset",
                    "model": "lfm2.5-24b-a2b"}
        return {"status": "ok", "model": "lfm2.5-24b-a2b", "text": "first attempt",
                "usage": {"total_tokens": 10}}

    r = ask("write a save email", table=TABLE, target="churned", _lane_overrides={
        **BASE_OV, "generate": _err_on_feedback_generate, "validate": _gate_pass,
        "score": _score_wrong,
    })
    assert len(calls) == 2
    assert r.final_answer == "first attempt", "the scored answer ships when the reprocess fails"
    assert r.score["reprocessed"] is True
    assert r.score["shipped"] == "original"
    assert "connection reset" in r.score["reprocess_error"]


def test_score_task_shape():
    from blend1.lanes import SCORE_FLOOR, _score_task

    task = _score_task()
    assert task.head_ids() == ["is_correct", "correctness"]
    assert task.find("is_correct").floor == SCORE_FLOOR
    assert task.find("correctness").score_max == 5


# ── contract + telemetry ──


def test_result_keys_grew_additively():
    from blend1.types import RESULT_KEYS

    for key in ("escalation", "score", "escalated_answer"):
        assert key in RESULT_KEYS
    r = ask("hello", _lane_overrides={"route": _ok_route(0.02, "direct"), "semantic": _ok_semantic})
    assert set(r.as_dict()) == set(RESULT_KEYS)


def test_telemetry_row_carries_the_new_keys_shapes(monkeypatch, tmp_path):
    import json
    import time

    from blend1 import telemetry

    monkeypatch.setattr(telemetry, "DEFAULT_ROOT", tmp_path)
    ask("write a save email", table=TABLE, target="churned", _lane_overrides={
        **BASE_OV, "generate": _tracking_generate([]), "validate": _gate_inconclusive,
        "score": _score_wrong, "escalation": _ok_escalation,
    })
    day = time.strftime("%Y/%m/%d")
    rows = [json.loads(line) for line in
            (tmp_path / day / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    assert len(rows) == 1
    row = rows[0]
    # the new keys' shapes ride in the row
    assert row["escalated_answer"] is True
    assert row["escalation"]["status"] == "ok"
    assert row["escalation"]["model"] == "big-model-70b"
    assert row["escalation"]["shipped"] is True
    assert row["score"]["status"] == "ok"
    assert row["score"]["reprocessed"] is True
    assert row["score"]["shipped"] == "reprocessed"
    # no prompt text, no answer text, no key material — anywhere in the row
    blob = json.dumps(row)
    assert "write a save email" not in blob
    assert "first attempt" not in blob
    assert "corrected attempt" not in blob
    assert "escalated answer" not in blob
    assert "test-key" not in blob and "Bearer" not in blob


def test_telemetry_redacts_text_and_keys():
    from blend1 import telemetry

    red = telemetry._redact({
        "text": "the answer", "api_key": "sk-live-xxx",
        "authorization": "Bearer sk-live-xxx",
        "nested": {"messages": [{"content": "secret"}], "keep": 1},
    })
    assert red["text"] == "<redacted>"
    assert red["api_key"] == "<redacted>"
    assert red["authorization"] == "<redacted>"
    assert red["nested"]["messages"] == "<redacted>"
    assert red["nested"]["keep"] == 1


def test_doctor_reports_the_new_legs(monkeypatch):
    import sys as sysmod

    from blend1 import lanes

    for var in ("BLEND1_ESCALATION_MODEL", "BLEND1_ESCALATION_BASE"):
        monkeypatch.delenv(var, raising=False)
    # keep the snapshot hermetic: no lane imports, no endpoint probes
    monkeypatch.setitem(sysmod.modules, "sem1", None)
    monkeypatch.setitem(sysmod.modules, "sdm1", None)
    report = lanes.lane_report()
    assert report["escalation"]["available"] is False
    assert "BLEND1_ESCALATION_MODEL" in report["escalation"]["reason"]
    assert report["score"]["provider"] == "liquid (d1) via sys1"
    assert report["score"]["reason"]

