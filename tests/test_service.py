"""Offline tests: the service's OpenAI-compatible surface over ask().

The endpoint's one contract is that blend1's truth survives the dialect — the
answer rides in `choices[0].message.content`, the verdict in the non-standard
`blend1` field, and nothing (an empty answer, a failed gate, a missing bearer)
turns into a 500. `blend1.ask` is stubbed at the module attribute the endpoint
calls; the orchestrator's own lane-override seam is exercised in
tests/test_orchestrator.py.

No network, no keys, no service started — the FastAPI TestClient drives the
app in-process. Live behavior is proven by scripts/smoke_live.py, not here.

"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import blend1
from service import main as service_main
from service.main import app

URL = "/v1/chat/completions"
MESSAGES = [{"role": "user", "content": "What runs the blend1 offline suite?"}]


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _result(**overrides) -> blend1.BlendResult:
    """A synthesized ask that passed its gate, with a measured cost — the
    happy-path shape the endpoint lifts its `blend1` field from."""
    fields = {
        "prompt_sha256": "abc123def456",
        "prompt_chars": 34,
        "path": "synthesized",
        "route": {"status": "ok", "needs_generation": 0.93, "answer_shape": "synthesized",
                  "needs_live_facts": 0.02, "latency_ms": 640},
        "generated": {"status": "ok", "model": "lfm2.5-24b-a2b", "text": "Dear customer 4471, ...",
                      "cost_usd": 0.0016, "usage": {"total_tokens": 412}},
        "validation": {"status": "ok", "passed": True, "outcome": "passed", "provider": "glide",
                       "answers_the_question": 0.91, "contradicts_facts": 0.04,
                       "number_check": "ok"},
        "final_answer": "Dear customer 4471, ...",
        "total_latency_ms": 8123,
        "escalated_answer": False,
    }
    fields.update(overrides)
    return blend1.BlendResult(**fields)


def _install_ask(monkeypatch, result, seen: dict):
    """Stub the ask() the endpoint calls; record what it received."""

    def _fake_ask(prompt, **kwargs):
        seen["prompt"] = prompt
        seen["kwargs"] = kwargs
        return result

    monkeypatch.setattr(blend1, "ask", _fake_ask)


# ── the envelope ──


def test_chat_completions_answers_in_the_openai_envelope(client, monkeypatch):
    _install_ask(monkeypatch, _result(), {})
    r = client.post(URL, json={"model": "blend1", "messages": MESSAGES})

    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "blend1"
    assert body["id"].startswith("chatcmpl-blend1-")
    assert len(body["id"]) == len("chatcmpl-blend1-") + 8
    assert isinstance(body["created"], int)

    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["message"] == {"role": "assistant", "content": "Dear customer 4471, ..."}
    assert choice["finish_reason"] == "stop"


def test_the_blend1_field_carries_the_verdict(client, monkeypatch):
    _install_ask(monkeypatch, _result(), {})
    body = client.post(URL, json={"model": "blend1", "messages": MESSAGES}).json()

    b = body["blend1"]
    # the field's key set IS the contract — additive only, like RESULT_KEYS
    assert set(b) == {"path", "route", "validation", "escalated_answer",
                      "cost_usd", "latency_ms"}
    assert b["path"] == "synthesized"
    assert b["route"]["needs_generation"] == 0.93
    assert b["validation"]["outcome"] == "passed"
    assert b["escalated_answer"] is False
    assert b["cost_usd"] == 0.0016
    assert b["latency_ms"] == 8123


def test_an_unmeasured_cost_stays_null(client, monkeypatch):
    _install_ask(monkeypatch, _result(generated={"status": "ok", "cost_usd": None}), {})
    body = client.post(URL, json={"model": "blend1", "messages": MESSAGES}).json()

    # "unmeasured" is the display word; the wire keeps null — never $0
    assert body["blend1"]["cost_usd"] is None
    assert body["choices"][0]["message"]["content"] == "Dear customer 4471, ..."


def test_an_empty_answer_is_a_200_with_the_reason_riding(client, monkeypatch):
    _install_ask(monkeypatch, _result(
        final_answer="",
        validation={"status": "error",
                    "error": "glide provider unavailable — the answer ships ungated"},
    ), {})
    r = client.post(URL, json={"model": "blend1", "messages": MESSAGES})

    assert r.status_code == 200, "an empty answer is a completed ask, not an outage"
    body = r.json()
    assert body["choices"][0]["message"]["content"] == ""
    assert "glide provider unavailable" in body["blend1"]["validation"]["error"]


def test_a_failed_gate_ships_the_answer_with_its_flag(client, monkeypatch):
    _install_ask(monkeypatch, _result(validation={
        "status": "ok", "passed": False, "outcome": "failed", "provider": "glide",
        "answers_the_question": 0.21, "contradicts_facts": 0.62,
        "retried": True, "retry_passed": False,
    }), {})
    r = client.post(URL, json={"model": "blend1", "messages": MESSAGES})

    assert r.status_code == 200, "a flag is information; only an outage is an outage"
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "Dear customer 4471, ..."
    assert body["blend1"]["validation"]["passed"] is False
    assert body["blend1"]["validation"]["outcome"] == "failed"


def test_the_escalation_flag_rides_when_the_arbiter_answered(client, monkeypatch):
    _install_ask(monkeypatch, _result(
        final_answer="the escalated answer",
        validation={"status": "ok", "passed": None, "outcome": "inconclusive"},
        escalated_answer=True,
    ), {})
    body = client.post(URL, json={"model": "blend1", "messages": MESSAGES}).json()

    assert body["choices"][0]["message"]["content"] == "the escalated answer"
    assert body["blend1"]["escalated_answer"] is True


# ── the ask() call ──


def test_blend1_project_passes_through_and_the_default_is_the_services_own(client, monkeypatch):
    seen: dict = {}
    _install_ask(monkeypatch, _result(), seen)

    client.post(URL, json={"model": "blend1", "messages": MESSAGES, "blend1_project": "churn"})
    assert seen["kwargs"]["project"] == "churn"
    assert seen["kwargs"]["retrieve"] is True
    assert "live" not in seen["kwargs"], "the auto path decides live; the endpoint never forces it"

    client.post(URL, json={"model": "blend1", "messages": MESSAGES})
    assert seen["kwargs"]["project"] is None, "absent → the service's own resolution"


def test_messages_render_as_role_content_lines(client, monkeypatch):
    seen: dict = {}
    _install_ask(monkeypatch, _result(), seen)

    client.post(URL, json={"model": "blend1", "messages": [
        {"role": "system", "content": "You are blend1."},
        {"role": "user", "content": "What runs the offline suite?"},
        {"role": "assistant", "content": "pytest."},
        {"role": "user", "content": "And the lint gate?"},
    ]})

    assert seen["prompt"] == (
        "system: You are blend1.\n"
        "user: What runs the offline suite?\n"
        "assistant: pytest.\n"
        "user: And the lint gate?"
    )


# ── the bearer ──


def test_an_unset_token_leaves_the_service_localhost_open(client, monkeypatch):
    _install_ask(monkeypatch, _result(), {})
    monkeypatch.delenv("BLEND1_BEARER_TOKEN", raising=False)
    monkeypatch.setattr(service_main, "TOKEN", None)

    r = client.post(URL, json={"model": "blend1", "messages": MESSAGES})
    assert r.status_code == 200


def test_a_set_token_is_required(client, monkeypatch):
    _install_ask(monkeypatch, _result(), {})
    monkeypatch.setenv("BLEND1_BEARER_TOKEN", "s3cret-token")

    body = {"model": "blend1", "messages": MESSAGES}
    assert client.post(URL, json=body).status_code == 401, "no header → refused"
    wrong = client.post(URL, json=body, headers={"authorization": "Bearer nope"})
    assert wrong.status_code == 401, "a wrong token is refused, not ignored"
    right = client.post(URL, json=body, headers={"authorization": "Bearer s3cret-token"})
    assert right.status_code == 200


def test_the_module_level_token_binds_too(client, monkeypatch):
    """The import-time snapshot still guards: the live env is absent and the
    module's TOKEN is what the service started with."""
    _install_ask(monkeypatch, _result(), {})
    monkeypatch.delenv("BLEND1_BEARER_TOKEN", raising=False)
    monkeypatch.setattr(service_main, "TOKEN", "snapshot-token")

    body = {"model": "blend1", "messages": MESSAGES}
    ok = client.post(URL, json=body, headers={"authorization": "Bearer snapshot-token"})
    assert ok.status_code == 200
    assert client.post(URL, json=body).status_code == 401


# ── the rest of the surface is unchanged ──


def test_the_other_endpoints_still_answer(client, monkeypatch):
    """The new route is additive: /v1/ask and /v1/health keep their shapes."""
    _install_ask(monkeypatch, _result(), {})

    ask = client.post("/v1/ask", json={"prompt": "write a save email for 4471"})
    assert ask.status_code == 200
    assert ask.json()["path"] == "synthesized"
    assert ask.json()["final_answer"] == "Dear customer 4471, ..."

    health = client.get("/v1/health")
    assert health.status_code == 200
    assert "version" in health.json()
