"""FastAPI wrapper — the same ask() over HTTP, any language.

    uvicorn service.main:app --port 8420
    curl -X POST http://localhost:8420/v1/ask -H 'Content-Type: application/json' \
        -d '{"prompt":"write a save email for 4471","table":[...],"target":"churned"}'
    curl -X POST http://localhost:8420/v1/chat/completions -H 'Content-Type: application/json' \
        -d '{"model":"blend1","messages":[{"role":"user","content":"What runs the blend1 offline suite?"}]}'
    curl -X POST http://localhost:8420/v1/search -H 'Content-Type: application/json' \
        -d '{"project":"blend1","query":"churn playbook"}'

`POST /v1/chat/completions` is the OpenAI-compatible face of the same ask():
the messages render to one prompt (`role: content` lines, system messages
included and in order — the last user message is the question), and the answer
rides back in the standard envelope (`choices[0].message.content`) so any
OpenAI client works unchanged. blend1's own truth rides beside it in a
NON-STANDARD `blend1` field:

    {path, route, validation, escalated_answer, cost_usd, latency_ms}

— the shape d1 chose, its nouls, the gate's verdict, whether the escalation
arbiter answered, the measured generative cost (`null` when unmeasured —
never $0), and the ask's latency. A client that reads only `content` loses
nothing it had before; a client that wants the verdict gets all of it.

The endpoint never 500s and never retries or reroutes: ask() fails open, so an
empty answer is a 200 with content "" and the reason in the `blend1` field,
and a gate that did not pass is a 200 with the answer shipping flagged — a
flag is information, an outage is an outage. The only non-200 is a 401 on a
bad bearer, and only when BLEND1_BEARER_TOKEN is set.

"""

from __future__ import annotations

import os

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

app = FastAPI(title="blend1", version="0.1.0")

TOKEN = os.environ.get("BLEND1_BEARER_TOKEN")


class AskPayload(BaseModel):
    prompt: str
    table: list[dict] | None = None
    target: str | None = None
    task_type: str = "classification"
    voice: bool = False
    retrieve: bool = True
    live: bool | None = None


class SearchPayload(BaseModel):
    project: str | None = None
    query: str
    k: int = 3


class ChatPayload(BaseModel):
    """The OpenAI chat envelope plus blend1's one extension: which corpus
    project the ask resolves against. Absent → the service's own resolution
    (BLEND1_PROJECT > cwd repo basename > the legacy store), so the same
    prompt does not answer differently by accident of the start directory."""

    model: str = "blend1"
    messages: list[dict]
    blend1_project: str | None = None


def _render_prompt(messages: list[dict]) -> str:
    """messages → the ask prompt: one `role: content` line per message, system
    messages included and in order. The last user message is naturally the
    question; everything before it is context the ask reads the same way."""
    lines = []
    for m in messages:
        role = m.get("role") or "user"
        content = m.get("content")
        lines.append(f"{role}: {content if content is not None else ''}")
    return "\n".join(lines)


def _authorized(authorization: str | None) -> bool:
    """The service's optional bearer, the module-level TOKEN's own pattern:
    unset = localhost-open, set = required on every call. The live environment
    wins over the import-time snapshot, so a token configured after start (or
    by a test) still binds."""
    expected = os.environ.get("BLEND1_BEARER_TOKEN") or TOKEN
    if not expected:
        return True
    return authorization == f"Bearer {expected}"


@app.get("/v1/health")
def health():
    import blend1

    return {"version": blend1.__version__, "lanes": blend1.doctor()}


@app.post("/v1/ask")
def ask(payload: AskPayload):
    import blend1

    result = blend1.ask(
        payload.prompt,
        table=payload.table,
        target=payload.target,
        task_type=payload.task_type,
        voice=payload.voice,
        retrieve=payload.retrieve,
        live=payload.live,
    )
    return result.as_dict()


@app.post("/v1/chat/completions")
def chat_completions(payload: ChatPayload, authorization: str | None = Header(default=None)):
    import time
    import uuid

    import blend1

    if not _authorized(authorization):
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    # ask() never raises: a lane that could not answer rides its reason inside
    # the BlendResult, so the endpoint has no error branch of its own — an
    # empty answer ships as a 200 with the reason in the blend1 field.
    result = blend1.ask(
        _render_prompt(payload.messages),
        retrieve=True,
        project=payload.blend1_project,
    )
    return {
        "id": f"chatcmpl-blend1-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result.final_answer},
            "finish_reason": "stop",
        }],
        # blend1's truth, non-standard on purpose: the router's verdict and the
        # gate's ride in beside the answer, never instead of it. cost_usd is
        # the measured generative cost — null stays null (unmeasured), never $0.
        "blend1": {
            "path": result.path,
            "route": result.route,
            "validation": result.validation,
            "escalated_answer": result.escalated_answer,
            "cost_usd": result.generated.get("cost_usd"),
            "latency_ms": result.total_latency_ms,
        },
    }


@app.post("/v1/search")
def search(payload: SearchPayload):
    import time

    from sem1 import Sem1Error

    import blend1
    from blend1.corpus import resolve_project

    t0 = time.monotonic()
    try:
        results = blend1.search(payload.query, project=payload.project, k=payload.k)
    except ValueError as e:  # bad project name — a usage error
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Sem1Error as e:  # embedding server down — retryable, not a crash
        raise HTTPException(status_code=503, detail=str(e)) from e
    return {
        "project": resolve_project(payload.project),
        "query": payload.query,
        "results": results,
        "latency_ms": int((time.monotonic() - t0) * 1000),
    }
