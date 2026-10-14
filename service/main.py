"""FastAPI wrapper — the same ask() over HTTP, any language.

    uvicorn service.main:app --port 8420
    curl -X POST http://localhost:8420/v1/ask -H 'Content-Type: application/json' \
        -d '{"prompt":"write a save email for 4471","table":[...],"target":"churned"}'
"""

from __future__ import annotations

import os

from fastapi import FastAPI
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
