# blend1

> The blended lane of the sys1 house pattern. [sys1](https://github.com/adelvillar1/sys1)
> judges what the work says, [sdm1](https://github.com/adelvillar1/sdm1) scores what the work
> measures, [sem1](https://github.com/adelvillar1/sem1) indexes what the work looks like —
> **blend1 blends them into one answer.**

One `ask()` runs the Liquid d1 router, the local EmbeddingGemma-2 semantic lane, and the
Prior Labs TabPFN tabular lane **in parallel**, then pays for the 24B generative leg
(Liquid LFM2-24B-A2B) **only when the router says the ask needs open-ended synthesis**.
Everything else is answered from computed facts — no tokens spent on prose a metric could
have answered.

```python
import blend1

# Router says "direct" → no 24B call; the answer IS the TabPFN number.
r = blend1.ask("What is the churn probability for customer 4471?", table=rows, target="churned")
print(r.path)          # "direct"
print(r.final_answer)  # "- churned = 1 (p=0.842), confidence=0.842 ..."

# Router says "synthesized" → the 24B lane writes the email, with the exact
# TabPFN fact and the nearest corpus entries injected into its system prompt.
r = blend1.ask("Write a save-strategy email for customer 4471.", table=rows, target="churned")
print(r.path)          # "synthesized"
print(r.generated["model"])  # "liquid/lfm-2.5-24b-a2b"
```

## The five lanes

| Lane | Backend | Answers | Keys |
|---|---|---|---|
| `route` | Liquid **d1** via [sys1](https://github.com/adelvillar1/sys1) (`/decisions/v1/systemone`, noul + choice heads) | does this ask need generative synthesis? | `LIQUID_API_KEY` |
| `semantic` | **EmbeddingGemma-2** via [sem1](https://github.com/adelvillar1/sem1) (local llama-server, 768d) | prompt embedding + corpus neighbors | none (localhost) |
| `tabular` | **TabPFN** via [sdm1](https://github.com/adelvillar1/sdm1) (Prior Labs cloud API, fit-per-call) | exact predictions on tabular rows | `TABPFN_API_KEY` |
| `generate` | **LFM2.5-24B-A2B** via Liquid's own API (`/openai/v1/chat/completions`) | the synthesized prose | `LIQUID_API_KEY` (same key as d1) |
| `voice` | gen1 `speak` cascade (qwen → stepfun → kokoro) | audio render of the final answer | `DASHSCOPE_API_KEY` / `STEPFUN_API_KEY` |
| `validate` | **GLiDE** via sys1 (Jev-class decision model) | rates every synthesized answer; definitive fails get one retry, inconclusive escalates | `FASTINO_API_KEY` |
| `critics` | qwen3.8-flash (Token Plan) → glm-5.3-flash (ollama-cloud) | break inconclusive gates — the escalation tier | `TOKEN_PLAN_API_KEY` / `OLLAMA_API_KEY` |

Everything hosted is one vendor: d1 and the 24B share `LIQUID_API_KEY`; TabPFN
is Prior Labs' cloud API. The only model that runs on this machine is
EmbeddingGemma-2 (llama-server, LaunchAgent-owned on `127.0.0.1:8901`).

## The validation gate (v0.2)

Every synthesized answer is rated before it ships — by **GLiDE**, a decision
model from a different vendor than the d1 router, so the critic is not the
model that framed the task. Three outcomes:

- **passed** — both nouls clear their floors and the deterministic number
  check (percentages canonicalized to probabilities) finds no invented
  numbers. The answer ships. Cost: ~$0.00001.
- **failed** — a noul sits a full 0.15 band beyond its floor (clear drift or
  contradiction). **One retry** with the verdict injected as feedback; the
  retry is gated too, and a failed retry ships the original answer *flagged*,
  never blocked.
- **inconclusive** — near-floor nouls or mixed signals. **Escalates to the
  critic chain** (qwen3.8-flash → glm-5.3-flash); the critic's verdict
  decides. No critic reachable → the answer ships ungated with
  `escalated: true`. The gate never blocks.

## Operating rules (inherited from the house)

- **Facts before prose.** The tabular number reaches the 24B as a given, never as
  something to recompute — the system prompt forbids altering it. LLMs do not do math;
  TabPFN does.
- **Every lane fails open.** A lane that cannot answer records its reason
  (`BlendResult.escalated` is True) and the blend composes the best answer from the
  lanes that did answer. An unkeyed lane names its missing variable; it never
  silently degrades.
- **The router is a decision model, not a prompt.** d1 answers a calibrated noul
  ("needs open-ended synthesis?", floor 0.70) plus a shape choice
  (direct/synthesized/decompose) — the same head contract every other sys1 task uses.
- **Redacted telemetry.** One JSONL row per ask under
  `~/.local/share/blend1/logs/` — sha256 + counts + latencies, never prompt or
  answer text, never key values.

## Install

```bash
cd ~/Projects/blend1
uv sync                      # installs blend1 + sys1/sem1/sdm1 as editable path deps
uv run blend1 doctor         # per-lane availability, reasons named
```

Keys live in `~/.config/blend1/env` (chmod 600, never committed) — the gen1 key-file
precedent. `LIQUID_API_KEY` stays where sys1 already reads it; blend1 never touches it.

## CLI

```bash
uv run blend1 ask "What is the churn probability for 4471?" --table churn.csv --target churned
uv run blend1 ask "Write a save email for 4471." --table churn.csv --target churned --json
uv run blend1 ask "Read this back to me." --voice      # spends money, on purpose
uv run blend1 seed corpus.jsonl                        # semantic memory: [{"key","text"}...]
uv run blend1 doctor
```

## HTTP service

```bash
cd ~/Projects/blend1 && service/.venv/bin/uvicorn service.main:app --port 8420
curl -X POST http://localhost:8420/v1/ask -H 'Content-Type: application/json' \
  -d '{"prompt":"write a save email for 4471","table":[…],"target":"churned"}'
```

### The OpenAI-compatible face — `POST /v1/chat/completions`

The same `ask()` behind the dialect any OpenAI client already speaks: the
`messages` render to one prompt (`role: content` lines, system messages
included and in order — the last user message is the question) and the answer
comes back in the standard envelope, so a client that reads only
`choices[0].message.content` works unchanged.

```bash
curl -X POST http://localhost:8420/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"blend1","messages":[{"role":"user","content":"What runs the blend1 offline suite?"}]}'
```

```json
{
  "id": "chatcmpl-blend1-9f3ac21b",
  "object": "chat.completion",
  "created": 1760481234,
  "model": "blend1",
  "choices": [{"index": 0,
               "message": {"role": "assistant", "content": "uv run --extra dev pytest …"},
               "finish_reason": "stop"}],
  "blend1": {
    "path": "synthesized",
    "route": {"status": "ok", "needs_generation": 0.93, "answer_shape": "synthesized"},
    "validation": {"status": "ok", "passed": true, "outcome": "passed"},
    "escalated_answer": false,
    "cost_usd": 0.0016,
    "latency_ms": 8123
  }
}
```

**The `blend1` field is non-standard on purpose** — blend1's truth rides *beside*
the answer, never instead of it:

| key | what it is |
|---|---|
| `path` | the shape d1 chose: `direct` \| `synthesized` \| `decompose` |
| `route` | d1's lane report — the nouls, including `needs_generation` |
| `validation` | the GLiDE gate's verdict — `passed` is `false` (a failed gate) or `null` (inconclusive) when the answer was never cleared |
| `escalated_answer` | `true` when the larger configured model answered after the gate could not clear |
| `cost_usd` | the measured generative cost — **`null` when unmeasured, never `$0`** |
| `latency_ms` | the ask's end-to-end latency |

The endpoint **never 500s and never retries or reroutes**: `ask()` fails open, so
an empty answer is a `200` with content `""` and the reason in the `blend1`
field, and a gate that did not pass is a `200` with the answer shipping flagged —
a flag is information, an outage is an outage. The non-200s are a `401` on a bad
bearer (only when `BLEND1_BEARER_TOKEN` is set; unset = localhost-open) and a
`400` on an envelope that cannot form an ask — empty `messages`, or a non-string
content part — refused before any spend, because the blend's ask is one text
prompt.

Two optional extensions to the envelope: `"model"` (echoed back, defaults to
`"blend1"`) and `"blend1_project"` — which corpus project the ask resolves
against, so the same prompt does not answer differently by accident of the
directory the service started in.

This is the surface the [agnostic-router-kit](https://github.com/adelvillar1/agnostic-router-kit)
routes a `composed` tier to — the plan is `2026-10-14-composed-tier.md` in that
repo (`docs/plans/`).

## What Gemini's draft got wrong (and what this builds instead)

| Draft | Reality (verified live 2026-10-13) |
|---|---|
| `liquid/d1` called via OpenRouter chat-completions with a JSON schema | d1 speaks the TypeSafe System One API (`/decisions/v1/systemone`, heads); sys1 already wraps it with calibrated confidence |
| `SentenceTransformer("google/embeddinggemma-2")` | local llama-server `/v1/embeddings` (768d, ~50–700ms, **text-only by declaration** — the build tokenizes images as text and returns confident garbage) |
| `TabPFNClassifier(device='cpu')` fit on 4 toy rows | sdm1's hosted lane: upload CSVs → fit → predict per call, with fail-open errors and metered telemetry |
| `liquid/lfm-24b` | `lfm2.5-24b-a2b` on **Liquid's own API** (`POST /openai/v1/chat/completions`, same `LIQUID_API_KEY` as d1 — `/v1/chat/completions` 404s) |
| TabPFN as a local CPU fit | TabPFN as **Prior Labs' cloud API** (sdm1's hosted lane) — the only local model is EmbeddingGemma-2 |
| (absent) | capability probes, fail-open semantics, redacted JSONL telemetry, the voice lane — the house pattern |

## Tests

```bash
uv run pytest        # offline: merge/routing/fail-open with stub lanes
uv run python scripts/smoke_live.py   # live: every lane that can answer today
```

## License

Apache-2.0. See [LICENSE](LICENSE).
