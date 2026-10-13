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

Everything hosted is one vendor: d1 and the 24B share `LIQUID_API_KEY`; TabPFN
is Prior Labs' cloud API. The only model that runs on this machine is
EmbeddingGemma-2 (llama-server, LaunchAgent-owned on `127.0.0.1:8901`).

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
cd service && uv sync && uv run blend1-serve --port 8420
curl -X POST http://localhost:8420/v1/ask -H 'Content-Type: application/json' \
  -d '{"prompt":"write a save email for 4471","table":[…],"target":"churned"}'
```

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
