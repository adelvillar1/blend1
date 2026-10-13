---
status: completed
created: 2026-10-13
updated: 2026-10-13
slug: blend1-foundation
---

# blend1 foundation — the blended lane

**Status: implemented + live-smoked 2026-10-13.** Offline suite 8/8 green, ruff
clean, live smoke exercises every lane that can answer today — and with the
TabPFN key wired, all four model lanes answer for real.

## Goal

One `ask()` that blends the house lanes into a single answer: Liquid d1
(sys1) decides whether the ask needs open-ended generative synthesis; the
local EmbeddingGemma-2 (sem1) and Prior Labs TabPFN (sdm1) lanes precompute
facts in parallel; Liquid LFM2-24B-A2B (OpenRouter) synthesizes only when the
router says so. Media rendering (gen1 speak) is the optional final leg.

## What this replaces

A drafted design (Gemini, 2026-10-13) that drew the right four-tier picture on
wrong wires: d1 via OpenRouter chat-completions, EmbeddingGemma via
SentenceTransformer, TabPFN via a local 4-row fit, and a nonexistent
`liquid/lfm-24b` id. This plan implements the same shape on the house's real,
verified contracts (table in README §"What the draft got wrong").

## Acceptance criteria

- [x] `ask(prompt)` runs route + semantic lanes concurrently; router gate on
      the d1 noul (`needs_generation`, floor 0.70) — direct asks never call
      the 24B leg.
- [x] `ask(prompt, table=..., target=...)` adds the tabular lane; TabPFN's
      exact prediction reaches the 24B system prompt as a given fact.
- [x] Every lane fails open with its reason named; `BlendResult.escalated`
      is True when any lane errored.
- [x] Semantic corpus: `seed_corpus([{key,text}])` → `retrieve` top-k by
      cosine over sem1's VectorStore (verified live: churn prompt retrieves
      churn-playbook at cos 0.771).
- [x] JSONL telemetry under ~/.local/share/blend1/logs — sha256 + counts,
      never prompt/answer text, never keys.
- [x] CLI (`ask` / `doctor` / `seed`), FastAPI service (`/v1/ask`,
      `/v1/health`), offline test suite (stub lanes), live smoke script.
- [x] Route lane live-verified: d1 returns 0.002 on "what is the churn
      probability…" (direct) and 0.978 on "write a save-strategy note…"
      (synthesized); ~200ms, ~$0.000004/call.

## Live state at ship time (2026-10-13, updated with the TabPFN key)

| Lane | State | Evidence |
|---|---|---|
| route (d1) | **live** | smoke: needs_generation 0.002 / 0.978 on the two probe prompts |
| semantic (EmbeddingGemma-2) | **live** | smoke: 768d, 60–110ms, correct neighbor ranking |
| tabular (TabPFN hosted) | **live** | full blend: real churn prediction p=0.924, 11.6s, key from `~/.config/sdm1/env` (extracted from sdm1/CLAUDE.local.md, "verified 2026-10-07") |
| generate (LFM2.5-24B-A2B) | **live** | full blend: synthesized answer carrying the exact TabPFN number + neighbors, $0.0016/ask, ~112 tok/s |
| voice (gen1) | opt-in | dashscope/stepfun keys present; off unless `voice=True` |

End-to-end cost per synthesized ask ≈ **$0.0016** (d1 $0.000004 +
TabPFN metered + 24B); direct asks cost **$0** in model spend.

## Deferred (needs a decision)

1. **Local 24B fallback** — LFM2-24B-A2B is open weights; a quantized GGUF on
   the existing llama.cpp build is viable on a machine with more than this
   one's 16 GB unified memory (Q4 ≈ 14 GB leaves no headroom here).
2. **Decompose path** — `answer_shape="decompose"` is recorded but not yet
   acted on (recursive split is a v0.2 decision).
3. **MRL truncation** — the draft's 128d truncation is unverified on this
   build; the lane keeps full 768d until probed.
4. **Calibration** — the router's 0.70 noul floor is a starting point; fit it
   from the JSONL stream + feedback rows like every other sys1 task.

## Cross-lane bug found and fixed upstream

`sem1.normalize_items` iterated a bare string into single-character items —
`embed("hello")` embedded 5 items and took 48× the server time, despite the
docstring promising str acceptance. Fixed in sem1 (str → one item) with a
regression test; blend1 passes explicit lists regardless.
