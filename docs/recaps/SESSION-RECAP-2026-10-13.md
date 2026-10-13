# Session Recap — 2026-10-13

## Summary

Explored and built **blend1** (`~/Projects/blend1`), a fifth lane of the sys1 house
pattern that blends the four existing providers into one `ask()`: Liquid d1 (sys1)
decides whether an ask needs generative synthesis, local EmbeddingGemma-2 (sem1) and
Prior Labs TabPFN (sdm1, cloud) precompute facts in parallel, and LFM2.5-24B-A2B
(Liquid's own API, same `LIQUID_API_KEY` as d1) synthesizes only when the router
says so. Gemini's four-tier draft was implemented on the house's real, verified
contracts — every wire in the draft was wrong in some way. All four model lanes are
live-verified end-to-end; the full blend returned a real TabPFN churn prediction
(p=0.924) synthesized by the 24B with retrieved corpus context for ≈$0.0016/ask.

## Plans worked on

- **`docs/plans/2026-10-13-blend1-foundation.md`** — status: **completed**. All
  acceptance criteria met (parallel lanes + router gate; tabular facts as givens;
  fail-open per lane with `escalated`; corpus seed/retrieve; redacted JSONL; CLI +
  service + tests + smoke; live route verification).
- **`docs/plans/2026-10-13-blend1-validation-gate.md`** — status: **completed**
  (v0.2). GLiDE validator + two-critic escalation tier + one bounded retry.

## Commits

| Repo | Hash | Message |
|------|------|---------|
| blend1 | `34b4126` | blend1 v0.1.0: the blended lane — d1 router + EmbeddingGemma-2 + TabPFN + LFM2.5-24B |
| blend1 | (pending) | methodology artifacts + validation gate (v0.2) — this session's doc + gate layer |
| sem1 | `b7bd58b` | fix: a bare string is one text item, not an iterable of characters |

## What was added

- `src/blend1/` — orchestrator (parallel lanes + route gate + merge), five lane
  callables, corpus store, BlendResult contract, env loader, redacted telemetry, CLI.
- `service/` — FastAPI wrapper (`/v1/ask`, `/v1/health`, port 8420).
- `tests/test_orchestrator.py` — 15 offline tests (stub lanes).
- `scripts/smoke_live.py` — live smoke (doctor + seed + three asks).
- **v0.2 validation gate** — `validate_lane` (GLiDE task `blend_validate`,
  d1 on oversized states), `escalate_lane` (qwen3.8-flash → glm-5.3-flash critic
  chain from the Hermes agent config), deterministic `number_check`, one
  bounded retry with verdict-as-feedback, `validation` field on BlendResult.
- Methodology artifacts: `CLAUDE.md`, `CLAUDE.local.md` (gitignored),
  `TECHNICAL-DOCUMENTATION.md`, `FUNCTIONAL-SPECIFICATIONS.md`, this recap,
  plan docs with frontmatter, full `.gitignore`.

## What was fixed

- **sem1 (upstream, committed):** `normalize_items` iterated a bare string into
  single-character items — `embed("hello")` sent 5 items (48× server work on real
  prompts, wrong counts) despite the docstring promising str acceptance. Root cause:
  no str special-case before the `for it in items` loop. Fixed + regression test;
  sem1 suite 20/20 green.
- **blend1 corpus:** `seed()` stored under the server's full model path while
  `retrieve()` looked up the short slug — the store was never found (neighbors
  always empty). Fixed: store under the fixed slug; the real model id rides in the
  manifest.
- **blend1 generate lane:** rewired from OpenRouter to Liquid's own API
  (`/openai/v1/chat/completions`, 404 on `/v1/chat/completions`), one key for both
  Liquid lanes; reasoning-model handling (empty content = failed leg); cost +
  tok/s telemetry; output cap raised to 16K per user (128K context window).

## Files changed

- New repo: blend1 (16 files, 1581 insertions) + methodology artifacts.
- sem1: `src/sem1/types.py`, `tests/test_sem1.py`.
- Config: `~/.config/sdm1/env` created (chmod 600) from the key documented in
  `~/Projects/sdm1/CLAUDE.local.md` — its canonical home remains that file.

## Doc updates applied

- README lane table + "what the draft got wrong" table corrected to the verified
  Liquid-API reality; plan live-state table updated (all four model lanes live).

## Doc updates deferred

- None. CLAUDE.md "Today's state" carries the live-verified state.

## Open questions / next steps

1. **Gate floor fitting** — the 0.70 noul floors are UNFITTED house defaults.
   The machinery ships (verdicts + raw nouls in the JSONL row); fitting waits
   for a feedback stream. **User endorsement 2026-10-13: blend1 is a keeper —
   the ~15s full-blend duration is not a concern; accuracy and savings
   justify it.** That settles the latency-vs-completeness tradeoff in favor
   of the complete gated blend — do not "optimize" the gate or the tabular
   fit away.
2. **`decompose` path** — `answer_shape="decompose"` is recorded but not acted
   on (recursive split is a v0.3 decision).
3. **Local 24B fallback** — LFM2-24B-A2B is open weights; a quantized GGUF on
   the existing llama.cpp build needs more than this machine's 16 GB unified
   memory.
4. **MRL truncation** — the draft's 128d truncation is unverified on this
   build; the lane keeps full 768d until probed.
5. **Neighbor relevance gating** — the gate now catches the drift irrelevant
   neighbors cause (the cheaper fix landed); a relevance floor on injection
   remains the complementary option.
6. **Live retry cycle** — proven offline; the organic drift observed earlier
   today rates 0.30 on GLiDE (clear fail → retry path), but the retry has not
   fired in a live run yet.
7. **Git** — this recap and the methodology artifacts are unstaged for review
   (methodology rule: recaps are never auto-committed); blend1's commits
   (`34b4126`, `d597b6d`, `00fa252`) and the sem1 fix (`b7bd58b`) are already in.
