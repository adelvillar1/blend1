# Plan: the escalation leg + the sys1-scored reprocess

status: completed (2026-10-14)

## What

Two capabilities the owner named, upstream in blend1, on a feature branch
(`feature/escalation-and-sys1-score`; no merge to main — that needs explicit
approval in the current turn).

1. **The escalation leg.** When the validation gate cannot clear an answer —
   an inconclusive verdict (`passed` None, no critic answered) or the bounded
   retry not clearing the gate — the ask escalates to a LARGER configured
   model with the same givens (prompt + computed tabular facts + semantic
   neighbors + web excerpts, the untrusted-source framing intact). The
   escalated answer ships FLAGGED via a new result key. The target is
   CONFIGURED (env), never hardcoded; unconfigured is a named skip, never a
   crash and never a silent spend. Bounded: one attempt per ask.
2. **The sys1-scored reprocess.** sys1 (d1) scores the generated answer's
   correctness against the givens — the same state shape the GLiDE critic
   reads. A WRONG score reprocesses once with the score's feedback (the
   retry's discipline). The GLiDE gate stays; the score is the correctness
   judgment beside it. An unavailable sys1 names the absence, never a
   fabricated score.

## Acceptance criteria

- [x] AC1 — escalation fires on an inconclusive verdict; the escalated answer
  ships flagged by a NEW result key; `escalated`'s meaning is unchanged.
- [x] AC2 — escalation fires when the bounded retry does not clear the gate.
- [x] AC3 — an unconfigured escalation target is a named skip (env vars named),
  never a crash, never a spend.
- [x] AC4 — a failed escalation ships the original answer with the failure
  named; the gate never blocks.
- [x] AC5 — one escalation attempt per ask; worst-case cost named in the
  docstring (initial generate + gate retry + escalation + score reprocess
  ≈ 4x generate cost, failure paths only).
- [x] AC6 — the sys1 score rides in the result (score, reprocessed, shipped).
- [x] AC7 — a wrong score reprocesses once with the score's feedback and the
  corrected answer ships.
- [x] AC8 — a right score does not reprocess (no double spend).
- [x] AC9 — an unavailable sys1 (the route lane errored, or the score lane
  itself fails) names the absence; no reprocess; never a fabricated score.
- [x] AC10 — `types.RESULT_KEYS` grows additively; no rename, no removal.
- [x] AC11 — the telemetry row carries the new keys' shapes and still no
  prompt/answer text or key material.
- [x] AC12 — offline suite green (before/after counts reported), ruff clean;
  no live paid calls in tests (the `_lane_overrides` seam + mocked transports).
- [x] AC13 — CLAUDE.md code map + TECHNICAL-DOCUMENTATION.md updated (flow,
  bounds, result keys, configuration), English-only.

## Design decisions (recorded for the re-verifier)

- **Order**: generate → GLiDE gate (unchanged) → sys1 score (on the generated
  answer; wrong → one reprocess) → escalation leg (when the gate could not
  clear) → final. The score judges the generated answer literally; the
  escalation gets the last word when the gate could not clear, so the largest
  model sees the best-informed answer and all feedback.
- **Escalation trigger** is exactly the brief's two: gate `status == "ok"` and
  `passed is None`, or `retry_passed is False`. A down critic (`status`
  "error") is unchanged — it already ships ungated.
- **Escalation config** (env, resolved like the critic chain's entries):
  `BLEND1_ESCALATION_MODEL` + `BLEND1_ESCALATION_BASE` (both required to arm
  the leg) + `BLEND1_ESCALATION_KEY_ENV` (names the key variable, default
  `BLEND1_ESCALATION_API_KEY`). No target or no key → `status: "skipped"`
  with the reason naming the vars; a transport/parse failure →
  `status: "error"` and the original ships.
- **New result keys** (additive): `escalation` (lane report), `score` (lane
  report: `correct`, `reprocessed`, `shipped`), `escalated_answer` (bool —
  the flag beside `escalated`, whose meaning stays exactly "a lane failed
  open").
- **Score floor** `SCORE_FLOOR = 0.70` (noul semantics, UNFITTED until
  feedback rows exist — the repo's gate-floor convention). Binary verdict:
  a noul at/above the floor is correct, below is wrong → reprocess.
- **The reprocess ships on lane success** without a second gate pass (bounded
  discipline: one attempt per leg; the gate's verdict on the original stands
  and rides in the result).

## Files

- `src/blend1/lanes.py` — `escalation_lane`, `score_lane`, `_score_task`,
  `SCORE_FLOOR`, `_escalation_target`, shared `_openai_chat`, doctor rows.
- `src/blend1/orchestrator.py` — the two legs, feedback renderers, result +
  telemetry keys, flow docstring.
- `src/blend1/types.py` — RESULT_KEYS + BlendResult grow; `escalated` covers
  the new lane reports.
- `tests/test_orchestrator.py` — new leg tests + lane-internals tests with
  mocked transports; existing synthesized tests gain `score`/`escalation`
  stubs so the offline suite never touches a live lane.
- `CLAUDE.md`, `TECHNICAL-DOCUMENTATION.md` — the two legs documented.
