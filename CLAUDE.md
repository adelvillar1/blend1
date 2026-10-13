# blend1

> The blended lane of the sys1 house pattern: one `ask()` that runs the Liquid d1 router
> (sys1), the local EmbeddingGemma-2 semantic lane (sem1), and the Prior Labs TabPFN
> tabular lane (sdm1) in parallel, then synthesizes with LFM2.5-24B-A2B (Liquid's own
> API) only when the router says generative synthesis is required.
> Repo: https://github.com/adelvillar1/blend1

**Lane keys, endpoints, and local paths are in `CLAUDE.local.md` (gitignored, auto-loaded
alongside this file).** Read it before any lane, service, or credential operation. Never
paste key values into commits, docs, or chat.

---

## Hard rules (non-negotiable)

- **Facts before prose.** TabPFN's number reaches the 24B as a given in the system prompt
  (`S2_SYSTEM_TEMPLATE`); the model is instructed never to invent, round, or alter it.
  LLMs do not do math — TabPFN does.
- **The router gates the paid leg.** The 24B is called ONLY when d1's
  `needs_generation` noul clears `ROUTE_FLOOR` (0.70, Jev-family semantics — the noul
  value IS the probability). Direct asks must never spend generative tokens.
- **Every lane fails open with its reason named.** A lane that cannot answer records its
  error in its lane report and the orchestrator composes from the lanes that did answer.
  `BlendResult.escalated` is True when any lane errored. Never silently degrade.
- **Never print, log, or return key values.** `doctor()` reports availability + key
  *names*; the JSONL log carries sha256 + counts only — never prompt text, answer text,
  or key material (`telemetry.REDACT_KEYS` enforces it).
- **Result keys are an additive contract.** `types.RESULT_KEYS` may only grow; renaming
  or removing a key is a breaking change and needs a plan.
- **Lane wires are upstream contracts.** d1 speaks the TypeSafe System One API via sys1;
  EmbeddingGemma-2 is text-only by declaration (the local build tokenizes images as text
  and returns confident garbage); TabPFN is Prior Labs' REST flow via sdm1; LFM2.5 is
  Liquid's `/openai/v1/chat/completions` (NOT `/v1/chat/completions` — that 404s).
- Default operation target is `main`. Feature branches are normal; merging to `main`
  requires explicit user approval *in the current turn*.

---

## Where to find things

**Contracts**
- Plan (foundation, all ACs met) → `docs/plans/2026-10-13-blend1-foundation.md`
- Session journal → `docs/recaps/`
- Technical contract → `TECHNICAL-DOCUMENTATION.md`
- Functional contract → `FUNCTIONAL-SPECIFICATIONS.md`

**Code map** (all under `src/blend1/`)
- `orchestrator.py` — `ask()`: parallel lanes, merge, route, telemetry
- `lanes.py` — the five lane callables + `lane_report()` (doctor)
- `corpus.py` — semantic memory (sem1 VectorStore, top-k cosine)
- `types.py` — `BlendResult` wire contract
- `env.py` — key-file loading (`~/.config/blend1/env` → `sdm1/env` → `dev-decisions/env` → `gen1/env`)
- `telemetry.py` — redacted JSONL log
- `cli.py` — `blend1 ask|doctor|seed`
- `service/main.py` — FastAPI wrapper (`/v1/ask`, `/v1/health`, port 8420)

**Sibling lanes** (path dependencies, editable): `../sys1`, `../sem1`, `../sdm1`, `../gen1`

---

## Common commands

```bash
uv sync                          # installs blend1 + the four siblings (editable path deps)
uv run --extra dev pytest        # offline suite (stub lanes, no keys)
uv run --extra dev ruff check .  # lint
uv run blend1 doctor             # per-lane availability, reasons named
uv run blend1 ask "..." --table churn.csv --target churned [--voice] [--json]
uv run blend1 seed corpus.jsonl  # semantic memory: [{"key","text"}...]
uv run python scripts/smoke_live.py   # live: every lane, real calls
cd ~/Projects/blend1 && service/.venv/bin/uvicorn service.main:app --port 8420   # HTTP service (run from PROJECT ROOT)
```

---

## Today's state (2026-10-13)

- v0.1.0 committed (`34b4126` + `.gitignore`). 15/15 offline tests green, ruff clean.
- **All four model lanes live-verified end-to-end**: d1 routes (0.002 direct / 0.978
  synthesized on the probe prompts), EmbeddingGemma-2 retrieves (768d, 60–110ms, churn
  prompt → churn-playbook cos 0.771), TabPFN returns real predictions (p=0.924 churn,
  ~12s), lfm2.5-24b-a2b synthesizes (~112 tok/s, $0.0016/ask, carries the exact TabPFN
  number + neighbors into its answer).
- **v0.2 validation gate shipped** (plan `2026-10-13-blend1-validation-gate`,
  completed): GLiDE rates every synthesized answer (`answers_the_question` +
  `contradicts_facts` nouls gate; `answer_quality` score is telemetry-only — glide's
  score confidence measures ~0.008); a deterministic number check (percent/probability
  canonicalized) flags model-invented numbers; definitive fails get ONE retry with the
  verdict injected as feedback; **inconclusive escalates to the critic chain**
  (qwen3.8-flash via Token Plan → glm-5.3-flash via ollama-cloud, both from the Hermes
  agent config). The gate never blocks — a down critic ships ungated + escalated.
- Voice lane (gen1) is opt-in (`voice=True`); keys present.
- Fixed upstream this session: sem1 `normalize_items` iterated a bare string into
  single-character items (committed in sem1 as `b7bd58b`).
- Open: `decompose` shape recorded but not acted on (v0.3?); gate floors UNFITTED
  until feedback rows exist; local-24B GGUF deferred (16 GB machine); neighbor
  relevance gating (recap #5) — the gate now catches the drift it causes, which is
  the cheaper fix.
- **User endorsement 2026-10-13: blend1 is a keeper for future use. The ~15s
  full-blend duration is NOT a concern — accuracy and savings justify it. Do not
  propose latency optimizations (skipping the gate, caching the tabular fit,
  trimming the parallel lanes) unless asked.**
