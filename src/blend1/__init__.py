"""blend1 — the blended lane of the sys1 house pattern.

sys1 judges what the work says (decision models), sdm1 scores what the work
measures (tabular foundation models), sem1 indexes what the work looks like
(embeddings), gen1 generates what the work needs heard and seen — **blend1
blends them into one answer**: the d1 router decides whether a 24B generative
call is required, the semantic and tabular lanes precompute the facts in
parallel, and the generative lane synthesizes only when the router says so.

Public API:
    ask(prompt, table=None, voice=False) -> BlendResult
    doctor() -> per-lane availability snapshot
    seed_corpus(items) -> rebuild the semantic memory store
"""

from __future__ import annotations

from . import telemetry  # noqa: F401  (importing configures the log dir)
from .env import load_env
from .orchestrator import ask
from .types import BlendResult

__version__ = "0.1.0"

# Lane keys are read from the environment or ~/.config/blend1/env (chmod 600,
# never committed) — the gen1/dev-decisions key-file precedent. sys1 keeps
# loading its own provider env file (~/.config/sys1/env or the legacy
# dev-decisions fallback); blend1 never touches sys1's key material.
load_env()

__all__ = ["BlendResult", "__version__", "ask", "doctor", "seed_corpus"]


def doctor() -> dict:
    """Per-lane availability snapshot — one row per lane, reasons named."""
    from .lanes import lane_report

    return lane_report()


def seed_corpus(items: list[dict]) -> dict:
    """Embed corpus items into the semantic memory store (sem1 VectorStore).

    items: [{"key": "...", "text": "..."}] — the semantic neighbors a later
    ask() retrieves. Idempotent by construction (store dedupes by key).
    """
    from .corpus import seed

    return seed(items)
