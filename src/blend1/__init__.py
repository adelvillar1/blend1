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
    seed_corpus(items, project=None) -> rebuild a project's semantic store
    search(query, project=None, k=3) -> semantic neighbors from a project's corpus
    refresh_corpus(project=None, sources=None) -> docs-derived refresh report

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

__all__ = [
    "BlendResult",
    "__version__",
    "ask",
    "doctor",
    "refresh_corpus",
    "search",
    "seed_corpus",
]


def doctor() -> dict:
    """Per-lane availability snapshot — one row per lane, reasons named."""
    from .lanes import lane_report

    return lane_report()


def seed_corpus(items: list[dict], *, project: str | None = None) -> dict:
    """Embed corpus items into a project's semantic memory store (sem1
    VectorStore). project resolves param > BLEND1_PROJECT > cwd repo
    basename > None (the legacy store).

    items: [{"key": "...", "text": "..."}] — the semantic neighbors a later
    ask() retrieves. Idempotent by construction (store dedupes by key).
    """
    from .corpus import seed

    return seed(items, project=project)


def refresh_corpus(
    *, project: str | None = None, sources: list[str] | None = None
) -> dict:
    """Re-derive the project's corpus from its docs and rebuild only when
    the text hashes changed (steady state: file reads, no embedding).

    sources: explicit files/globs replacing the default docs set. Report:
    {project, source_files, items, changed, action, store} — an empty
    source set is action "no-source" and creates no store.
    """
    from .corpus import refresh

    return refresh(project=project, sources=sources)


def search(query: str, *, project: str | None = None, k: int = 3) -> list[dict]:
    """Semantic neighbors for a query against a project's corpus —
    [{key, score, text}], best first. An unseeded store returns []."""
    from .corpus import search as corpus_search

    return corpus_search(query, project=project, k=k)
