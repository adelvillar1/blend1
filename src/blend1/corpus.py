"""Semantic memory: a small corpus embedded with sem1, stored with sem1's
VectorStore (packed float32 + JSON manifest), retrieved by cosine.

The store lives under ~/.local/share/blend1/vectors — blend1's own root, so
it never collides with sem1's default store. Corpus text is stored in the
manifest entries (a local corpus, not telemetry); the JSONL log never sees it.

Per-project corpora: a named project gets its own store at
<root>/<project>/<slug>/…, so any process (not just blend1) can read a
project's facts without touching another project's — the store is plain
files under a documented path and the embedding server is shared local
HTTP. project None (or "default") IS the legacy store at the root itself:
byte-identical behavior, zero migration.

Docs-derived refresh indexes the project's own docs: the manifest already
carries a per-entry sha256 of each text, so the staleness check is file
reads + hashes (milliseconds, no embedding) and only a changed source
triggers a rebuild.
"""

from __future__ import annotations

import glob
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

STORE_ROOT = (
    Path(os.environ["BLEND1_VECTORS_DIR"])
    if os.environ.get("BLEND1_VECTORS_DIR")
    else Path.home() / ".local" / "share" / "blend1" / "vectors"
)
MODEL_SLUG = "embeddinggemma-2"  # store slug; the real model id rides in the manifest

# Docs-derived refresh — the default source set (relative to CWD): every
# docs/**/*.md plus the root contract docs when they exist.
DEFAULT_ROOT_DOCS = ("CLAUDE.md", "TECHNICAL-DOCUMENTATION.md", "FUNCTIONAL-SPECIFICATIONS.md")
SECTION_MAX_CHARS = 2000

# The one-shot embed's budget, in seconds: the corpus a refresh derives is
# embedded in a single call, and a corpus past http_llama's inherited 60-second
# default times out mid-embed (KI-222). Ten minutes is the measured headroom
# for a corpus several times fleet's own 638 entries.
EMBED_TIMEOUT_SECONDS = 600.0  # v1 chunking cap — tunable; keys are visible in search output
# CLAUDE.local.md carries key values — it is NEVER in a corpus, even when a
# caller's glob would otherwise match it: other processes read these stores.
EXCLUDED_FILENAMES = frozenset({"CLAUDE.local.md"})


def _base_root() -> Path:
    """The store root, resolved from the environment AT CALL TIME — tests
    monkeypatch BLEND1_VECTORS_DIR, so the module-level STORE_ROOT constant
    is kept for backward compatibility only and never read here."""
    env = os.environ.get("BLEND1_VECTORS_DIR")
    return Path(env) if env else Path.home() / ".local" / "share" / "blend1" / "vectors"


def store_root(project: str | None) -> Path:
    """Store root for a project. None/"default" → the legacy root itself
    (zero migration); a named project → <root>/<project>."""
    root = _base_root()
    if project is None or project == "default":
        return root
    # A project name is ONE path component — no separators, no dot-dirs.
    # These stores are documented as shared with other processes, so a
    # crafted --project must not escape the root.
    if project in (".", "..") or "/" in project or Path(project).name != project:
        raise ValueError(
            f"project name must be a single path component (got {project!r}); "
            "it names a directory under the store root"
        )
    return root / project


def resolve_project(explicit: str | None = None) -> str | None:
    """explicit param > BLEND1_PROJECT env > cwd git-repo basename > None.

    No cwd-name fallback: a non-repo cwd resolves to None (legacy behavior),
    never a guess from the directory name. The subprocess runs only when
    neither the param nor the env already decided."""
    if explicit:
        return explicit
    env = os.environ.get("BLEND1_PROJECT")
    if env:
        return env
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None  # not a repo (or git missing/hung) → legacy
    return Path(top).name or None  # a repo at the filesystem root has no name → legacy


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _rebuild(items: list[dict], root: Path) -> dict:
    """Embed `items` and rebuild the store under `root` wholesale — sem1's
    rebuild replaces the model's store (dedup by key, atomic tmp+rename),
    so a rebuild over the same corpus is idempotent by construction.

    KI-222: the embed rides a longer timeout than http_llama's inherited
    60-second default. A corpus past ~500 chunks — fleet's own is 638
    entries over 85 sources as of 2026-10-17 — cannot finish inside it,
    and the ceiling was measured as "unbuildable through the lane's
    one-shot embed" when it is only unbuildable inside that default.
    The kwarg exists at `sem1/providers/http_llama.py:28`; this passes
    it, and the corpus size is named in the docstring so the next
    ceiling is a measurement rather than a surprise."""
    import sem1
    from sem1.store import VectorStore

    texts = [it["text"] for it in items]
    res = sem1.embed(texts, timeout=EMBED_TIMEOUT_SECONDS)
    entries = [
        (it["key"], _hash_text(it["text"]), vec)
        for it, vec in zip(items, res.vectors, strict=True)
    ]
    meta = [{"text": it["text"]} for it in items]
    # Store under the fixed slug (the server's model id is a full gguf path —
    # it stays out of the directory name, and the manifest carries only the
    # slug + provider, so retrieve() finds the store regardless of which
    # snapshot served).
    manifest = VectorStore(root).rebuild(
        MODEL_SLUG, res.dims, res.provider, entries, meta
    )
    return {"count": manifest["count"], "dims": manifest["dims"], "model": manifest["model"]}


def seed(items: list[dict], *, project: str | None = None) -> dict:
    """Embed corpus items and rebuild the project's store. Idempotent by construction.

    items: [{"key": "...", "text": "..."}]
    """
    if not items:
        raise ValueError("seed needs at least one {key, text} item")
    return _rebuild(items, store_root(resolve_project(project)))


def retrieve(
    query_vector: list[float],
    *,
    k: int = 3,
    min_score: float = 0.0,
    project: str | None = None,
) -> list[dict]:
    """Top-k corpus neighbors for an embedded query. Empty when unseeded."""
    from sem1.similarity import cosine
    from sem1.store import VectorStore

    manifest, vectors = VectorStore(store_root(resolve_project(project))).load(MODEL_SLUG)
    if not manifest or not vectors:
        return []
    scored = []
    for entry, vec in zip(manifest["entries"], vectors, strict=True):
        score = cosine(query_vector, vec)
        if score >= min_score:
            scored.append({
                "key": entry["key"],
                "score": round(score, 4),
                "text": entry.get("text", ""),
            })
    scored.sort(key=lambda n: n["score"], reverse=True)
    return scored[:k]


def _source_files(sources: list[str] | None) -> tuple[list[Path], list[str]]:
    """The files to derive from: `sources` (file paths or globs) REPLACES the
    default set (docs/**/*.md + the root contract docs). .md-only by
    construction and CLAUDE.local.md never — it carries key values and these
    stores are read by other processes. Deduped, sorted, deterministic.

    Returns (files, missing): the second is every declared source that
    resolved to nothing. KI-225 — a job whose task string lists sources
    that do not exist used to derive silently from whatever remained,
    so a stale job looked like a successful refresh of fewer files. A
    missing source is named in the report instead."""
    if sources is not None:
        matched: list[Path] = []
        missing: list[str] = []
        for pattern in sources:
            hits = glob.glob(pattern, recursive=True)
            if not hits and not Path(pattern).is_file():
                missing.append(pattern)
            matched.extend(Path(p) for p in hits)
    else:
        matched = list(Path("docs").rglob("*.md"))
        matched.extend(Path(name) for name in DEFAULT_ROOT_DOCS if Path(name).is_file())
    seen: set[Path] = set()
    out: list[Path] = []
    for p in sorted(matched):
        resolved = p.resolve()
        # The exclusion tests BOTH names: p.name catches a direct match,
        # resolved.name catches a symlink under an innocent name (a link
        # named docs/keys.md pointing at CLAUDE.local.md resolves to it).
        if (
            resolved in seen
            or p.suffix != ".md"
            or p.name in EXCLUDED_FILENAMES
            or resolved.name in EXCLUDED_FILENAMES
        ):
            continue
        if not p.is_file():
            continue
        seen.add(resolved)
        out.append(p)
    return out, (missing if sources is not None else [])


def _chunks(path: Path) -> list[tuple[str, str, str]]:
    """(relpath, heading, text) chunks for one file: split on ^## headings
    (### stays in the section body); a non-empty preamble before the first
    ## is one entry keyed (preamble). Text = heading line + section body,
    capped at SECTION_MAX_CHARS. The key is the ABSOLUTE path, POSIX.

    KI-223: the key used to be relative to the process's cwd, so the
    same absolute sources under a different cwd derived a different key
    set with byte-identical text — every cwd variation rebuilt the
    whole store and re-embedded unchanged text. The absolute path is
    the one cwd-independent name a source has; the cost is that a store
    built before this change rebuilds once and then holds."""
    rel = path.resolve().as_posix()
    # errors="replace": a non-UTF-8 .md file degrades to replacement chars,
    # never a traceback mid-refresh.
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    preamble: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    heading = ""
    current: list[str] | None = None
    for line in lines:
        if line.startswith("## "):
            if current is not None:
                sections.append((heading, current))
            heading = line[3:].strip()
            current = [line]
        elif current is None:
            preamble.append(line)
        else:
            current.append(line)
    if current is not None:
        sections.append((heading, current))
    out: list[tuple[str, str, str]] = []
    if any(line.strip() for line in preamble):
        out.append((rel, "(preamble)", "\n".join(preamble)))
    out.extend((rel, h, "\n".join(body)) for h, body in sections)
    return [(r, h, t[:SECTION_MAX_CHARS]) for r, h, t in out]


def _derive(sources: list[str] | None) -> tuple[list[dict], int, list[str], int]:
    """(items, source_file_count, missing_sources, dropped_duplicates) —
    items deduplicated by key, first occurrence kept, sorted by
    (relpath, heading) so a rebuild over unchanged docs is byte-stable.

    A doc that repeats a `## heading` with different bodies used to
    leave the derived (key, sha) set permanently larger than the store,
    which keeps one entry per key — every refresh reported rebuilt and
    re-embedded a byte-identical corpus, four projects recorded
    (KI-297). The dropped count keeps the truncation visible instead of
    silent: a repeated heading is a doc shape worth knowing about."""
    files, missing = _source_files(sources)
    triples: list[tuple[str, str, str]] = []
    for path in files:
        triples.extend(_chunks(path))
    triples.sort(key=lambda t: (t[0], t[1]))
    items: list[dict] = []
    seen: set[str] = set()
    dropped = 0
    for rel, heading, text in triples:
        key = f"{rel}#{heading}"
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        items.append({"key": key, "text": text})
    return items, len(files), missing, dropped


def derive_items(sources: list[str] | None = None) -> list[dict]:
    """Derive corpus items from the source docs: one item per ## section
    (+ non-empty preambles), keys `relpath#heading`, sorted by (relpath,
    heading) — deterministic."""
    return _derive(sources)[0]


def refresh(*, project: str | None = None, sources: list[str] | None = None) -> dict:
    """Re-derive the corpus from docs and rebuild ONLY when the derived
    (key, sha256) set differs from the store manifest — steady state is file
    reads + hashes, no embedding. Never raises on a missing/empty source
    set: action "no-source", no store created. Report:
    {project, source_files, items, changed, action, store}.

    Known limitation (documented, not fixed): the check compares text
    hashes only. If the embedding server's model/dims change behind the
    fixed slug, hashes still match and the store reports "unchanged" —
    model drift is undetectable without an embedding call. Re-seed
    explicitly (`blend1 seed`) after a model change.
    """
    resolved = resolve_project(project)
    root = store_root(resolved)
    items, n_files, missing, dropped = _derive(sources)
    report: dict = {
        "project": resolved,
        "source_files": n_files,
        "items": len(items),
        "dropped_duplicates": dropped,
        "changed": False,
        "action": "no-source",
        "store": str(root),
        "missing_sources": missing,
    }
    if not items:
        return report
    from sem1.store import VectorStore

    manifest, _vectors = VectorStore(root).load(MODEL_SLUG)
    current = {(e["key"], e["sha256"]) for e in manifest["entries"]} if manifest else set()
    derived = {(it["key"], _hash_text(it["text"])) for it in items}
    if current == derived:
        report["action"] = "unchanged"
        return report
    _rebuild(items, root)
    report["changed"] = True
    report["action"] = "rebuilt"
    return report


def search(query: str, *, project: str | None = None, k: int = 3) -> list[dict]:
    """Embed the query and return top-k corpus neighbors [{key, score, text}].

    A direct call, not a lane: an unreachable embedding server raises here
    and the CLI/service reports it; an unseeded store returns [] cleanly."""
    import sem1

    res = sem1.embed([query])
    return retrieve(res.vectors[0], k=k, project=project)


def migrate_legacy(project: str) -> dict:
    """Move the legacy store (<root>/<slug>/) under a project
    (<root>/<project>/<slug>/) when the legacy path exists and the target
    doesn't. Idempotent no-op otherwise. Report: {moved, from, to}."""
    legacy = _base_root() / MODEL_SLUG
    target = store_root(project) / MODEL_SLUG
    report = {"moved": False, "from": str(legacy), "to": str(target)}
    if legacy.is_dir() and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy), str(target))
        report["moved"] = True
    return report
