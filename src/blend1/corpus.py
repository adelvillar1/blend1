"""Semantic memory: a small corpus embedded with sem1, stored with sem1's
VectorStore (packed float32 + JSON manifest), retrieved by cosine.

The store lives under ~/.local/share/blend1/vectors — blend1's own root, so
it never collides with sem1's default store. Corpus text is stored in the
manifest entries (a local corpus, not telemetry); the JSONL log never sees it.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

STORE_ROOT = (
    Path(os.environ["BLEND1_VECTORS_DIR"])
    if os.environ.get("BLEND1_VECTORS_DIR")
    else Path.home() / ".local" / "share" / "blend1" / "vectors"
)
MODEL_SLUG = "embeddinggemma-2"  # store slug; the real model id rides in the manifest


def seed(items: list[dict]) -> dict:
    """Embed corpus items and rebuild the store. Idempotent by construction.

    items: [{"key": "...", "text": "..."}]
    """
    import sem1
    from sem1.store import VectorStore

    if not items:
        raise ValueError("seed needs at least one {key, text} item")
    texts = [it["text"] for it in items]
    res = sem1.embed(texts)
    entries = [
        (it["key"], hashlib.sha256(it["text"].encode("utf-8")).hexdigest()[:16], vec)
        for it, vec in zip(items, res.vectors, strict=True)
    ]
    meta = [{"text": it["text"]} for it in items]
    # Store under the fixed slug (the server's model id is a full gguf path —
    # it rides inside the manifest, never in the directory name, so
    # retrieve() finds the store regardless of which snapshot served).
    manifest = VectorStore(STORE_ROOT).rebuild(
        MODEL_SLUG, res.dims, res.provider, entries, meta
    )
    return {"count": manifest["count"], "dims": manifest["dims"], "model": manifest["model"]}


def retrieve(query_vector: list[float], *, k: int = 3, min_score: float = 0.0) -> list[dict]:
    """Top-k corpus neighbors for an embedded query. Empty when unseeded."""
    from sem1.similarity import cosine
    from sem1.store import VectorStore

    manifest, vectors = VectorStore(STORE_ROOT).load(MODEL_SLUG)
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
