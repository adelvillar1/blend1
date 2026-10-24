"""Offline tests: the corpus module — project resolution, namespaced stores,
docs-derived refresh with hash-checked staleness, legacy migration, search.

The embedding server is stubbed: a fake `sem1` module (deterministic
bag-of-words vectors) is installed in sys.modules, while submodule imports
(sem1.store, sem1.similarity) resolve through the fake's __path__ to the
REAL sem1 — so the VectorStore, manifest shape, and atomic rebuild under
test are the real ones. No network, ever.

"""

from __future__ import annotations

import json
import subprocess
import sys
import types

import pytest

from blend1 import corpus

# Deterministic fake embeddings: one count per vocabulary word. A query
# shares words with the item text it should retrieve — enough to prove the
# retrieval wiring without a server.
VOCAB = ["churn", "playbook", "billing", "errors", "docs", "alpha", "beta", "login"]


def _fake_vector(text: str) -> list[float]:
    lowered = text.lower()
    return [float(lowered.count(word)) for word in VOCAB]


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Point the store root at a tmp dir (resolved from the env AT CALL
    time) and clear the project env — every test that touches a store."""
    root = tmp_path / "vectors"
    monkeypatch.setenv("BLEND1_VECTORS_DIR", str(root))
    monkeypatch.delenv("BLEND1_PROJECT", raising=False)
    return root


@pytest.fixture
def fake_sem1(monkeypatch):
    """Install a fake sem1 whose embed() is deterministic; the fake carries
    the real package's __path__ so `from sem1.store import VectorStore`
    still resolves to the real store."""
    import sem1 as real_sem1

    fake = types.ModuleType("sem1")
    fake.__path__ = list(real_sem1.__path__)  # submodules resolve to the real ones

    class _Res:
        def __init__(self, texts: list[str]):
            self.vectors = [_fake_vector(t) for t in texts]
            self.dims = len(VOCAB)
            self.provider = "fake-embed"
            self.telemetry = {"latency_ms": 1}

    def embed(items, provider="llama-server", **kw):
        if isinstance(items, str):
            items = [items]
        return _Res(items)

    fake.__dict__["embed"] = embed
    monkeypatch.setitem(sys.modules, "sem1", fake)
    return fake


def _git_repo(path) -> None:
    subprocess.run(["git", "init", "-q", str(path)], check=True, capture_output=True)


# ── project resolution ──


def test_resolve_project_prefers_explicit_over_env(store, monkeypatch):
    monkeypatch.setenv("BLEND1_PROJECT", "env-project")
    assert corpus.resolve_project("explicit-project") == "explicit-project"


def test_resolve_project_env_beats_git_repo(store, monkeypatch, tmp_path):
    monkeypatch.setenv("BLEND1_PROJECT", "env-project")
    repo = tmp_path / "some-repo"
    repo.mkdir()
    _git_repo(repo)
    monkeypatch.chdir(repo)
    assert corpus.resolve_project() == "env-project"


def test_resolve_project_falls_back_to_git_repo_basename(store, monkeypatch, tmp_path):
    repo = tmp_path / "my-repo"
    repo.mkdir()
    _git_repo(repo)
    monkeypatch.chdir(repo)
    assert corpus.resolve_project() == "my-repo"


def test_resolve_project_non_repo_cwd_is_none(store, monkeypatch, tmp_path):
    plain = tmp_path / "not-a-repo"
    plain.mkdir()
    monkeypatch.chdir(plain)
    # No cwd-name fallback: a non-repo cwd is legacy (None), never a guess.
    assert corpus.resolve_project() is None


# ── namespaced stores ──


def test_project_stores_are_isolated(store, fake_sem1):
    corpus.seed([{"key": "alpha-doc", "text": "alpha billing errors docs"}], project="alpha")
    corpus.seed([{"key": "beta-doc", "text": "beta churn playbook login"}], project="beta")

    alpha_hits = corpus.retrieve(_fake_vector("alpha billing"), project="alpha")
    assert [h["key"] for h in alpha_hits] == ["alpha-doc"]

    beta_hits = corpus.retrieve(_fake_vector("beta churn"), project="beta")
    assert [h["key"] for h in beta_hits] == ["beta-doc"]

    # Isolation is about WHICH store is read: alpha's store cannot contain
    # beta's entry (min_score 0.0 may still admit a zero-cosine row).
    cross = corpus.retrieve(_fake_vector("beta churn"), project="alpha")
    assert all(h["key"] != "beta-doc" for h in cross)


def test_legacy_store_untouched_by_project_seed(store, fake_sem1, monkeypatch, tmp_path):
    # A non-repo cwd resolves to None → the legacy root store; the named
    # project's seed must leave it byte-identical.
    monkeypatch.chdir(tmp_path)
    corpus.seed([{"key": "legacy-doc", "text": "legacy churn docs"}])  # project None → root
    corpus.seed([{"key": "alpha-doc", "text": "alpha billing"}], project="alpha")

    assert (store / corpus.MODEL_SLUG / "manifest.json").is_file()
    assert (store / "alpha" / corpus.MODEL_SLUG / "manifest.json").is_file()
    legacy = json.loads((store / corpus.MODEL_SLUG / "manifest.json").read_text())
    assert [e["key"] for e in legacy["entries"]] == ["legacy-doc"]


def test_default_project_is_the_legacy_store(store, fake_sem1):
    corpus.seed([{"key": "d", "text": "docs"}], project="default")
    assert (store / corpus.MODEL_SLUG / "manifest.json").is_file()
    assert not (store / "default").exists()


def test_retrieve_unseeded_store_is_empty(store, fake_sem1):
    assert corpus.retrieve(_fake_vector("anything"), project="alpha") == []


# ── docs-derived refresh ──


DOCS_TREE = {
    "docs/guide.md": (
        "# Guide\n"  # preamble (everything before the first ##)
        "\n"
        "## Setup\n"
        "install things\n"
        "\n"
        "### Deep detail\n"
        "stays in the section body\n"
        "\n"
        "## Teardown\n"
        "remove things\n"
    ),
    "docs/sub/extra.md": "## Only section\n\nbody\n",
    "docs/huge.md": "## Big\n" + "x" * 5000 + "\n",
    "CLAUDE.md": "## Router\n\nthe slim router\n",
    "TECHNICAL-DOCUMENTATION.md": "preamble only, no sections\n",
    "FUNCTIONAL-SPECIFICATIONS.md": "## Behavior\n\nwhat ask guarantees\n",
    "CLAUDE.local.md": "## Keys\n\nLIQUID_API_KEY lives here\n",
    "notes.txt": "not markdown, never derived\n",
}


def _write_tree(root) -> None:
    for rel, text in DOCS_TREE.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")


def test_derive_items_chunks_sections_with_preamble(store, monkeypatch, tmp_path):
    _write_tree(tmp_path)
    monkeypatch.chdir(tmp_path)
    items = corpus.derive_items()
    keys = [it["key"] for it in items]

    # KI-223 (fleet's carry): the key is the source's ABSOLUTE path, so the
    # same source derives the same key set from any working directory — a
    # refresh's change detection cannot be poisoned by the cwd. One entry
    # per ## section + one non-empty preamble per file, sorted by
    # (abspath, heading) — deterministic.
    absolute = tmp_path.resolve().as_posix()
    assert keys == [
        f"{absolute}/CLAUDE.md#Router",
        f"{absolute}/FUNCTIONAL-SPECIFICATIONS.md#Behavior",
        f"{absolute}/TECHNICAL-DOCUMENTATION.md#(preamble)",
        f"{absolute}/docs/guide.md#(preamble)",
        f"{absolute}/docs/guide.md#Setup",
        f"{absolute}/docs/guide.md#Teardown",
        f"{absolute}/docs/huge.md#Big",
        f"{absolute}/docs/sub/extra.md#Only section",
    ]
    by_key = {it["key"]: it["text"] for it in items}
    assert by_key[f"{absolute}/docs/guide.md#Setup"].startswith("## Setup")
    assert "### Deep detail" in by_key[f"{absolute}/docs/guide.md#Setup"], "### stays in the section body"
    assert all(len(it["text"]) <= corpus.SECTION_MAX_CHARS for it in items)
    assert len(by_key[f"{absolute}/docs/huge.md#Big"]) == corpus.SECTION_MAX_CHARS, "text is capped"
    # CLAUDE.local.md carries key values — never in the corpus, by construction
    assert not any("CLAUDE.local.md" in k for k in keys)
    # non-.md files never enter the set
    assert not any("notes.txt" in k for k in keys)


def test_derive_items_explicit_sources_replace_the_default(store, monkeypatch, tmp_path):
    _write_tree(tmp_path)
    monkeypatch.chdir(tmp_path)
    items = corpus.derive_items(["CLAUDE.md", "docs/sub/*.md"])
    base = tmp_path.resolve().as_posix()
    assert [it["key"] for it in items] == [
        f"{base}/CLAUDE.md#Router",
        f"{base}/docs/sub/extra.md#Only section",
    ]


def _write_source(root, body: str = "## Section one\n\nfirst body\n") -> None:
    p = root / "docs" / "plan.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


def test_refresh_rebuilds_once_then_reports_unchanged(store, fake_sem1, monkeypatch, tmp_path):
    _write_source(tmp_path)
    monkeypatch.chdir(tmp_path)

    first = corpus.refresh()
    assert first["action"] == "rebuilt" and first["changed"] is True
    assert first["items"] == 1 and first["source_files"] == 1
    assert first["project"] is None

    manifest_path = store / corpus.MODEL_SLUG / "manifest.json"
    mtime_ns = manifest_path.stat().st_mtime_ns

    second = corpus.refresh()
    assert second["action"] == "unchanged" and second["changed"] is False
    assert manifest_path.stat().st_mtime_ns == mtime_ns, "unchanged sources must not rewrite"

    # an edited source file derives different text → rebuild
    _write_source(tmp_path, "## Section one\n\nfirst body, now edited\n")
    third = corpus.refresh()
    assert third["action"] == "rebuilt" and third["changed"] is True


def test_refresh_targets_the_resolved_project(store, fake_sem1, monkeypatch, tmp_path):
    _write_source(tmp_path)
    monkeypatch.chdir(tmp_path)
    report = corpus.refresh(project="blend1")
    assert report["project"] == "blend1"
    assert report["store"] == str(store / "blend1")
    assert (store / "blend1" / corpus.MODEL_SLUG / "manifest.json").is_file()


def test_refresh_no_source_creates_no_store(store, fake_sem1, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # empty dir — no docs/, no root docs
    report = corpus.refresh()
    assert report["action"] == "no-source"
    assert report["changed"] is False
    assert report["items"] == 0
    assert not store.exists(), "no store is created for an empty source set"


# ── legacy migration ──


def test_migrate_legacy_moves_once_then_noops(store, fake_sem1):
    legacy = store / corpus.MODEL_SLUG
    legacy.mkdir(parents=True)
    (legacy / "manifest.json").write_text("{}", encoding="utf-8")

    first = corpus.migrate_legacy("alpha")
    assert first["moved"] is True
    assert not legacy.exists()
    assert (store / "alpha" / corpus.MODEL_SLUG / "manifest.json").is_file()

    second = corpus.migrate_legacy("alpha")
    assert second["moved"] is False, "idempotent no-op once the target exists"


def test_migrate_legacy_without_legacy_store_is_a_noop(store):
    report = corpus.migrate_legacy("alpha")
    assert report["moved"] is False
    assert not (store / "alpha").exists()


# ── search surface ──


def test_search_returns_the_seeded_entry(store, fake_sem1):
    corpus.seed([
        {"key": "churn-playbook", "text": "customers with falling login frequency churn within 60 days"},
        {"key": "billing-faq", "text": "billing errors cluster around invoice dates"},
    ], project="alpha")
    hits = corpus.search("churn playbook login", project="alpha")
    assert hits, "search must return the seeded entry"
    assert hits[0]["key"] == "churn-playbook"
    assert hits[0]["score"] > 0
    assert "churn" in hits[0]["text"]


def test_ask_threads_the_resolved_project_to_the_semantic_lane(store, fake_sem1):
    """Plan AC3: ask(..., project="alpha") retrieves alpha's neighbor, not
    beta's — the orchestrator resolves the project once and threads it to
    the semantic lane. Only the route lane is stubbed; the real semantic
    lane runs against the fake embedder and the real stores."""
    from blend1.orchestrator import ask

    corpus.seed([{"key": "alpha-doc", "text": "alpha billing errors docs"}], project="alpha")
    corpus.seed([{"key": "beta-doc", "text": "beta churn playbook login"}], project="beta")

    def _ok_route(prompt):
        return {"status": "ok", "needs_generation": 0.0, "answer_shape": "direct",
                "needs_live_facts": 0.0, "latency_ms": 1}

    r = ask("alpha billing?", project="alpha", log=False, _lane_overrides={"route": _ok_route})
    keys = [n["key"] for n in r.semantic["neighbors"]]
    assert keys == ["alpha-doc"], "ask must retrieve from the named project's store"


# ── review-fix regressions (quality review 2026-10-14) ──

def test_resolve_project_repo_at_filesystem_root_is_none(monkeypatch):
    """A repo at / has no basename (Path('/').name == ''): the resolver must
    return None (legacy), not the empty string — store_root('') would
    silently land in the legacy root and report a phantom project."""
    monkeypatch.delenv("BLEND1_PROJECT", raising=False)

    def _fake_run(cmd, **kw):
        assert cmd[:3] == ["git", "rev-parse", "--show-toplevel"]
        return subprocess.CompletedProcess(cmd, 0, stdout="/\n", stderr="")

    monkeypatch.setattr(corpus.subprocess, "run", _fake_run)
    assert corpus.resolve_project(None) is None


def test_store_root_rejects_path_escape(store):
    """A project name is one path component: separators and dot-dirs are a
    ValueError, not a write outside the store root. These stores are
    documented as shared with other processes — the boundary holds."""
    for bad in ("../../etc", "a/b", "..", "."):
        with pytest.raises(ValueError, match="single path component"):
            corpus.store_root(bad)
    assert corpus.store_root("ok-name").name == "ok-name"


def test_source_files_exclude_local_env_by_any_name(store, tmp_path, monkeypatch):
    """The CLAUDE.local.md exclusion holds for an explicit --from AND for a
    symlink under an innocent name (resolved.name is checked too) — that
    file carries key values and must never enter a shared store."""
    monkeypatch.chdir(tmp_path)
    local = tmp_path / "CLAUDE.local.md"
    local.write_text("LIQUID_API_KEY=secret-value\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    link = tmp_path / "docs" / "keys.md"
    link.symlink_to(local)
    (tmp_path / "docs" / "real.md").write_text("## Real\n\nbody\n", encoding="utf-8")

    # KI-225 (fleet's carry): _source_files returns (files, missing) — a
    # declared source that resolves to nothing is named, not skipped.
    files, missing = corpus._source_files(["CLAUDE.local.md", "docs/*", "docs/keys.md"])
    assert [p.name for p in files] == ["real.md"], "the env file is excluded by every path"
    assert missing == [], "the symlink resolves to the env file and is excluded, not missing"
    items = corpus.derive_items(["docs/*"])
    assert all("secret-value" not in it["text"] for it in items)
