"""Tests for blend1's corpus derivation — the two keys the KI triage fixed.

The derivation is the lane's own, so its tests live beside the adapter's
in this vendored tree's tests/ dir. These pin:

- KI-223: the chunk key is the source's ABSOLUTE path, so the same
  source derives the same key set from any working directory. The key
  used to be relative to the process's cwd, which made the refresh's
  change detection report "rebuilt" for byte-identical text whenever the
  lane was launched from a different directory.
- KI-225: a declared source that resolves to nothing is named in the
  refresh's `missing_sources`, rather than the refresh deriving
  silently from whatever remained.

Run: python3 -m pytest services/blend1/tests/test_corpus_keys.py
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
# The refresh's embed path imports the sibling sem1 core, the way the
# launch spec's BLEND1_SEM1_PYTHONPATH wires it for the lane.
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent / "sem1" / "src")
)

from blend1 import corpus  # noqa: E402


def test_the_key_is_the_absolute_path_so_the_cwd_cannot_poison_it(tmp_path):
    """KI-223: the same source derives the same keys from any cwd."""
    source = tmp_path / "doc.md"
    source.write_text("# Title\n\n## One\n\nfirst\n\n## Two\n\nsecond\n", encoding="utf-8")
    absolute = str(source.resolve())

    here = os.getcwd()
    try:
        items_from_here, _, _, _ = corpus._derive([absolute])
        os.chdir(tmp_path)
        items_from_there, _, _, _ = corpus._derive([absolute])
    finally:
        os.chdir(here)

    keys_here = [item["key"] for item in items_from_here]
    keys_there = [item["key"] for item in items_from_there]
    assert keys_here == keys_there, (
        f"the same source derived different keys per cwd:\n{keys_here}\n{keys_there}"
    )
    assert all(key.startswith(str(tmp_path.resolve())) for key in keys_here), keys_here


def test_a_source_that_does_not_exist_is_named_not_skipped(tmp_path):
    """KI-225: a declared source resolving to nothing rides the report."""
    present = tmp_path / "here.md"
    present.write_text("## Only\n\ntext\n", encoding="utf-8")
    absent = str(tmp_path / "gone.md")

    files, missing = corpus._source_files([str(present), absent])
    assert [path.name for path in files] == ["here.md"], files
    assert missing == [absent], f"the absent source is named: {missing}"


def test_the_missing_sources_ride_the_refresh_report(tmp_path, monkeypatch=None):
    """The refresh's report carries them, so a stale job says so."""
    report = corpus.refresh(project="ki-225-probe", sources=[str(tmp_path / "gone.md")])
    assert report["action"] == "no-source", report
    assert report["missing_sources"] == [str(tmp_path / "gone.md")], report


def test_a_repeated_heading_dedupes_at_derive_and_counts_what_it_dropped(tmp_path):
    """KI-297: a doc that repeats a `##` heading with different bodies
    used to leave the derived (key, sha) set permanently larger than the
    store, which keeps one entry per key — every refresh reported
    rebuilt and re-embedded a byte-identical corpus. Derive keeps one
    occurrence per key (the first, in document order) and counts what it
    dropped, so the truncation is visible instead of silent."""
    source = tmp_path / "doc.md"
    source.write_text(
        "# Title\n\n## Notes\n\nfirst copy\n\n## Notes\n\nsecond copy\n",
        encoding="utf-8",
    )
    resolved = str(source.resolve())
    items, _, _, dropped = corpus._derive([str(source.resolve())])
    assert [item["key"] for item in items] == [
        f"{resolved}#(preamble)",
        f"{resolved}#Notes",
    ], items
    assert dropped == 1, dropped
    # A doc that repeats nothing drops nothing.
    plain = tmp_path / "plain.md"
    plain.write_text("## Only\n\ntext\n", encoding="utf-8")
    _, _, _, dropped = corpus._derive([str(plain.resolve())])
    assert dropped == 0, dropped


def test_a_repeated_heading_converges_in_the_refresh(tmp_path, monkeypatch):
    """The steady state the row asked for by name: a second refresh over
    a corpus whose doc repeats a heading reports unchanged — no
    re-embed, no byte-identical store rewrite."""
    monkeypatch.setenv("BLEND1_VECTORS_DIR", str(tmp_path / "vectors"))
    source = tmp_path / "doc.md"
    source.write_text(
        "# Title\n\n## Notes\n\nfirst copy\n\n## Notes\n\nsecond copy\n",
        encoding="utf-8",
    )
    try:
        import sem1  # noqa: F401  the refresh's own embed path

        sem1.embed(["probe"])
    except Exception:
        pytest.skip("this machine has no embedding backend")
    first = corpus.refresh(project="ki-297-probe", sources=[str(source)])
    assert first["action"] == "rebuilt", first
    assert first["dropped_duplicates"] == 1, first
    second = corpus.refresh(project="ki-297-probe", sources=[str(source)])
    assert second["action"] == "unchanged", second
    assert second["dropped_duplicates"] == 1, second
