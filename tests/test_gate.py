"""The refusal gate on a hand-built bench index: each check catches what it names, and a clean corpus passes."""

from __future__ import annotations

import json

import pytest

from binding_bench import gate as G

BENCH = ["A chat between Jack and Carol.\nCarol: Here, Jack, take my cup.\nRight now Jack is holding the",
         "The tin holds the nail. The bag holds the leaf. Then the tin is emptied and the coin is put in it. "
         "So the tin holds the"]
POOL = frozenset({"jack", "carol", "cup", "tin", "nail", "bag", "leaf", "coin"})


def _index() -> G.BenchIndex:
    lines, skels, grams = set(), set(), set()
    for p in BENCH:
        for ln in p.splitlines():
            if len(ln) >= G.MIN_LINE_CHARS:
                lines.add(ln)
            if len(G._words(ln)) >= G.MIN_TEMPLATE_WORDS:
                skels.add(G.skeleton(ln, POOL))
        grams.update(G._grams(p))
    return G.BenchIndex(lines, skels, grams, POOL, "test")


def test_skeleton_slots_pool_words_and_numbers():
    assert G.skeleton("Carol: Here, Jack, take my cup 3.", POOL) == "_ here _ take my _ #"


def test_a_clean_corpus_passes():
    rep = G.check([{"text": "Quorn waved at Ylsa.\nYlsa: The moth-lamp is mine now.", "fillers": ["Quorn", "Ylsa"]}],
                  _index())
    assert not rep.refused and rep.records == 1


@pytest.mark.parametrize("rec, which", [
    ({"text": "Carol: Here, Jack, take my cup."}, "line"),
    ({"text": "Brask: Here, Ylsa, take my quill.", "fillers": ["Brask", "Ylsa", "quill"]}, "template"),
    ({"text": "Tom met someone.", "fillers": ["carol"]}, "pool"),
    ({"text": "Then the tin is emptied and the coin is put in it. So the tin holds the nail, said Ylsa."}, "ngram"),
])
def test_each_check_refuses_what_it_names(rec, which):
    rep = G.check([rec], _index())
    assert rep.refused and getattr(rep, which).count >= 1


def test_a_record_without_text_is_refused_loudly():
    with pytest.raises(ValueError):
        G.check([{"fillers": []}], _index())


def test_the_cli_writes_a_manifest_and_exits_nonzero_on_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(G.BenchIndex, "build", staticmethod(lambda root, rows=(): _index()))
    corpus = tmp_path / "c.jsonl"
    corpus.write_text(json.dumps({"text": "Carol: Here, Jack, take my cup."}) + "\n")
    man = tmp_path / "m.json"
    assert G.main(["--corpus", str(corpus), "--prompts-dir", str(tmp_path), "--manifest", str(man)]) == 1
    m = json.loads(man.read_text())
    assert m["verdict"] == "REFUSED" and m["line"]["count"] == 1 and len(m["corpus_sha256"]) == 64
