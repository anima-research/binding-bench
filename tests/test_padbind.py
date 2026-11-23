"""padbind-v1: filler sits inside the queried scene, between its context and the query; a scene's rows share one filler;
fillers nest across pads; no filler mentions a candidate or a name of its scene; the ``docs`` condition separates the
documents by EOS; the register reader reads every pad against the unpadded scene. With ``$KOTODAMA_BINDING_PROMPTS``
holding the pinned pool, the generator rebuilds the pin byte for byte."""

from __future__ import annotations

import collections
import hashlib
import json
import os
import re
from pathlib import Path

import pytest

from binding_bench.binding import batteries as bt
from binding_bench.binding import padbind as pb
from binding_bench.binding import readers
from binding_bench.binding import register as rg
from binding_bench.binding.stagegauges import write_rows

WORDS = ("river", "valley", "measure", "season", "history", "planet", "signal", "harvest", "lantern", "theory")
TEST = pb.Version("padbind-test", "register-v3", 6, 1)


@pytest.fixture(scope="module")
def rows(tmp_path_factory: pytest.TempPathFactory) -> list[dict]:
    """A small version over a synthetic pool (plain words; ``tokens`` = word count), with planted documents that name
    every candidate and name of every scene, which must never be drawn."""
    scenes, items = pb.sources(TEST)
    vocab = sorted(set().union(*(pb.scene_words(u) for u in scenes + items)))
    path = tmp_path_factory.mktemp("pool") / "pool.jsonl"
    docs = [{"id": f"d{i}", "text": " ".join(WORDS[(i + k) % len(WORDS)] for k in range(400)), "tokens": 400}
            for i in range(300)]
    for i in range(0, 300, 10):
        docs[i] = {"id": f"planted{i}", "text": " ".join(vocab), "tokens": len(vocab)}
    path.write_text("".join(json.dumps(d) + "\n" for d in docs), encoding="utf-8")
    mp = pytest.MonkeyPatch()
    mp.setattr(pb, "POOL_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    mp.setattr(pb, "VERSIONS", {TEST.name: TEST})
    out = pb.padbind_rows(TEST.name, path)
    mp.undo()
    return out


def _key(r: dict) -> str:
    return r["id"].rsplit("|pad", 1)[0]


def _filler(r: dict, s: dict) -> str:
    """The text a padded row adds to its unpadded source, between the scene's context and the query."""
    head, _, last = s["prompt"].rpartition(rg.SEEN_SEPARATOR)
    context, query = last.rsplit("\n", 1)
    prefix = head + rg.SEEN_SEPARATOR + context
    assert r["prompt"].startswith(prefix) and r["prompt"].endswith(query)
    return r["prompt"][len(prefix):len(r["prompt"]) - len(query)]


def test_shape(rows: list[dict]) -> None:
    c = collections.Counter((r["construct"], r["register"]) for r in rows)
    regs = {"seen"} | {f"pad{p}-{k}" for p in pb.PADS for k in pb.CONDITIONS}
    assert {reg for _, reg in c} == regs
    assert c[("static", "seen")] == TEST.scenes * 3 and c[("set", "seen")] == TEST.set_per_cell * 10
    assert all(n == c[("static", "seen")] for (con, _), n in c.items() if con == "static")
    assert len({r["id"] for r in rows}) == len(rows)


def test_scene_leads_and_query_ends(rows: list[dict]) -> None:
    src = {_key(r): r for r in rows if r["pad"] == 0}
    for r in rows:
        s = src[_key(r)]
        fill = _filler(r, s)
        assert bool(fill.replace(pb.EOS, "").strip()) == bool(r["pad"])
        assert (r["candidates"], r.get("gold"), r.get("gold_set")) == (s["candidates"], s.get("gold"), s.get("gold_set"))


def test_filler_never_names_the_scene(rows: list[dict]) -> None:
    src = {_key(r): r for r in rows if r["pad"] == 0}
    for r in rows:
        if r["pad"]:
            assert not any(d.startswith("planted") for d in r["filler_docs"])
            words = pb.scene_words([src[_key(r)]])
            pat = re.compile(r"\b(" + "|".join(map(re.escape, sorted(words))) + r")\b", re.I)
            assert not pat.search(_filler(r, src[_key(r)]))


def test_scene_rows_share_one_filler_and_fillers_nest(rows: list[dict]) -> None:
    by: dict[tuple, set] = collections.defaultdict(set)
    for r in rows:
        if r["pad"]:
            by[(r["base_id"], r["pad"], r["filler"])].add(tuple(r["filler_docs"]))
    assert all(len(v) == 1 for v in by.values())
    for base in {k[0] for k in by}:
        seqs = [next(iter(by[(base, p, "span")])) for p in pb.PADS]
        for a, b in zip(seqs, seqs[1:]):
            assert b[: len(a)] == a


def test_docs_condition_separates_by_eos(rows: list[dict]) -> None:
    for r in rows:
        last = r["prompt"].rsplit(rg.SEEN_SEPARATOR, 1)[-1]
        if r["filler"] == "docs":
            assert last.count(pb.EOS) == len(r["filler_docs"]) + 1
        else:
            assert pb.EOS not in r["prompt"]


def test_filler_budget_grows_with_the_pad(rows: list[dict]) -> None:
    sizes: dict[str, list[int]] = collections.defaultdict(list)
    for r in rows:
        if r["filler"] == "span":
            sizes[_key(r)].append(r["filler_tokens"])
    for v in sizes.values():
        assert len(v) == len(pb.PADS) and v == sorted(v) and v[-1] > v[0]
        assert all(t <= p - pb.PROMPT_ALLOWANCE + 1 for t, p in zip(v, pb.PADS))


def test_register_reader_reads_every_pad_against_the_unpadded_scene(rows: list[dict]) -> None:
    scored = []
    for r in rows:
        gold = set(r["gold_set"]) if r["construct"] == "set" else {r["gold"]}
        scored.append({**r, "cand_lps": {c.strip(): (0.0 if c.strip() in gold else -3.0) for c in r["candidates"]}})
    res = readers.register(scored)
    for p in pb.PADS:
        for k in pb.CONDITIONS:
            reg = f"pad{p}-{k}"
            assert res[f"static/{reg}/4shot"]["lookup"] == 1.0
            assert "swap_addressee" not in res[f"static/{reg}/4shot"]
            assert res[f"set/{reg}/fair"]["hit1"] == 1.0
            assert f"static/swap_self/{reg}/4shot" in res["transfer"] and f"set/soft/{reg}/fair" in res["transfer"]


def test_pool_pin_is_enforced(tmp_path: Path) -> None:
    f = tmp_path / "pool.jsonl"
    f.write_text('{"id": "x", "text": "y", "tokens": 1}\n')
    with pytest.raises(ValueError, match="not the pinned padbind pool"):
        pb.load_pool(f)


def test_rebuilds_the_pin(tmp_path: Path) -> None:
    root = os.environ.get(bt.PROMPTS_ENV)
    pool = Path(root) / pb.POOL_FILE if root else None
    if pool is None or not pool.exists():
        pytest.skip(f"${bt.PROMPTS_ENV} does not hold {pb.POOL_FILE}")
    dest = write_rows(pb.padbind_rows("padbind-v1", pool), tmp_path / "padbind-v1.jsonl")
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == bt.BATTERIES["padbind-v1"].sha256
