"""Situation-model battery: the lexicon refuses what doesn't fit, the simulator refuses impossible stories, every arm
has its registered property, and renderings differ only in wording."""

from __future__ import annotations

import random
import re

import pytest

from binding_bench import lexicon as lx
from binding_bench.situation import items as si
from binding_bench.situation.world import Event, StoryError, check_story


def test_lexicon_refuses_bad_entries_and_unfit_phrases():
    with pytest.raises(lx.LexiconError):
        lx.Noun("glue", "object", count=True, article=None)
    with pytest.raises(lx.LexiconError):
        lx.Phrase("x {a}", frozenset({"no_such_tag"}))
    with pytest.raises(lx.LexiconError):
        lx.PhraseSet("s", (lx.Phrase("{a} x"), lx.Phrase("{b} x")))
    ps = lx.PhraseSet("s", (lx.Phrase("{a} now", frozenset({"implies_change"})), lx.Phrase("{a} so")))
    assert ps.fitting(forbid={"implies_change"}) == (1,)
    with pytest.raises(lx.LexiconError):
        ps.pick(random.Random(0), forbid={"implies_change"}, index=0)
    with pytest.raises(lx.LexiconError):
        lx.check_pools({"a": [lx.Name("Tom", "m")], "b": [lx.Name("tom", "m")]})


def test_simulator_refuses_impossible_stories():
    ppl, pl = ["A", "B"], ["x", "y", "z"]
    with pytest.raises(StoryError):
        check_story(ppl, pl, [Event("put", "A", place="x"), Event("move", "B", place="y", src="z")])
    with pytest.raises(StoryError):
        check_story(ppl, pl, [Event("put", "A", place="x"), Event("leave", "A"), Event("move", "A", place="y", src="x")])
    with pytest.raises(StoryError):
        check_story(ppl, pl, [Event("put", "A", place="x"), Event("reveal", "B", place="y")])
    st = check_story(ppl, pl, [Event("put", "A", place="x"), Event("leave", "A"), Event("move", "B", place="y", src="x"),
                               Event("return", "A")])
    assert (st.world, st.belief["A"], st.belief["B"]) == ("y", "x", "y")


@pytest.fixture(scope="module")
def items():
    return si.generate(40, seed=7)


def test_every_item_passes_the_gate_and_arms_hold(items):
    for it in items:
        si.gate(it)
    by = {a: [i for i in items if i.arm == a] for a in si.ARMS}
    assert all(not i.meta["gold_is_last_place_mention"] for i in by["R"])
    assert all(i.meta["n_changes"] == 1 for i in by["N"])
    for i in by["B"]:
        bel = next(p for p in i.probes if p.kind == "belief")
        assert bel.gold != i.meta["world"]
    for i in by["D"]:
        assert {p.kind for p in i.probes} == {"situation", "belief", "knot"}
        assert next(p for p in i.probes if p.kind == "belief").gold != i.meta["world"]
    for arm in ("U", "N", "R"):
        assert all(next(p for p in i.probes if p.kind == "belief").gold == i.meta["world"] for i in by[arm])
    for i in by["L"]:
        assert {p.kind for p in i.probes} == {"situation", "knot"}


def test_no_position_or_role_rule_finds_the_gold(items):
    """The pilot's lesson: the knot gold is not tied to the second-named person, and the belief gold is not always the
    first place the object was put."""
    knots = [next(p for p in i.probes if p.kind == "knot") for i in items]
    second = sum(k.gold == k.candidates[1] for k in knots) / len(knots)
    named_second = sum(next(p for p in i.probes if p.kind == "knot").gold == i.people[1] for i in items) / len(items)
    assert .3 < second < .7 and .3 < named_second < .7
    bel = [(i, next(p for p in i.probes if p.kind == "belief")) for i in items if i.arm != "L"]
    first_put = sum(b.gold == next(e for e in i.events if e["kind"] == "put")["place"] for i, b in bel) / len(bel)
    world = sum(b.gold == i.meta["world"] for i, b in bel) / len(bel)
    assert first_put < .85 and .4 < world < .8


def test_renderings_differ_only_in_wording(items):
    for it in items[:30]:
        texts = [si.render_story(it, r) for r in range(si.N_RENDERINGS)]
        assert len(set(texts)) == si.N_RENDERINGS
        for t in texts:
            mentioned = [w for w in re.findall(r"[a-z]+", t) if w in it.places]
            assert set(mentioned) == set(it.places)
            for w in it.people + [it.obj]:
                assert w in t


def test_places_share_one_preposition_and_rows_are_complete(items):
    assert {p.prep for p in si.PLACES} == {"in"}
    rows = si.rows(items)
    assert len(rows) == sum(len(i.probes) for i in items) * si.N_RENDERINGS
    for r in rows:
        assert " " + r["gold"] in r["candidates"] and not r["prompt"].endswith(" ")


def test_cli_gen_writes_gated_items_and_rows(tmp_path):
    import json

    from binding_bench.situation.__main__ import main as cli_main
    assert cli_main(["gen", "--out", str(tmp_path), "--per-arm", "12"]) == 0
    items = [json.loads(ln) for ln in (tmp_path / "items.jsonl").read_text().splitlines()]
    rows = [json.loads(ln) for ln in (tmp_path / "rows.jsonl").read_text().splitlines()]
    assert len(items) == 12 * len(si.ARMS)
    assert len(rows) == 2 * sum(len(i["probes"]) for i in items) * si.N_RENDERINGS      # 0shot + fair


def test_transcript_rendering_and_fair_rows(items):
    for it in items[:10]:
        t = si.render_story(it, si.NARRATION)
        lines = t.split("\n")
        assert lines[0].startswith("A chat between ") and len(lines) == len(it.events) + 1
        assert all(re.match(r"^(\w+): ", ln) and ln.split(":")[0] in it.people for ln in lines[1:])
    fair = si.fair_rows(items)
    assert len(fair) == len(si.rows(items))
    by = {(r["item_id"], r["probe"]): r for r in fair if r["rendering"] == 0}
    for (iid, kind), r in list(by.items())[:40]:
        it = next(i for i in items if i.item_id == iid)
        shots = r["prompt"].split(si.SEPARATOR)[:-1]
        assert len(shots) == si.FAIR_K
        words = {w for sh in shots for w in re.findall(r"[A-Za-z]+", sh)}
        assert not words & set(it.people + [it.obj] + it.places), iid


def test_nli_hypotheses_cover_every_candidate(items, tmp_path):
    import json
    from dataclasses import asdict

    from binding_bench.situation import nli
    for it in items[:10]:
        for pr in it.probes:
            hs = nli.hypotheses(it, pr)
            assert len(hs) == len(pr.candidates) and all(c in h for c, h in zip(pr.candidates, hs))
    f = tmp_path / "items.jsonl"
    f.write_text("".join(json.dumps(asdict(i)) + "\n" for i in items[:5]))
    assert [i.item_id for i in nli.load_items(f)] == [i.item_id for i in items[:5]]


def test_put_question_names_the_first_place_and_only_a_put_it_there(items):
    for it in items:
        k = next(p for p in it.probes if p.kind == "knot")
        if k.lead != "q_knot_put":
            continue
        first = next(e for e in it.events if e["kind"] == "put")
        assert k.slots["p"] == first["place"] and k.gold == first["actor"]
        assert not any(e["kind"] in ("put", "move") and e["place"] == first["place"] and e["actor"] != k.gold
                       for e in it.events)
