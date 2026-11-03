"""The bench-wide surface policies: each rule's pick on hand-built prompts, the report's κ and conflict share, and the
riders structural rules."""

from __future__ import annotations

import pytest

from binding_bench import shortcuts as S


def _probe(prompt: str, cands: tuple[str, ...], gold: str, pid: str = "x", sl: str = "") -> S.Probe:
    body, q = S.split_query(prompt)
    return S.Probe(pid, sl, body, q, cands, gold)


def test_split_query_takes_the_last_line_or_the_last_sentence():
    assert S.split_query("a b.\nc d") == ("a b.", "c d")
    one_line = "The tin holds the nail. So the tin holds the"
    assert S.split_query(one_line) == ("The tin holds the nail", "So the tin holds the")


def test_each_text_policy_picks_what_it_says():
    p = _probe("Ann has the cup. Bo has the pen. Bo drops the pen.\nRight now Ann is holding the",
               ("cup", "pen", "key"), "cup")
    picks = S.picks([p])["x"]
    assert picks["first_mentioned"] == "cup"
    assert picks["last_mentioned"] == "pen"
    assert picks["most_mentioned"] == "pen"
    assert picks["nearest_to_query_key"] == "cup"        # "Ann" is the query's content word
    assert picks["first_in_query"] is None


def test_last_assignment_reads_the_queried_variable_only():
    p = _probe('x = "cup"\ny = "pen"\nx = "key"\ny = "cup"\nprint(x)  #', ("cup", "pen", "key"), "key")
    assert S.picks([p])["x"]["last_assignment"] == "key"
    q = _probe("no code here\nwhat is it", ("cup", "pen"), "cup")
    assert S.picks([q])["x"]["last_assignment"] is None


def test_report_scores_abstentions_at_chance_and_counts_the_best_policys_misses():
    right = _probe("A cup.\nThe", ("cup", "pen"), "cup", "r", "s1")
    wrong = _probe("A pen.\nThe", ("cup", "pen"), "cup", "w", "s1")
    rep = S.report([right, wrong])
    assert set(rep) == {"all", "s1"}
    r = rep["all"]
    assert r.kappa["first_mentioned"] == pytest.approx(0.0)       # one right, one wrong at chance .5
    assert r.kappa["first_in_query"] == pytest.approx(0.0)        # abstains on both
    assert r.conflict_share == pytest.approx(0.5)
    with pytest.raises(ValueError):
        S.report([])


def test_riders_structural_rules_follow_the_item():
    probes = S.riders_probes(lite_only=True)
    assert len(probes) == 600 and {p.slice.split("|")[0] for p in probes} == {"code", "narrative", "screenplay"}
    m0 = [p for p in probes if p.slice.endswith("m=0")]
    assert m0 and all(p.extra["declared_contents"] == p.gold for p in m0)
    rep = S.report(probes)
    assert rep["code|m=0"].kappa["declared_contents"] == pytest.approx(1.0)


def test_gold_distance_counts_back_from_the_query():
    p = _probe("Ann has the cup. Bo has the pen.\nRight now Ann is holding the", ("cup", "pen"), "cup")
    assert S.gold_distance(p) == len("cup. Bo has the pen.")
    assert S.gold_distance(_probe("Ann waves.\nShe has the", ("cup", "pen"), "cup")) is None
