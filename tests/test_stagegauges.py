"""The stage-gauge generators reproduce their pinned batteries byte for byte, and refuse what they cannot build."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from binding_bench.binding import batteries as bt
from binding_bench.binding import stagegauges as sg


def _pinned(rows, name: str, tmp_path: Path) -> bool:
    return bt.sha256(sg.write_rows(rows, tmp_path / bt.BATTERIES[name].file)) == bt.BATTERIES[name].sha256


def test_joinlite_regenerates_its_pin(tmp_path):
    rows = sg.joinlite_rows()
    assert len(rows) == 1500 and _pinned(rows, "joinlite-v1", tmp_path)
    assert {r["qkind"] for r in rows} == {"third", "self", "addressee"}
    assert all(r["gold"] in {c.strip() for c in r["candidates"]} for r in rows)


def test_joinlite_refuses_an_unknown_verb():
    with pytest.raises(ValueError, match="holding or has"):
        sg.joinlite_rows(verb="owns")


def _deixis_items() -> Path:
    root = os.environ.get(bt.PROMPTS_ENV)
    src = Path(root or "/nonexistent") / bt.BATTERIES["bp-deixis-v2"].file
    if not src.exists():
        pytest.skip(f"${bt.PROMPTS_ENV} does not hold {bt.BATTERIES['bp-deixis-v2'].file}")
    return src


def test_the_swapdyn_chain_regenerates_its_pins(tmp_path):
    swap = sg.swapdyn_rows(_deixis_items())
    decomp = sg.decomp_rows(swap)
    core = sg.decomp_core(decomp)
    frame = sg.frame_rows(swap)
    assert (len(swap), len(decomp), len(core), len(frame)) == (1440, 4320, 2160, 2160)
    for rows, name in ((swap, "swapdyn-v1"), (decomp, "stage5-decomp-v1"), (core, "stage5-decomp-core"),
                       (frame, "frame-ladder-v1")):
        assert _pinned(rows, name, tmp_path), name


def _swap_row(prompt: str, gold: str, cands: list[str]) -> dict:
    return {"id": "x|0shot|orig", "arm": "orig", "prompt": prompt, "base_id": "x", "qkind": "self", "E": 1,
            "condition": "0shot", "candidates": [" " + c for c in cands], "gold_orig": gold,
            "gold_is_last_mention": False}


def test_decomp_rewrites_one_axis_and_round_trips():
    prompt = "A chat between Ana and Ben.\nAna: I have the cup.\nBen: The lamp is mine.\nBen: Here, take my lamp.\n" \
             "Ana: Right now I'm holding the"
    rows = sg.decomp_rows([_swap_row(prompt, "cup", ["cup", "lamp"])])
    assert rows == []      # Ana holds both objects after the transfer, so the gold is not unique: skipped
    prompt = "A chat between Ana and Ben.\nAna: I have the cup.\nAna: Here, take my cup.\nBen: Right now I'm holding the"
    rows = sg.decomp_rows([_swap_row(prompt, "cup", ["cup", "lamp"])])
    by = {r["variant"]: r["prompt"] for r in rows}
    assert set(by) == set(sg.DECOMP_VARIANTS) and by["deictic"] == prompt
    assert by["third"].endswith("Right now Ben is holding the") and "[Ana gives the cup to Ben.]" in by["named_transfer"]
    assert [r["variant"] for r in sg.decomp_core(rows)] == list(sg.DECOMP_CORE_VARIANTS)


def test_decomp_refuses_a_candidate_that_collides_with_an_extra_object():
    prompt = "A chat between Ana and Ben.\nAna: I have the kettle.\nBen: Right now you're holding the"
    with pytest.raises(ValueError, match="collide"):
        sg.decomp_rows([_swap_row(prompt, "kettle", ["kettle", "cup"])])


def test_frame_ladder_adds_one_cue_per_rung():
    prompt = "A chat between Ana and Ben.\nAna: I have the cup.\nAna: Here, take my cup.\nBen: Right now I'm holding the"
    by = {r["rung"]: r["prompt"] for r in sg.frame_rows([_swap_row(prompt, "cup", ["cup", "lamp"])])}
    assert set(by) == set(sg.FRAME_RUNGS) and by["r0"] == prompt
    assert by["r1"].startswith("A chat between Ana and Ben, who speak only to each other.")
    assert "Ana (to Ben): Here, take my cup." in by["r2"] and "Ana (says): Here, take my cup." in by["r2null"]
    for rung in ("r1", "r2", "r2null"):
        assert by[rung].endswith("\nBen: Right now I'm holding the")    # the query line is R0's
    assert "[Ana gives the cup to Ben.]" in by["r3"] and by["r4"].endswith("Right now Ben is holding the")


def test_frame_ladder_skips_three_shot_rows():
    row = _swap_row("A chat between Ana and Ben.\nAna: I have the cup.\nBen: Right now you're holding the", "cup",
                    ["cup", "lamp"])
    row["condition"] = "3shot"
    assert sg.frame_rows([row]) == []
