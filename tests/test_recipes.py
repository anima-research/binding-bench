"""Every pinned battery has a way back to its bytes: a generator here, the stage-gauge chain, or a historical recipe."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from binding_bench import registry as rg
from binding_bench.binding import batteries as bt
from tools import replay_recipe as rr

KOTODAMA_ENV = "BINDING_BENCH_KOTODAMA_CLONE"


def test_every_battery_names_a_recipe():
    for e in rg.REGISTRY.values():
        assert "frozen" not in e.recipe and e.recipe, e.name
    named = {b for e in rg.REGISTRY.values() for b in e.batteries}
    historical = {k for k in rg.HISTORICAL_RECIPES}
    assert historical <= named and all(bt.BATTERIES[k].kind == "bindprobes" for k in historical)


def test_historical_recipes_are_well_formed():
    for name, r in rg.HISTORICAL_RECIPES.items():
        assert len(r.kotodama_commit) >= 7 and int(r.kotodama_commit, 16) >= 0, name
        assert r.per_cell and all(n > 0 for n in r.per_cell.values()), name
        fams = bt.BATTERIES[name].families
        assert not fams or tuple(r.per_cell) == fams, name
        assert all(len(c) == 3 for c in r.keep_cells), name


def test_keep_filters_cells_in_order():
    lines = [json.dumps({"cell": c, "n": i}) + "\n" for i, c in
             enumerate([[3, 1, "owner"], [2, 1, "self"], [3, 2, "owner"], [3, 3, "owner"], [3, 1, "owner"]])]
    kept = rr.keep(lines, ((3, 1, "owner"), (3, 2, "owner")))
    assert [json.loads(x)["n"] for x in kept] == [0, 2, 4]
    assert rr.keep(lines, ()) == lines


def test_replay_reproduces_a_historical_pin(tmp_path):
    clone = os.environ.get(KOTODAMA_ENV)
    if not clone or not (Path(clone) / ".git").exists():
        pytest.skip(f"${KOTODAMA_ENV} does not name a kotodama clone")
    assert rr.main(["--kotodama", clone, "--battery", "bp-deixis-v2", "--out", str(tmp_path)]) == 0
