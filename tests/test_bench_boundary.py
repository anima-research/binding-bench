"""The benchmark's package boundary and its registry.

Nothing in ``binding_bench`` imports kotodama, or any other model codebase: checkpoint formats other than ``hf:``
arrive as loader plugins (``binding_bench.binding.models``). Every pinned battery has a registry row, and every
registry row names real batteries and constructs.
"""

from __future__ import annotations

import ast
from pathlib import Path

from binding_bench import registry as rg
from binding_bench.binding import batteries as bt
from binding_bench.bindprobes.items import CONDITIONS as BP_CONDITIONS

PKG = Path(__file__).resolve().parents[1] / "src" / "binding_bench"


def _absolute_imports(path: Path) -> list[tuple[int, str]]:
    out = []
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Import):
            out += [(node.lineno, a.name) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            out.append((node.lineno, node.module))
    return out


def test_the_bench_imports_no_model_codebase():
    files = sorted(PKG.rglob("*.py"))
    assert files
    leaks = [f"{f.relative_to(PKG.parent)}:{line} imports {name}" for f in files
             for line, name in _absolute_imports(f) if name == "kotodama" or name.startswith("kotodama.")]
    assert not leaks, "the benchmark imports kotodama:\n" + "\n".join(leaks)


def test_every_pinned_battery_has_a_registry_row():
    carried = {b for e in rg.REGISTRY.values() for b in e.batteries}
    assert set(bt.BATTERIES) <= carried, f"no registry row: {sorted(set(bt.BATTERIES) - carried)}"
    assert carried <= set(bt.BATTERIES), f"unknown batteries in the registry: {sorted(carried - set(bt.BATTERIES))}"
    assert all(rg.evals_for(b) for b in bt.BATTERIES)


def test_registry_rows_are_well_formed():
    for e in rg.REGISTRY.values():
        assert e.construct in rg.CONSTRUCTS, e.name
        assert e.recipe and e.clean, e.name
        if e.source == "bindprobes":
            assert set(e.clean) <= set(BP_CONDITIONS), e.name
            assert all(bt.BATTERIES[b].kind == "bindprobes" for b in e.batteries), e.name
        elif e.source == "binding":
            assert all(bt.BATTERIES[b].kind == "rows" for b in e.batteries), e.name
        else:
            assert not e.batteries, e.name
        for f in e.flags:
            assert f.reading and f.finding and f.panel and f.test, e.name
    assert rg.REGISTRY["riders"].source == "riders"


def test_bp_v32_recipe_names_its_battery_families():
    assert set(rg.BP_V32_RECIPE["per_cell"]) == set(bt.BATTERIES["bp-v3.2"].families)


def test_panel_reliability_names_real_evals_and_slots_a_large_model_check_for_panel_floors():
    assert set(r.eval for r in rg.PANEL_RELIABILITY.values()) <= set(rg.REGISTRY)
    assert all(0 <= r.floor_share <= 1 and -1 <= r.reliability <= 1 for r in rg.PANEL_RELIABILITY.values())
    floored = {k for k, r in rg.PANEL_RELIABILITY.items() if r.floor_share > 0.5}
    assert floored == {"deixis-v2 addressee, deictic", "swapdyn addressee, 0shot"}
    assert all(rg.PANEL_RELIABILITY[k].large_model_check for k in floored)
