"""The stage map: every battery a stage is read from is named, and a change to a stage gauge's bytes is never silent."""

from __future__ import annotations

from binding_bench import registry as rg
from binding_bench.binding import batteries as bt


def test_stage_map_names_real_batteries_and_stages():
    for u in rg.STAGE_MAP:
        assert u.battery in bt.BATTERIES, u.battery
        assert u.stages and set(u.stages) <= {1, 2, 3, 4, 5}, u
    covered = {s for u in rg.STAGE_MAP if u.role == "gauge" for s in u.stages}
    assert covered == {1, 2, 3, 4, 5}, "every stage needs a gauge"


def test_stage_gauge_pins_are_current():
    gauges = {u.battery for u in rg.STAGE_MAP if u.role == "gauge"}
    assert set(rg.STAGE_GAUGE_PINS) == gauges
    for name, pin in rg.STAGE_GAUGE_PINS.items():
        assert bt.BATTERIES[name].sha256 == pin, (
            f"stage gauge {name} changed: stages "
            f"{sorted({s for u in rg.STAGE_MAP if u.battery == name for s in u.stages})} can move. Rerun the stage "
            f"readout on the panel, record the before / after, then update STAGE_GAUGE_PINS")


def test_previous_pins_differ_from_the_current_and_stay_at_other_paths():
    for name, pins in bt.PREVIOUS_PINS.items():
        cur = bt.BATTERIES[name]
        for p in pins:
            assert p.sha256 != cur.sha256 and p.file != cur.file, name
            assert (p.render_sha256 is None) == (cur.kind == "rows"), name
    assert set(rg.STAGE_GAUGE_PINS) <= set(bt.PREVIOUS_PINS)   # every gauge's earlier bytes stay citable
