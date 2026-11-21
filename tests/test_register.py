"""register-v1: the generator rebuilds the pin byte for byte, every scene's registers agree on candidates and gold, the
seen rendering is the source battery's, and the reader reads a binder as binding in every register."""

from __future__ import annotations

import collections
import hashlib
import json
import math
from pathlib import Path

import pytest

from binding_bench.binding import batteries as bt
from binding_bench.binding import readers
from binding_bench.binding import register as rg
from binding_bench.binding.stagegauges import write_rows


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    return rg.register_rows()


def test_rebuilds_the_pin(rows: list[dict], tmp_path: Path) -> None:
    dest = write_rows(rows, tmp_path / "register-v1.jsonl")
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == bt.BATTERIES["register-v1"].sha256


def test_shape(rows: list[dict]) -> None:
    c = collections.Counter((r["construct"], r["register"], r["condition"]) for r in rows)
    for reg in rg.REGISTERS:
        assert c[("static", reg, "3shot")] == rg.STATIC_SCENES * 5
        assert c[("dynamic", reg, "3shot")] == 2400
        assert c[("set", reg, "fair")] == 600
    assert c[("static", "seen", "0shot")] == rg.STATIC_SCENES * 5 and c[("dynamic", "seen", "0shot")] == 2400
    assert len({r["id"] for r in rows}) == len(rows)


def test_twins_agree(rows: list[dict]) -> None:
    grp: dict[tuple, dict[str, tuple]] = collections.defaultdict(dict)
    for r in rows:
        if r["condition"] in ("3shot", "fair"):
            k = (r["construct"], r["base_id"], r.get("qkind"), r.get("arm"))
            grp[k][r["register"]] = (tuple(r["candidates"]), r.get("gold", r.get("gold_orig")),
                                     r.get("gold_is_last_mention"))
    assert grp and all(set(v) == set(rg.REGISTERS) and len(set(v.values())) == 1 for v in grp.values())


def test_unseen_registers_carry_no_seen_scaffolding(rows: list[dict]) -> None:
    for r in rows:
        if r["register"] == "usyntax":
            assert "A chat between" not in r["prompt"] and "holding the" not in r["prompt"]
            assert ": " not in r["prompt"] and "* * *" not in r["prompt"]
        if r["register"] == "utag":
            assert "A chat between" not in r["prompt"] and "* * *" not in r["prompt"]


def _score(rows: list[dict]) -> list[dict]:
    """A model that puts most of its mass on the gold (dynamic: the gold for the query's speaker) in every register."""
    out = []
    for r in rows:
        if r["construct"] == "set":
            gold = set(r["gold_set"])
        elif r["construct"] == "dynamic":
            gold = {r["gold_orig"]} if r["arm"] == "orig" else set()
        else:
            gold = {r["gold"]}
        lps = {c.strip(): (0.0 if c.strip() in gold else -3.0) for c in r["candidates"]}
        out.append({**r, "cand_lps": lps})
    return out


def test_reader_reads_a_binder(rows: list[dict]) -> None:
    sample = [r for r in rows if r["base_id"] in {x["base_id"] for x in rows[::97]}]
    res = readers.register(_score(sample))
    for reg in rg.REGISTERS:
        assert res[f"static/{reg}/3shot"]["swap_addressee"]["shift"] > 0.5
        assert res[f"dynamic/{reg}/3shot"]["swap_addressee"]["shift"] > 0.5
        assert res[f"set/{reg}/fair"]["hit1"] == 1.0
    t = res["transfer"]["static/swap_addressee/usyntax/3shot"]
    assert math.isclose(t["ratio"], 1.0, abs_tol=1e-9) and abs(t["diff"]) < 1e-9
    assert "dynamic/swap_self/utag/3shot" in res["transfer"]


def test_ratio_undefined_when_seen_does_not_bind(rows: list[dict]) -> None:
    sample = [r for r in rows if r["construct"] == "static" and r["scene"] < 40]
    flat = [{**r, "cand_lps": {c.strip(): 0.0 for c in r["candidates"]}} for r in sample]
    t = readers.register(flat)["transfer"]["static/swap_addressee/utag/3shot"]
    assert t["ratio"] is None and "undefined" in t["ratio_note"]


def test_rows_are_json(rows: list[dict]) -> None:
    json.dumps(rows[0], ensure_ascii=False)
