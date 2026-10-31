"""riders-lite: a small, balanced slice of the rider grid for in-loop scoring.

A training loop scores this slice on the live weights at every eval point, so the
selection is deterministic and the same items appear in every register. Pieces:

* :func:`select_items` — N std-tier items, seeded, round-robin over the (n, m)
  difficulty cells so cells stay balanced; the SAME items in every register
  (paired register comparisons).
* :func:`reletter_shots` — the exemplar shots with their box letters remapped
  (F->A, G->B, H->C by default) to letters no std item uses, re-validated with
  the item simulator. The ``3shot_relet`` condition asks whether 3-shot gains
  come from the worked procedure or from copying the exemplars' box letters.
* :func:`fair_shots` — the ``fair`` condition's exemplars: per item, :data:`FAIR_K` shots
  from cells other than the item's own (rotated so every cell is an exemplar
  equally often), jointly covering every op kind, with box letters the items never
  use and objects from the disjoint shot pool. Cell-matched exemplars can teach the
  answer's role rather than the procedure; these cannot. ``fair`` is opt-in: it is
  not in :data:`CONDITIONS`, so the in-loop default slice is unchanged.
* :func:`build_jobs` — one ``/logprobs``-shaped request per (item, register,
  condition).
* :func:`summarize` — accuracy, chance-corrected accuracy (``aggregate``'s
  formula), mean margin and coarse error-type shares per register x condition,
  plus the shot gaps; keys are ``riders/{register}/{condition}/{metric}``.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from binding_bench.riders.aggregate import cc_point
from binding_bench.riders.errors import COARSE_TYPES
from binding_bench.riders.grid import build_payload_item
from binding_bench.riders.items import BOX_LABELS, CELLS, N_SHOTS, Item, Op, _sim, gen_item

REGISTERS: tuple[str, ...] = ("narrative", "screenplay", "chat", "code")
PROSE: tuple[str, ...] = ("narrative", "screenplay", "chat")
CONDITIONS: tuple[str, ...] = ("0shot", "3shot", "3shot_relet")
COND_NSHOTS: dict[str, int] = {"0shot": 0, "3shot": N_SHOTS, "3shot_relet": N_SHOTS}
RELETTER_MAP: dict[str, str] = {"F": "A", "G": "B", "H": "C"}
DEFAULT_PER_REGISTER = 200
DEFAULT_SEED = 0
FAIR = "fair"
FAIR_K = 4
FAIR_LETTERS: tuple[str, ...] = tuple("ABCDEJNQSTUVWXYZ")     # none is in BOX_LABELS
FAIR_CELLS: tuple[tuple[int, int], ...] = tuple(c for c in CELLS if c[1] >= 1)   # an exemplar works an update
OP_KINDS: tuple[str, ...] = ("swap", "move", "put", "overwrite")


@dataclass(frozen=True)
class Job:
    """One scoring request + where its row goes."""
    id: str
    prompt: str
    candidates: list[str]
    item_index: int       # index into the selected item list
    renderer: str
    condition: str
    nshots: int


# ── Selection ──────────────────────────────────────────────────────────────────

def select_items(items: Sequence[Item], n: int = DEFAULT_PER_REGISTER,
                 seed: int = DEFAULT_SEED,
                 cells: Sequence[tuple[int, int]] = CELLS) -> list[Item]:
    """``n`` std-tier items, round-robin over ``cells`` (each cell shuffled with
    ``seed``); cells that run out drop out of the rotation. Deterministic."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    rng = random.Random(seed)
    pools: list[list[Item]] = []
    for cell in cells:
        pool = [it for it in items if it.tier == "std" and (it.n, it.m) == tuple(cell)]
        rng.shuffle(pool)
        if pool:
            pools.append(pool)
    if not pools:
        raise ValueError("no std-tier items in any cell")
    out: list[Item] = []
    k = 0
    while len(out) < n and any(k < len(p) for p in pools):
        for p in pools:
            if k < len(p) and len(out) < n:
                out.append(p[k])
        k += 1
    return out


# ── Re-lettered shots ──────────────────────────────────────────────────────────

def _check_sim(item: Item) -> None:
    final = _sim(item.boxes0, [Op(**o) for o in item.ops])
    if final != item.final:
        raise ValueError(f"{item.item_id}: simulated final {final} != stored {item.final}")
    if final.get(item.query_box) != item.gold:
        raise ValueError(f"{item.item_id}: final[{item.query_box}]="
                         f"{final.get(item.query_box)!r} != gold {item.gold!r}")


def reletter_item(item: Item, mapping: Mapping[str, str] = RELETTER_MAP) -> Item:
    """``item`` with box letters remapped (boxes0, ops i/j, final, query_box)."""
    used = set(item.boxes0) | set(item.final) | {item.query_box}
    used |= {op[k] for op in item.ops for k in ("i", "j") if op.get(k) is not None}
    missing = used - set(mapping)
    if missing:
        raise ValueError(f"{item.item_id}: letters {sorted(missing)} not in the remap")
    ops = []
    for op in item.ops:
        o = dict(op)
        o["i"] = mapping[o["i"]]
        if o.get("j") is not None:
            o["j"] = mapping[o["j"]]
        ops.append(o)
    new = Item(**{**asdict(item),
                  "item_id": f"{item.item_id}_relet",
                  "boxes0": {mapping[b]: v for b, v in item.boxes0.items()},
                  "ops": ops,
                  "final": {mapping[b]: v for b, v in item.final.items()},
                  "query_box": mapping[item.query_box]})
    _check_sim(new)
    return new


def reletter_shots(shots: Sequence[Item], items: Sequence[Item] = (),
                   mapping: Mapping[str, str] = RELETTER_MAP) -> list[Item]:
    """Re-lettered copies of ``shots``, each validated (final[query_box] == gold).

    Raises if a target letter is one any of ``items`` uses (the point of the
    condition is letters the queried items never show)."""
    item_letters = {b for it in items for b in it.boxes0}
    clash = item_letters & set(mapping.values())
    if clash:
        raise ValueError(f"re-letter targets {sorted(clash)} are used by the items")
    for s in shots:
        _check_sim(s)
    return [reletter_item(s, mapping) for s in shots]


# ── Fair shots ─────────────────────────────────────────────────────────────────

def _fair_cells(item: Item, t: int) -> list[tuple[int, int]]:
    """FAIR_K exemplar cells for the t-th item: the next ones in FAIR_CELLS's rotation, skipping the item's own."""
    out: list[tuple[int, int]] = []
    pos = t * FAIR_K
    while len(out) < FAIR_K:
        c = FAIR_CELLS[pos % len(FAIR_CELLS)]
        pos += 1
        if c != (item.n, item.m) and c not in out:
            out.append(c)
    return out


def fair_shots(items: Sequence[Item], shot_pool: Sequence[str], seed: int = DEFAULT_SEED,
               max_tries: int = 200) -> list[list[Item]]:
    """Per item, FAIR_K exemplars (module doc). Deterministic in (seed, item_id); each shot re-validated with the
    simulator. Refuses an item whose exemplars cannot cover every op kind within ``max_tries`` draws."""
    if set(FAIR_LETTERS) & set(BOX_LABELS):
        raise ValueError("fair letters overlap the item letters")
    out: list[list[Item]] = []
    for t, it in enumerate(items):
        cells = _fair_cells(it, t)
        for attempt in range(max_tries):
            rng = random.Random(f"{seed}:{it.item_id}:fair:{attempt}")
            shots = []
            for k, (n, m) in enumerate(cells):
                base = gen_item(rng, n, m, list(shot_pool), False, f"{it.item_id}_fair{k}")
                mapping = dict(zip(BOX_LABELS, rng.sample(FAIR_LETTERS, len(BOX_LABELS))))
                shot = reletter_item(base, mapping)
                answerable = sorted(b for b, v in shot.final.items() if v is not None)
                q = rng.choice(answerable)
                shot = Item(**{**asdict(shot), "item_id": f"{it.item_id}_fair{k}", "query_box": q,
                               "gold": shot.final[q]})
                _check_sim(shot)
                shots.append(shot)
            if {op["kind"] for s in shots for op in s.ops} >= set(OP_KINDS):
                out.append(shots)
                break
        else:
            raise ValueError(f"{it.item_id}: no fair exemplar set covers every op kind in {max_tries} draws")
    return out


# ── Jobs ───────────────────────────────────────────────────────────────────────

def build_jobs(items: Sequence[Item], shots: Sequence[Item],
               relet_shots: Optional[Sequence[Item]],
               registers: Sequence[str] = REGISTERS,
               conditions: Sequence[str] = CONDITIONS,
               fair: Optional[Sequence[Sequence[Item]]] = None) -> list[Job]:
    """Every (register, condition, item) request; koto prompts (chat = ChatML). ``fair`` (from :func:`fair_shots`, one
    list per item) is needed for the ``fair`` condition."""
    jobs: list[Job] = []
    for cond in conditions:
        if cond == FAIR:
            if fair is None or len(fair) != len(items):
                raise ValueError("condition 'fair' needs fair_shots(items, …), one list per item")
            nshots = FAIR_K
        elif cond not in COND_NSHOTS:
            raise ValueError(f"unknown condition {cond!r}; choose from {CONDITIONS + (FAIR,)}")
        else:
            nshots = COND_NSHOTS[cond]
        shot_set = relet_shots if cond == "3shot_relet" else shots
        if cond != FAIR and shot_set is None:
            raise ValueError(f"condition {cond!r} needs shots")
        for rd in registers:
            for k, it in enumerate(items):
                these = list(fair[k]) if cond == FAIR else list(shot_set)  # type: ignore[index, arg-type]
                payload, _ = build_payload_item(it, rd, nshots, these, "koto")
                jobs.append(Job(id=f"{it.item_id}|{rd}|{cond}", prompt=payload["prompt"],
                                candidates=payload["candidates"], item_index=k,
                                renderer=rd, condition=cond, nshots=nshots))
    return jobs


# ── Summary ────────────────────────────────────────────────────────────────────

def _cell_metrics(rows: list[Mapping[str, Any]]) -> dict[str, float]:
    acc = float(np.mean([r["gold_rank_cand"] == 1 for r in rows]))
    out = {"acc": acc, "cc_acc": cc_point(list(rows)),
           "margin": float(np.mean([r["margin"] for r in rows])), "n": float(len(rows))}
    wrong = [r for r in rows if r["gold_rank_cand"] != 1]
    for t in COARSE_TYPES:
        out[f"err_{t}"] = (sum(r.get("error_type") == t for r in wrong) / len(wrong)
                           if wrong else 0.0)
    return out


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """``riders/{register}/{condition}/{acc,cc_acc,margin,n,err_*}`` per register
    plus the ``prose`` pool (narrative+screenplay+chat); ``riders/{register}/shot_gap``
    = cc(3shot) - cc(0shot) and ``relet_gap`` = cc(3shot_relet) - cc(3shot)."""
    by: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        by[(r["renderer"], r["condition"])].append(r)
        if r["renderer"] in PROSE:
            by[("prose", r["condition"])].append(r)
    out: dict[str, float] = {}
    regs = sorted({rd for rd, _ in by})
    for rd in regs:
        cc: dict[str, float] = {}
        for cond in CONDITIONS + (FAIR,):
            sub = by.get((rd, cond))
            if not sub:
                continue
            m = _cell_metrics(list(sub))
            cc[cond] = m["cc_acc"]
            for k, v in m.items():
                out[f"riders/{rd}/{cond}/{k}"] = v
        if "3shot" in cc and "0shot" in cc:
            out[f"riders/{rd}/shot_gap"] = cc["3shot"] - cc["0shot"]
        if "3shot_relet" in cc and "3shot" in cc:
            out[f"riders/{rd}/relet_gap"] = cc["3shot_relet"] - cc["3shot"]
        if FAIR in cc and "0shot" in cc:
            out[f"riders/{rd}/fair_gap"] = cc[FAIR] - cc["0shot"]
    return out
