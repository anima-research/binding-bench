"""Surface policies for every battery: rules that answer an item from its prompt's surface, with no model.

A policy reads the 0-shot prompt, split into a body and its final query line, and picks a candidate: the first, last
or most often mentioned one, the one nearest the query's content words in the body, one the query line itself
contains, or (in code) the literal on the last line assigning the queried variable. Riders adds structural rules
read off the item: the queried box's declared contents, and the object of the last put or overwrite. A policy
that cannot answer abstains and is scored at chance.

For each battery and slice, :func:`report` gives every policy's chance-corrected accuracy κ, (accuracy − chance) /
(1 − chance); the best policy; and the conflict share, the share of items the best policy gets wrong (abstentions
included). A model's agree / conflict split is its accuracy on the items the best policy gets right vs wrong;
:func:`picks` gives each item's pick per policy, keyed by the id a score row carries.

The situation battery has its own story-level policies (:mod:`binding_bench.situation.shortcuts`).

    python -m binding_bench.shortcuts [--prompts-dir DIR] [--batteries all] [--riders] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Mapping, Optional, Sequence

from binding_bench.binding import batteries as B

WORD = re.compile(r"[A-Za-z][A-Za-z'_]*")
STOP = frozenset({
    "the", "a", "an", "is", "was", "are", "has", "have", "had", "with", "who", "what", "now", "so", "right", "of",
    "in", "to", "it", "my", "your", "me", "you", "i", "and", "for", "on", "at", "by", "its", "holding", "holds",
    "then", "end", "contains", "held", "okay", "task", "print", "state", "box", "after", "that",
})
ASSIGN = re.compile(r"^\s*([A-Za-z_][\w\[\]\"']*)\s*=\s*(.+)$")


@dataclass(frozen=True)
class Probe:
    """One item as a policy sees it: the prompt split into body and query, the candidates, the gold."""
    id: str
    slice: str
    body: str
    query: str
    candidates: tuple[str, ...]
    gold: str
    extra: Mapping[str, str] = field(default_factory=dict)   # structural picks (riders), policy name -> candidate


def split_query(prompt: str) -> tuple[str, str]:
    """(body, query): the prompt's last line, or its last sentence when it is one line."""
    body, _, q = prompt.rstrip().rpartition("\n")
    if not body:
        body, _, q = prompt.rstrip().rpartition(". ")
    return body, q


def _pos(text: str, word: str) -> list[int]:
    return [m.start() for m in re.finditer(rf"(?<![\w']){re.escape(word)}(?![\w'])", text)]


def _first(p: Probe) -> Optional[str]:
    seen = [(min(_pos(p.body, c)), c) for c in p.candidates if _pos(p.body, c)]
    return min(seen)[1] if seen else None


def _last(p: Probe) -> Optional[str]:
    seen = [(max(_pos(p.body, c)), c) for c in p.candidates if _pos(p.body, c)]
    return max(seen)[1] if seen else None


def _most(p: Probe) -> Optional[str]:
    n = {c: len(_pos(p.body, c)) for c in p.candidates}
    top = max(n.values())
    best = [c for c in p.candidates if n[c] == top]
    return best[0] if top and len(best) == 1 else None


def _near_key(p: Probe) -> Optional[str]:
    """The candidate nearest (in characters) to the last body occurrence of any query content word."""
    keys = [w for w in WORD.findall(p.query) if w.lower() not in STOP and w not in p.candidates]
    anchors = [x for k in keys for x in _pos(p.body, k)]
    if not anchors:
        return None
    a = max(anchors)
    d = {c: min(abs(x - a) for x in _pos(p.body, c)) for c in p.candidates if _pos(p.body, c)}
    return min(d, key=lambda c: d[c]) if d else None


def _in_query(p: Probe, last: bool) -> Optional[str]:
    seen = [(x, c) for c in p.candidates for x in _pos(p.query, c)]
    return (max(seen) if last else min(seen))[1] if seen else None


def _last_assign(p: Probe) -> Optional[str]:
    """Code: the literal on the last body line that assigns the queried variable."""
    lines = p.body.splitlines()
    targets = {m.group(1) for ln in lines if (m := ASSIGN.match(ln))}
    qv = [t for t in targets if t in p.query]
    if not qv:
        return None
    q = max(qv, key=len)
    for ln in reversed(lines):
        m = ASSIGN.match(ln)
        if m and m.group(1) == q:
            hit = [(x, c) for c in p.candidates for x in _pos(m.group(2), c)]
            return max(hit)[1] if hit else None
    return None


Policy = Callable[[Probe], Optional[str]]
POLICIES: dict[str, Policy] = {
    "first_mentioned": _first,
    "last_mentioned": _last,
    "most_mentioned": _most,
    "nearest_to_query_key": _near_key,
    "first_in_query": lambda p: _in_query(p, False),
    "last_in_query": lambda p: _in_query(p, True),
    "last_assignment": _last_assign,
}
RIDERS_POLICIES: tuple[str, ...] = ("declared_contents", "last_put_object")

# Score-row field(s) a battery is sliced on, beside the pooled reading. Bindprobes slice on family and variant.
SLICES: dict[str, tuple[str, ...]] = {
    "joinlite-v1": ("qkind",), "swapdyn-v1": ("condition", "qkind"), "stage5-decomp-v1": ("variant",),
    "stage5-decomp-core": ("variant",), "frame-ladder-v1": ("rung",), "capacity-v1": ("K",),
    "fluid-v1-morph": ("rung",), "fluid-v1-algebra": ("rung",), "fluid-v1-rebind": ("register",),
    "fluid-v2-algebra": ("rung",), "fluid-v2-rebind": ("register",), "fluid-v1-seqind": ("rung",),
    "cryst-v1": ("rung",), "fluid-v2-supplied": ("family", "rung"), "formupdate-v1": ("condition",), "formupdate-v2": ("condition",),
    "formupdate-v3": ("condition", "register"),
}


def _row_probes(b: B.Battery, path: Path) -> Iterator[Probe]:
    for ln in path.open():
        r = json.loads(ln)
        gold = r.get("gold") if r.get("gold") is not None else r.get("gold_orig")
        if gold is None:
            raise ValueError(f"{b.name}: row {r.get('id')} has no gold")
        body, q = split_query(r["prompt"])
        sl = "|".join(f"{f}={r[f]}" for f in SLICES.get(b.name, ()) if f in r)
        yield Probe(r["id"], sl, body, q, tuple(c.strip() for c in r["candidates"]), str(gold).strip())


def _bp_probes(b: B.Battery, path: Path) -> Iterator[Probe]:
    from binding_bench.bindprobes.items import build_prompt, load_items
    for it in load_items(path):
        body, q = split_query(build_prompt(it, "0shot"))
        v = it.meta.get("variant") or it.meta.get("qkind") or ""
        sl = f"{it.family}|{v}" if v else it.family
        yield Probe(it.item_id, sl, body, q, tuple(c.strip() for c in it.candidates), it.gold.strip())


def battery_probes(b: B.Battery, root: Path) -> list[Probe]:
    path = B.path_for(b, root)
    return list((_bp_probes if b.kind == "bindprobes" else _row_probes)(b, path))


def riders_probes(lite_only: bool = False) -> list[Probe]:
    """The riders std-tier grid (generated, banked names and seed) in its four registers, 0-shot; or riders-lite's
    200 items."""
    from binding_bench.riders import items as R
    from binding_bench.riders import lite as RL
    its, _ = R.generate(list(R.BANKED_VERIFIED_NAMES), 200, 200, R.DEFAULT_SEED)    # the banked grid's recipe
    its = RL.select_items(its) if lite_only else [it for it in its if it.tier == "std"]
    out = []
    for it in its:
        puts = [o["obj"] for o in it.ops if o["kind"] in ("put", "overwrite")]
        extra = {"declared_contents": it.boxes0.get(it.query_box) or "",
                 "last_put_object": puts[-1] if puts else ""}
        for reg in ("code", "narrative", "screenplay"):      # chat shares the screenplay's lines
            body, q = split_query(R.build_prompt(it, reg, [], "raw")["prompt"])
            out.append(Probe(f"{it.item_id}|{reg}|0", f"{reg}|m={it.m}", body, q, tuple(it.candidates), it.gold,
                             extra))
    return out


def gold_distance(p: Probe) -> Optional[int]:
    """Characters from the gold's last mention in the body to the body's end (where the query starts); None when the
    body never names the gold."""
    pos = _pos(p.body, p.gold)
    return len(p.body) - max(pos) if pos else None


def picks(probes: Sequence[Probe]) -> dict[str, dict[str, Optional[str]]]:
    """``{probe id: {policy: pick or None}}``."""
    out: dict[str, dict[str, Optional[str]]] = {}
    for p in probes:
        row: dict[str, Optional[str]] = {k: f(p) for k, f in POLICIES.items()}
        row.update({k: (v or None) for k, v in p.extra.items()})
        out[p.id] = row
    return out


@dataclass
class SliceReading:
    n: int
    kappa: dict[str, float]
    best: str
    conflict_share: float


def _read(probes: Sequence[Probe], pk: Mapping[str, Mapping[str, Optional[str]]]) -> SliceReading:
    names = list(pk[probes[0].id])
    acc: dict[str, list[float]] = defaultdict(list)
    chance = [1 / len(p.candidates) for p in probes]
    for p, h in zip(probes, chance):
        for k in names:
            pick = pk[p.id].get(k)
            acc[k].append(h if pick not in p.candidates else float(pick == p.gold))
    hbar = sum(chance) / len(chance)
    kappa = {k: (sum(v) / len(v) - hbar) / (1 - hbar) for k, v in acc.items()}
    best = max(kappa, key=lambda k: kappa[k])
    wrong = sum(1 for p in probes if pk[p.id].get(best) != p.gold)
    return SliceReading(len(probes), kappa, best, wrong / len(probes))


def report(probes: Sequence[Probe]) -> dict[str, SliceReading]:
    """``{slice: reading}``; the pooled reading is under ``"all"``."""
    if not probes:
        raise ValueError("no probes")
    pk = picks(probes)
    groups: dict[str, list[Probe]] = defaultdict(list)
    for p in probes:
        groups["all"].append(p)
        if p.slice:
            groups[p.slice].append(p)
    return {k: _read(v, pk) for k, v in sorted(groups.items())}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts-dir", default=None)
    ap.add_argument("--batteries", default="all")
    ap.add_argument("--riders", action="store_true", help="also read the riders grid and riders-lite")
    ap.add_argument("--json", default="", help="write {battery: {slice: reading}} here")
    a = ap.parse_args(argv)
    sets: dict[str, list[Probe]] = {}
    if a.batteries:
        root = B.prompts_dir(a.prompts_dir)
        for b in B.resolve(a.batteries):
            try:
                sets[b.name] = battery_probes(b, root)
            except (B.PinError, OSError, ValueError) as e:
                print(f"skip {b.name}: {e}", file=sys.stderr)
    if a.riders:
        sets["riders-grid"] = riders_probes()
        sets["riders-lite"] = riders_probes(lite_only=True)
    out: dict[str, dict[str, dict]] = {}
    print("| battery | slice | n | best policy | κ | conflict share | runner-up |\n|---|---|---|---|---|---|---|")
    for name, probes in sets.items():
        rep = report(probes)
        out[name] = {k: vars(v) for k, v in rep.items()}
        for sl, r in rep.items():
            second = sorted(r.kappa, key=lambda k: -r.kappa[k])[1]
            print(f"| {name} | {sl} | {r.n} | {r.best} | {100 * r.kappa[r.best]:.0f} | {r.conflict_share:.2f} | "
                  f"{second} {100 * r.kappa[second]:.0f} |")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
