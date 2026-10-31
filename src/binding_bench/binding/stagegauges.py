"""The stage-gauge batteries' generators: joinlite-v1, swapdyn-v1, stage5-decomp-v1 and stage5-decomp-core.

Each one reproduces its pinned file byte for byte (``tests/test_stagegauges.py``):

* :func:`joinlite_rows` (joinlite-v1) — a JOIN ruler with the lookup hop made trivial. Ownership is stated once, in
  narration, with no transfers, so the only thing left to fail is the I / you → name translation. Each scene is
  queried as ``third`` (``Now X is holding the``: the lookup alone), ``self`` (``S: Right now I'm holding the``) and
  ``addressee`` (P2 ``S: Right now you're holding the``; P3 with a vocative), and the self and P2 addressee queries
  also run with the query speaker swapped, which flips the gold. Candidates are every object in the scene, including
  the table object; the addressee target's object cycles through the scene sentence's slots across scenes, so "gold
  = last-mentioned object" sits near chance.
* :func:`swapdyn_rows` (swapdyn-v1) — the speaker-swap counterfactual on deixis v2's P2 deictic self / addressee
  items (the bp-deixis-v2 battery), under 0shot and 3shot: two prompts that differ only in the name on the final
  query line. A model that binds I / you to the speaker label moves mass off the original gold under the swap.
* :func:`decomp_rows` (stage5-decomp-v1) — every swapdyn ``orig`` item rewritten along one axis at a time:
  ``deictic`` (the original, round-trip checked), ``third`` (no deixis anywhere), ``vocative`` (the query names the
  addressee), ``named_transfer`` (the transfer as a stage direction), ``named_query`` (a narrated query) and
  ``three_vocative`` (a third participant joins, so "you" is not simply "the other one"). A simulator re-derives
  the gold from the parse, and an item whose target does not hold exactly the original gold is skipped.
* :func:`decomp_core` (stage5-decomp-core) — the deictic, third and named_transfer variants of stage5-decomp-v1.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

Row = dict[str, Any]

# ---- joinlite --------------------------------------------------------------------------------------------------------

JOINLITE_SEED = 7
JOINLITE_SCENES = 300

FILLERS = ("It is getting late.", "The coffee here is good.", "That was a long meeting.", "Nice weather today.",
           "I like this room.", "The train was slow this morning.", "We should start soon.", "It is quiet in here.",
           "I had lunch already.", "The music is nice.", "Good to see everyone.", "I almost forgot about tonight.")


def _scene_sentence(holders: list[tuple[str, str]], table: str, rng: random.Random, gold_pos: int,
                    gold_obj: str) -> tuple[str, list[str]]:
    """'Kevin has the cup, Neil has the lamp, and a spoon is on the table.' with ``gold_obj`` at slot ``gold_pos``;
    returns the sentence and the object mention order."""
    parts = [(f"{n} has the {o}", o) for n, o in holders] + [(f"a {table} is on the table", table)]
    others = [p for p in parts if p[1] != gold_obj]
    rng.shuffle(others)
    g = next(p for p in parts if p[1] == gold_obj)
    pos = min(gold_pos, len(others))
    seq = others[:pos] + [g] + others[pos:]
    txt = ", ".join(s for s, _ in seq[:-1]) + ", and " + seq[-1][0] + "."
    return txt[0].upper() + txt[1:], [o for _, o in seq]


def joinlite_rows(seed: int = JOINLITE_SEED, scenes: int = JOINLITE_SCENES, verb: str = "holding") -> list[Row]:
    """joinlite prompt rows (5 per scene); ``verb`` ``holding`` is joinlite-v1, ``has`` matches the scene's surface."""
    from binding_bench.bindprobes.pools import load_pools

    if verb not in ("holding", "has"):
        raise ValueError(f"verb must be holding or has, got {verb!r}")
    rng = random.Random(seed)
    pools = load_pools().item
    rows: list[Row] = []
    for s in range(scenes):
        P = 2 if s % 2 == 0 else 3
        people = rng.sample(list(pools.names), P)
        objs = rng.sample(list(pools.objects), P + 1)
        held = dict(zip(people, objs[:P]))
        table = objs[P]
        cands = [" " + o for o in objs]
        spk = people[0]
        other = people[1]                                           # P2: the addressee; P3: the vocative target
        third_p = people[2] if P == 3 else None
        n_fill = 2 + s % 2
        fill_spk = [people[(i + (s // 2) % P) % P] for i in range(n_fill)]
        if (s // 4) % 2 == 0:
            fill_spk[-1] = spk                                       # last filler speaker = query speaker half the time
        fill = [f"{w}: {rng.choice(FILLERS)}" for w in fill_spk]
        lead = f"A chat between {', '.join(people[:-1])} and {people[-1]}."
        # one scene sentence per scene (orig and swap differ only in the query speaker's name)
        sent, order = _scene_sentence(list(held.items()), table, random.Random(f"{seed}-{s}"), (s // 2) % (P + 1),
                                      held[other])

        def emit(qkind: str, arm: str, query: str, gold_person: str) -> None:
            gold = held[gold_person]
            prompt = "\n".join([lead + " " + sent] + fill + [query])
            rows.append({"id": f"jl{s:04d}-{qkind}-{arm}", "scene": s, "P": P, "qkind": qkind, "arm": arm,
                         "prompt": prompt, "candidates": cands, "gold": gold,
                         "gold_is_last_mention": order[-1] == gold,
                         "q_speaker": query.split(":")[0] if qkind != "third" else None,
                         "q_person": gold_person, "people": people})

        v3, vi, vy = (("is holding", "I'm holding", "you're holding") if verb == "holding"
                      else ("has", "I have", "you have"))
        emit("third", "orig", f"Now {other} {v3} the", other)
        emit("self", "orig", f"{spk}: Right now {vi} the", spk)
        emit("self", "swap", f"{other}: Right now {vi} the", other)
        if P == 2:
            emit("addressee", "orig", f"{spk}: Right now {vy} the", other)
            emit("addressee", "swap", f"{other}: Right now {vy} the", spk)
        else:
            emit("addressee", "orig", f"{spk}: {other}, right now {vy} the", other)
            emit("third", "orig3", f"Now {third_p} {v3} the", third_p)
    return rows


# ---- swapdyn ---------------------------------------------------------------------------------------------------------

SWAPDYN_CONDITIONS = ("0shot", "3shot")


def swapdyn_rows(deixis_items: Path) -> list[Row]:
    """The speaker swap on every P2 deictic self / addressee item of a deixis items file (the bp-deixis-v2 battery's
    for swapdyn-v1), under :data:`SWAPDYN_CONDITIONS`."""
    from binding_bench.bindprobes.grid import build_payload_item
    from binding_bench.bindprobes.items import load_items

    rows: list[Row] = []
    for it in load_items(deixis_items):
        p_, e_, q_ = it.cell
        if p_ != 2 or q_ not in ("self", "addressee") or it.meta["variant"] != "deictic":
            continue
        people = it.spec["people"]
        spk = it.spec["query"]["speaker"]
        other = next(x for x in people if x != spk)
        for c in SWAPDYN_CONDITIONS:
            pay = build_payload_item(it, c)
            head, last = pay["prompt"].rsplit("\n", 1)
            if not last.startswith(f"{spk}: "):
                raise ValueError(f"{it.item_id}: query line {last!r} not spoken by {spk}")
            swapped = f"{head}\n{other}: {last[len(spk) + 2:]}"
            base = {"base_id": it.item_id, "qkind": q_, "E": e_, "condition": c, "candidates": pay["candidates"],
                    "gold_orig": it.gold, "gold_is_last_mention": it.meta["gold_is_last_mention"]}
            for tag, prompt in (("orig", pay["prompt"]), ("swap", swapped)):
                rows.append({"id": f"{it.item_id}|{c}|{tag}", "arm": tag, "prompt": prompt, **base})
    return rows


# ---- stage-5 decomposition -------------------------------------------------------------------------------------------

DECOMP_VARIANTS = ("deictic", "third", "vocative", "named_transfer", "named_query", "three_vocative")
DECOMP_CORE_VARIANTS = ("deictic", "third", "named_transfer")
THIRD_NAMES = ["Rosa", "Felix", "Nadia", "Otto", "Vera", "Igor", "Lena", "Hans"]
EXTRA_OBJS = ["kettle", "candle", "ladder", "napkin", "pillow", "trumpet"]   # never among swapdyn candidates (checked)
SHOT_SEPARATOR = "\n\n* * *\n\n"


@dataclass
class Turn:
    who: str
    kind: str      # own_have | own_mine | give_take | give_giving | give_yours | q_self | q_you
    obj: str = ""


@dataclass
class Episode:
    a: str
    b: str
    turns: list[Turn] = field(default_factory=list)
    answer: str = ""        # shots carry their answer


_PATTERNS = [(re.compile(r"^I have the (\w+)\.$"), "own_have"), (re.compile(r"^The (\w+) is mine\.$"), "own_mine"),
             (re.compile(r"^Here, take my (\w+)\.$"), "give_take"),
             (re.compile(r"^I'm giving you my (\w+)\.$"), "give_giving"),
             (re.compile(r"^My (\w+) is yours now\.$"), "give_yours"),
             (re.compile(r"^Right now I'm holding the(?: (\w+)\.)?$"), "q_self"),
             (re.compile(r"^Right now you're holding the(?: (\w+)\.)?$"), "q_you")]

_DEICTIC = {"own_have": "I have the {o}.", "own_mine": "The {o} is mine.", "give_take": "Here, take my {o}.",
            "give_giving": "I'm giving you my {o}.", "give_yours": "My {o} is yours now."}


def _parse(block: str) -> Episode:
    lines = block.strip("\n").split("\n")
    m = re.match(r"^A chat between (\w+) and (\w+)\.$", lines[0])
    if not m:
        raise ValueError(f"bad header {lines[0]!r}")
    ep = Episode(m.group(1), m.group(2))
    for ln in lines[1:]:
        who, utt = ln.split(": ", 1)
        for rx, kind in _PATTERNS:
            mm = rx.match(utt)
            if mm:
                obj = mm.group(1) or ""
                if kind.startswith("q_"):
                    ep.answer = obj
                    obj = ""
                ep.turns.append(Turn(who, kind, obj))
                break
        else:
            raise ValueError(f"unparsed turn {utt!r}")
    return ep


def _other(ep: Episode, who: str) -> str:
    return ep.b if who == ep.a else ep.a


def _simulate(ep: Episode) -> tuple[dict[str, str], str]:
    owner: dict[str, str] = {}
    target = ""
    for t in ep.turns:
        if t.kind.startswith("own"):
            owner[t.obj] = t.who
        elif t.kind.startswith("give"):
            owner[t.obj] = _other(ep, t.who)
        else:
            target = t.who if t.kind == "q_self" else _other(ep, t.who)
    return owner, target


def _render(ep: Episode, variant: str, with_answer: bool, third: str = "", extra: str = "") -> str:
    q = ep.turns[-1]
    tgt = _simulate(ep)[1]
    ans = f" {ep.answer}." if with_answer else ""
    if variant == "third":
        out = []
        for t in ep.turns[:-1]:
            out.append(f"{t.who} has the {t.obj}." if t.kind.startswith("own")
                       else f"{t.who} gives the {t.obj} to {_other(ep, t.who)}.")
        return "\n".join(out + [f"Right now {tgt} is holding the{ans}"])
    head = (f"A chat between {ep.a}, {ep.b} and {third}." if variant == "three_vocative"
            else f"A chat between {ep.a} and {ep.b}.")
    out = [head]
    for k, t in enumerate(ep.turns[:-1]):
        if variant == "named_transfer" and t.kind.startswith("give"):
            out.append(f"[{t.who} gives the {t.obj} to {_other(ep, t.who)}.]")
        elif variant == "three_vocative" and t.kind.startswith("give"):  # three people: the transfer's "you" is named
            u = _DEICTIC[t.kind].format(o=t.obj)
            out.append(f"{t.who}: {_other(ep, t.who)}, " + (u if u.startswith("I'm") else u[0].lower() + u[1:]))
        else:
            out.append(f"{t.who}: " + _DEICTIC[t.kind].format(o=t.obj))
        if variant == "three_vocative" and k == 0:
            out.append(f"{third}: I have the {extra}.")
    if variant == "named_query":
        out.append(f"Right now {tgt} is holding the{ans}")
    else:
        pron = "I'm" if q.kind == "q_self" else "you're"
        voc = f"{_other(ep, q.who)}, right now" if variant in ("vocative", "three_vocative") else "Right now"
        out.append(f"{q.who}: {voc} {pron} holding the{ans}")
    return "\n".join(out)


def decomp_rows(swapdyn: Sequence[Row]) -> list[Row]:
    """Every swapdyn ``orig`` row under each of :data:`DECOMP_VARIANTS`. Refuses a deictic rendering that does not
    reproduce its source prompt, a shot whose answer the simulator disagrees with, and extra objects that collide with
    a candidate."""
    cand_all = {c.strip() for r in swapdyn for c in r["candidates"]}
    if cand_all & set(EXTRA_OBJS):
        raise ValueError(f"extra objects collide with candidates: {cand_all & set(EXTRA_OBJS)}")
    out: list[Row] = []
    for r in swapdyn:
        if r["arm"] != "orig":
            continue
        eps = [_parse(b) for b in r["prompt"].split(SHOT_SEPARATOR)]
        owner, tgt = _simulate(eps[-1])
        held = [c.strip() for c in r["candidates"] if owner.get(c.strip()) == tgt]
        if held != [r["gold_orig"]]:
            continue
        for s in eps[:-1]:
            o, t = _simulate(s)
            if o.get(s.answer) != t:
                raise ValueError(f"{r['id']}: shot answer {s.answer} not held by {t}")
        rng = random.Random(r["id"])
        names = [n for n in THIRD_NAMES if all(n not in (e.a, e.b) for e in eps)]
        for v in DECOMP_VARIANTS:
            parts = []
            for k, e in enumerate(eps):
                th = rng.choice(names) if v == "three_vocative" else ""
                ex = rng.choice(EXTRA_OBJS) if v == "three_vocative" else ""
                parts.append(_render(e, v, with_answer=k < len(eps) - 1, third=th, extra=ex))
            prompt = SHOT_SEPARATOR.join(parts)
            if v == "deictic" and prompt != r["prompt"]:
                raise ValueError(f"round-trip mismatch on {r['id']}")
            out.append({"id": f"{r['base_id']}|{r['condition']}|{v}", "prompt": prompt,
                        "candidates": r["candidates"], "gold": r["gold_orig"], "qkind": r["qkind"],
                        "condition": r["condition"], "variant": v, "base_id": r["base_id"], "E": r.get("E")})
    return out


def decomp_core(rows: Iterable[Row]) -> list[Row]:
    """The :data:`DECOMP_CORE_VARIANTS` rows of a stage5-decomp-v1 row list, in order."""
    return [r for r in rows if r["variant"] in DECOMP_CORE_VARIANTS]


# frame-ladder-v1: one cue per rung on the swapdyn orig episodes (0-shot). R0, R3 and R4 are the decomp deictic,
# named_transfer and third renderings; R1 states in the scene line who speaks to whom; R2 tags every utterance before
# the query with its addressee; R2-null keeps R2's tag format with no addressee. The query line of R0–R2 is R0's.
FRAME_RUNGS = ("r0", "r1", "r2", "r2null", "r3", "r4")
_RUNG_DECOMP = {"r0": "deictic", "r3": "named_transfer", "r4": "third"}


def _render_frame(ep: Episode, rung: str) -> str:
    if rung in _RUNG_DECOMP:
        return _render(ep, _RUNG_DECOMP[rung], with_answer=False)
    if rung not in FRAME_RUNGS:
        raise ValueError(f"unknown rung {rung!r}; choose from {FRAME_RUNGS}")
    q = ep.turns[-1]
    head = (f"A chat between {ep.a} and {ep.b}, who speak only to each other." if rung == "r1"
            else f"A chat between {ep.a} and {ep.b}.")
    out = [head]
    for t in ep.turns[:-1]:
        tag = {"r1": "", "r2": f" (to {_other(ep, t.who)})", "r2null": " (says)"}[rung]
        out.append(f"{t.who}{tag}: " + _DEICTIC[t.kind].format(o=t.obj))
    pron = "I'm" if q.kind == "q_self" else "you're"
    out.append(f"{q.who}: Right now {pron} holding the")
    return "\n".join(out)


def frame_rows(swapdyn: Sequence[Row]) -> list[Row]:
    """frame-ladder-v1: every 0-shot swapdyn ``orig`` row under each of :data:`FRAME_RUNGS`. Refuses an R0 that does not
    reproduce its source prompt; skips an episode whose simulated holder disagrees with the gold, as decomp does."""
    out: list[Row] = []
    for r in swapdyn:
        if r["arm"] != "orig" or r["condition"] != "0shot":
            continue
        ep = _parse(r["prompt"])
        owner, tgt = _simulate(ep)
        held = [c.strip() for c in r["candidates"] if owner.get(c.strip()) == tgt]
        if held != [r["gold_orig"]]:
            continue
        for rung in FRAME_RUNGS:
            prompt = _render_frame(ep, rung)
            if rung == "r0" and prompt != r["prompt"]:
                raise ValueError(f"round-trip mismatch on {r['id']}")
            out.append({"id": f"{r['base_id']}|{rung}", "prompt": prompt, "candidates": r["candidates"],
                        "gold": r["gold_orig"], "qkind": r["qkind"], "rung": rung, "base_id": r["base_id"],
                        "E": r.get("E")})
    return out


def write_rows(rows: Iterable[Row], out: Path) -> Path:
    """One JSON object per line, key order as built (the pinned files' serialisation)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return out
