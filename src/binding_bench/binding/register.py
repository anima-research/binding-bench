"""register-v1 to -v3: the same binding scenes in a seen register and in two registers no text uses, paired item by item.

A model that binds in the registers it has read may or may not bind when the same scene is written in a form it has
never seen. Each scene is rendered three ways, with the same names, objects, pronouns, candidates and gold:

* ``seen``: the form the stage gauges and setbind use (a chat with ``Name: line`` turns; one sentence per line).
* ``utag``: an invented transcript markup around the same English lines: a ``⁜ A ⁘ B`` header, ``⟦Name⟧ ≻`` speaker
  tags, ``⁂`` narration lines, ``⁂⁂⁂`` between worked examples.
* ``usyntax``: an invented relational syntax in which only the pronouns (I / you / my) and the nouns stay English:
  ``⟦Name⟧ ≻ I ⧉ cup`` (holds), ``⟦Name⟧ ≻ my cup ⇝ you`` (gives), ``⌂ ⧉ spoon`` (lies unheld), ``◷ … ⧉`` (the query:
  who holds what now), with nonce filler turns.

Three constructs, each read as its gauge reads it:

* ``static``: P2 scenes stated once in narration (joinlite's design), queried as ``third`` (the lookup control),
  ``self`` and ``addressee``, the self and addressee queries also with the query speaker swapped. Worked examples in
  the item's register, drawn from the shot pools (v3: four of the query's own kind, two spoken by each person); the
  seen register also runs 0-shot.
* ``dynamic``: deixis P2 deictic self / addressee items (objects change hands through I / my / you lines), with the
  speaker of the query line swapped (swapdyn's design); the item's own 3 exemplars, in the item's register; the seen
  register also runs 0-shot. The seen 3-shot rendering reproduces the deixis payload byte for byte.
* ``set``: setbind (generation 4, the ``lines`` rendering) under ``fair`` (4 balanced exemplars) in each register.

``register_rows`` rebuilds each pinned file byte for byte (``tests/test_register.py``); the reader is
:func:`binding_bench.binding.readers.register`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

from binding_bench.binding.stagegauges import FILLERS, Episode, Row, _parse, _scene_sentence

REGISTERS = ("seen", "utag", "usyntax")
SEED = 2611


@dataclass(frozen=True)
class Version:
    """One pinned generation of the battery: its sizes and the static construct's worked examples."""
    name: str
    static_scenes: int
    dynamic_per_cell: int         # deixis base items per cell (×3 cells: P2 addressee; ×3 more: P2 self)
    set_per_cell: int             # setbind base items per cell (×10 cells)
    static_shots: tuple[str, ...]  # query kinds of the static construct's worked examples
    static_cond: str              # that condition's name in the rows
    shuffle_shots: bool           # v1 drew the order of its three kinds per scene
    matched: bool = False         # v3: every query is shown worked examples of its own kind (``static_shots`` = one kind)


# v1 and v2 (smoke only): mixed static examples. v1's (one third-person, one self, one addressee query) and v2's (two
# self, two addressee) both left a model that reads "you" right with no examples reading it as the speaker: its static
# "you" swap shift inverted in every register. v3 shows each query worked examples of its own kind, as swapdyn does;
# the speaker swap then still asks who the other person is, which is the binding.
VERSIONS: dict[str, Version] = {
    "register-v1": Version("register-v1", 400, 200, 60, ("third", "self", "addressee"), "3shot", True),
    "register-v2": Version("register-v2", 800, 300, 240, ("self", "self", "addressee", "addressee"), "4shot", False),
    "register-v3": Version("register-v3", 800, 300, 240, ("matched",) * 4, "4shot", False, True),
}
STATIC_SCENES = VERSIONS["register-v1"].static_scenes

# the invented registers' scaffolding: glyphs and nonce syllables, none of them an English word
HEAD, PAIR, NARR, SAYS = "⁜", "⁘", "⁂", "≻"
HOLD, GIVE, PLACE, NOW = "⧉", "⇝", "⌂", "◷"
U_SEPARATOR = "\n\n⁂⁂⁂\n\n"
NONCE_FILLERS = ("~ tobu wela ~", "~ ru ka ru ~", "~ silo ~", "~ fen dari ~", "~ omu omu ~", "~ ke pasu ~",
                 "~ lori ~", "~ vane tu ~", "~ ishi ~", "~ moku ra ~", "~ peli sa ~", "~ dunu ~")
SEEN_SEPARATOR = "\n\n* * *\n\n"


def tag(name: str) -> str:
    return f"⟦{name}⟧"


def _sep(register: str) -> str:
    return SEEN_SEPARATOR if register == "seen" else U_SEPARATOR


# ---- dynamic: deixis episodes -----------------------------------------------------------------------------------------

_USYNTAX_TURN = {"own_have": "I {h} {o}", "own_mine": "I {h} {o}", "own_hand": "I {h} {o}",
                 "give_take": "my {o} {g} you", "give_giving": "my {o} {g} you", "give_yours": "my {o} {g} you",
                 "give_hand": "my {o} {g} you"}
_SEEN_TURN = {"own_have": "I have the {o}.", "own_mine": "The {o} is mine.", "give_take": "Here, take my {o}.",
              "give_giving": "I'm giving you my {o}.", "give_yours": "My {o} is yours now.",
              "own_hand": "The {o} is in my hand.", "give_hand": "My {o} is in your hand now."}


def render_episode(ep: Episode, register: str, with_answer: bool, speaker: str = "") -> str:
    """One deixis P2 deictic episode; ``speaker`` overrides the query line's speaker (the swap)."""
    q = ep.turns[-1]
    who_q = speaker or q.who
    ans = ep.answer if with_answer else ""
    if register == "seen":
        lines = [f"A chat between {ep.a} and {ep.b}."]
        lines += [f"{t.who}: " + _SEEN_TURN[t.kind].format(o=t.obj) for t in ep.turns[:-1]]
        pron = "I'm" if q.kind == "q_self" else "you're"
        lines.append(f"{who_q}: Right now {pron} holding the" + (f" {ans}." if ans else ""))
    elif register == "utag":
        lines = [f"{HEAD} {ep.a} {PAIR} {ep.b}"]
        lines += [f"{tag(t.who)} {SAYS} " + _SEEN_TURN[t.kind].format(o=t.obj) for t in ep.turns[:-1]]
        pron = "I'm" if q.kind == "q_self" else "you're"
        lines.append(f"{tag(who_q)} {SAYS} Right now {pron} holding the" + (f" {ans}." if ans else ""))
    elif register == "usyntax":
        lines = [f"{HEAD} {ep.a} {PAIR} {ep.b}"]
        lines += [f"{tag(t.who)} {SAYS} " + _USYNTAX_TURN[t.kind].format(o=t.obj, h=HOLD, g=GIVE) for t in ep.turns[:-1]]
        pron = "I" if q.kind == "q_self" else "you"
        lines.append(f"{tag(who_q)} {SAYS} {NOW} {pron} {HOLD}" + (f" {ans}" if ans else ""))
    else:
        raise ValueError(f"unknown register {register!r}")
    return "\n".join(lines)


def dynamic_rows(v: Version, seed: int = SEED) -> list[Row]:
    from binding_bench.bindprobes import deixis
    from binding_bench.bindprobes.grid import build_payload_item
    from binding_bench.bindprobes.items import generate_family
    from binding_bench.bindprobes.pools import load_pools

    rows: list[Row] = []
    for it in generate_family("deixis", load_pools(), v.dynamic_per_cell, seed):
        p_, e_, q_ = it.cell
        if p_ != 2 or q_ not in ("self", "addressee") or it.meta["variant"] != "deictic":
            continue
        shots = [_parse(deixis.render(s, True)) for s in it.shot_items()[:3]]
        query = _parse(deixis.render(it, False))
        spk = query.turns[-1].who
        other = query.b if spk == query.a else query.a
        cands = [" " + c for c in it.candidates]
        for register in REGISTERS:
            conds = ("3shot", "0shot") if register == "seen" else ("3shot",)
            for cond in conds:
                head = [render_episode(s, register, True) for s in shots] if cond == "3shot" else []
                for arm, who in (("orig", spk), ("swap", other)):
                    prompt = _sep(register).join(head + [render_episode(query, register, False, who)])
                    if register == "seen" and arm == "orig":
                        ref = build_payload_item(it, cond)["prompt"]
                        if prompt != ref:
                            raise ValueError(f"{it.item_id} {cond}: the seen rendering does not reproduce the payload")
                    rows.append({"id": f"{it.item_id}|{register}|{cond}|{arm}", "construct": "dynamic",
                                 "register": register, "condition": cond, "arm": arm, "base_id": it.item_id,
                                 "qkind": q_, "E": e_, "prompt": prompt, "candidates": cands,
                                 "gold_orig": it.gold, "gold_is_last_mention": it.meta["gold_is_last_mention"]})
    return rows


# ---- static: P2 scenes stated once ------------------------------------------------------------------------------------

def _static_scene(rng: random.Random, names: Sequence[str], objects: Sequence[str], s: int) -> dict[str, Any]:
    people = rng.sample(list(names), 2)
    objs = rng.sample(list(objects), 3)
    held = dict(zip(people, objs[:2]))
    table = objs[2]
    spk, other = people
    n_fill = 2 + s % 2
    fill_spk = [people[(i + (s // 2) % 2) % 2] for i in range(n_fill)]
    if (s // 4) % 2 == 0:
        fill_spk[-1] = spk
    fills = [(w, rng.randrange(len(FILLERS))) for w in fill_spk]
    sent, order = _scene_sentence(list(held.items()), table, random.Random(f"register-static-{s}"), (s // 2) % 3,
                                  held[other])
    return {"people": people, "held": held, "table": table, "objs": objs, "fills": fills, "sentence": sent,
            "order": order}


def _scene_usyntax(sc: dict[str, Any]) -> str:
    parts = []
    for o in sc["order"]:
        owner = next((p for p, x in sc["held"].items() if x == o), None)
        parts.append(f"{tag(owner)} {HOLD} {o}" if owner else f"{PLACE} {HOLD} {o}")
    return f"{NARR} " + " · ".join(parts)


def render_static(sc: dict[str, Any], register: str, qkind: str, speaker: str, answer: str = "") -> str:
    """One scene; ``qkind`` third / self / addressee, ``speaker`` the query line's speaker (third: the person asked
    about)."""
    a, b = sc["people"]
    if register == "seen":
        lines = [f"A chat between {a} and {b}. {sc['sentence']}"]
        lines += [f"{w}: {FILLERS[k]}" for w, k in sc["fills"]]
        q = {"third": f"Right now {speaker} is holding the", "self": f"{speaker}: Right now I'm holding the",
             "addressee": f"{speaker}: Right now you're holding the"}[qkind]
        return "\n".join(lines + [q + (f" {answer}." if answer else "")])
    if register == "utag":
        lines = [f"{HEAD} {a} {PAIR} {b}", f"{NARR} {sc['sentence']}"]
        lines += [f"{tag(w)} {SAYS} {FILLERS[k]}" for w, k in sc["fills"]]
        q = {"third": f"{NARR} Right now {speaker} is holding the",
             "self": f"{tag(speaker)} {SAYS} Right now I'm holding the",
             "addressee": f"{tag(speaker)} {SAYS} Right now you're holding the"}[qkind]
        return "\n".join(lines + [q + (f" {answer}." if answer else "")])
    if register == "usyntax":
        lines = [f"{HEAD} {a} {PAIR} {b}", _scene_usyntax(sc)]
        lines += [f"{tag(w)} {SAYS} {NONCE_FILLERS[k]}" for w, k in sc["fills"]]
        q = {"third": f"{NARR} {NOW} {tag(speaker)} {HOLD}", "self": f"{tag(speaker)} {SAYS} {NOW} I {HOLD}",
             "addressee": f"{tag(speaker)} {SAYS} {NOW} you {HOLD}"}[qkind]
        return "\n".join(lines + [q + (f" {answer}" if answer else "")])
    raise ValueError(f"unknown register {register!r}")


def _static_gold(sc: dict[str, Any], qkind: str, speaker: str) -> str:
    a, b = sc["people"]
    if qkind == "third":
        return sc["held"][speaker]
    if qkind == "self":
        return sc["held"][speaker]
    return sc["held"][b if speaker == a else a]


def static_rows(v: Version, seed: int = SEED) -> list[Row]:
    from binding_bench.bindprobes.pools import load_pools

    split = load_pools()
    names_i = list(split.item.female) + list(split.item.male)
    names_s = list(split.shot.female) + list(split.shot.male)
    rng = random.Random(f"{v.name}:static:{seed}")
    n_shots = len(v.static_shots)
    rows: list[Row] = []
    for s in range(v.static_scenes):
        sc = _static_scene(rng, names_i, split.item.objects, s)
        srng = random.Random(f"{v.name}:static:{seed}:{s}:shots")
        shot_kinds = list(v.static_shots)
        if v.shuffle_shots:
            srng.shuffle(shot_kinds)
            first = [j % 2 == 0 for j in range(n_shots)]          # v1: the speaker alternates by position
        else:
            order = list(range(n_shots))
            srng.shuffle(order)
            shot_kinds = [v.static_shots[i] for i in order]
            first = [i % 2 == 0 for i in order]                     # v2: each kind spoken once by each person
        shots_for: dict[str, list[tuple[dict[str, Any], str, str, str]]] = {}
        for kind in (("third", "self", "addressee") if v.matched else ("all",)):
            shots = []
            for j, qk in enumerate(shot_kinds):
                qk = kind if v.matched else qk
                ex = _static_scene(srng, names_s, split.shot.objects, s * n_shots + j)
                a, b = ex["people"]
                spk = (a if j % 2 == 0 else b) if v.matched else (b if qk == "third" else (a if first[j] else b))
                shots.append((ex, qk, spk, _static_gold(ex, qk, spk)))
            shots_for[kind] = shots
        a, b = sc["people"]
        cands = [" " + o for o in sc["objs"]]
        queries = [("third", "orig", b), ("self", "orig", a), ("self", "swap", b), ("addressee", "orig", a),
                   ("addressee", "swap", b)]
        for register in REGISTERS:
            conds = (v.static_cond, "0shot") if register == "seen" else (v.static_cond,)
            for cond in conds:
                for qkind, arm, spk in queries:
                    shots = shots_for[qkind if v.matched else "all"]
                    head = ([render_static(ex, register, qk, xs, ans) for ex, qk, xs, ans in shots]
                            if cond == v.static_cond else [])
                    gold = _static_gold(sc, qkind, spk)
                    prompt = _sep(register).join(head + [render_static(sc, register, qkind, spk)])
                    rows.append({"id": f"rs{s:04d}|{register}|{cond}|{qkind}|{arm}", "construct": "static",
                                 "register": register, "condition": cond, "arm": arm, "base_id": f"rs{s:04d}",
                                 "scene": s, "qkind": qkind, "prompt": prompt, "candidates": cands, "gold": gold,
                                 "gold_is_last_mention": sc["order"][-1] == gold, "q_speaker": spk})
    return rows


# ---- set: setbind specs -----------------------------------------------------------------------------------------------

def render_set(spec: dict[str, Any], register: str, answer: str = "") -> str:
    from binding_bench.bindprobes.setbind import _lead, _sentence

    if register == "seen":
        text = "\n".join([_sentence(s) for s in spec["sentences"]] + [_lead(spec)])
        return f"{text} {answer}." if answer else text
    if register == "utag":
        text = "\n".join([f"{NARR} {_sentence(s)}" for s in spec["sentences"]] + [f"{NARR} {_lead(spec)}"])
        return f"{text} {answer}." if answer else text
    if register == "usyntax":
        out = []
        for s in spec["sentences"]:
            if s[0] == "bind":
                out.append(f"{tag(s[1])} {HOLD} " + ", ".join(s[2]))
            elif s[0] == "free":
                out.append(f"{PLACE} {HOLD} {s[1]}")
            else:
                out.append(f"{tag(s[1])} {s[2]} {GIVE} {tag(s[3])}")
        out.append(f"{NOW} {tag(spec['query'])} {HOLD}")
        text = "\n".join(out)
        return f"{text} {answer}" if answer else text
    raise ValueError(f"unknown register {register!r}")


def set_rows(v: Version, seed: int = SEED) -> list[Row]:
    from binding_bench.bindprobes.grid import build_payload_item
    from binding_bench.bindprobes.items import FairPool, fair_shots, generate_family
    from binding_bench.bindprobes.pools import load_pools

    items = [it for it in generate_family("setbind", load_pools(), v.set_per_cell, seed)
             if it.meta.get("variant") == "lines"]
    pool = FairPool(items)
    rows: list[Row] = []
    for it in items:
        shots = fair_shots(it, pool)
        cands = [" " + c for c in it.candidates]
        for register in REGISTERS:
            head = [render_set(s.spec, register, s.gold) for s in shots]
            prompt = _sep(register).join(head + [render_set(it.spec, register)])
            if register == "seen" and prompt != build_payload_item(it, "fair", pool)["prompt"]:
                raise ValueError(f"{it.item_id}: the seen rendering does not reproduce the fair payload")
            rows.append({"id": f"{it.item_id}|{register}|fair", "construct": "set", "register": register,
                         "condition": "fair", "base_id": it.meta["pair_id"], "cell": list(it.cell), "prompt": prompt,
                         "candidates": cands, "gold": it.gold, "gold_set": it.meta["gold_set"], "k": it.meta["k"]})
    return rows


def register_rows(name: str = "register-v2", seed: int = SEED) -> list[Row]:
    """One version's rows (:data:`VERSIONS`): static, then dynamic, then set."""
    if name not in VERSIONS:
        raise ValueError(f"unknown register battery {name!r}; choose from {sorted(VERSIONS)}")
    v = VERSIONS[name]
    return static_rows(v, seed) + dynamic_rows(v, seed) + set_rows(v, seed)
