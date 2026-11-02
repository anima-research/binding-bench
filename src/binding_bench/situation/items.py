"""Situation-model items: arms, renderings, probes, prompt rows, gates.

An item is one short story (:mod:`binding_bench.situation.world`) with probes at its end, all over the same text:

  situation  "So the key is in the"              → a place      (where the object IS)
  belief     "Ana will look for the key in the"   → a place      (where a person THINKS it is; arm B and L)
  knot       "The one who moved the key was"      → a person     (who did one mentioned thing)

Arms (each a fixed event skeleton, filled from the pools):
  U  update          put by A, move by B                         situation = the new place
  N  no change       put by A, filler                            situation = the first place (guards "always the update")
  R  recency trap    put, move, then someone opens the OLD place for something else
                                                                  situation = the new place; the last place mention is the old
  B  belief          A puts it, A leaves, B moves it, A returns  situation = new; belief(A) = old
  L  resolved lie    A puts, A leaves, B moves, A returns, B tells A it's where A left it (false), A then sees it
                     in the new place                             situation = new; belief(A) = new (seen after the lie)

Renderings: each phrase slot has three paraphrases; rendering ``r`` takes paraphrase ``r`` of every slot, so a story's
renderings differ only in wording. The setup sentence names all three places, so every candidate is mentioned.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

from binding_bench.lexicon import Name, Noun, Phrase, PhraseSet, check_pools, draw
from binding_bench.situation.world import CHANGES, Event, State, check_story

Arm = Literal["U", "N", "R", "B", "L"]
ARMS: tuple[Arm, ...] = ("U", "N", "R", "B", "L")
ProbeKind = Literal["situation", "belief", "knot"]
N_RENDERINGS = 3
SEPARATOR = "\n\n* * *\n\n"
DEFAULT_SEED = 1103

# ── pools ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Places: containers that all take "in" (so the probe's preposition never narrows the candidates). Objects: small
# count nouns, never scored. Names: gendered first names, scored in knot probes. The single-token check over the grid
# tokenizers is :func:`check_tokens`.
_P = lambda w, art="a": Noun(w, "place", article=art)
PLACES: tuple[Noun, ...] = tuple(_P(w) for w in (
    "drawer", "box", "bag", "basket", "jar", "tin", "closet", "backpack", "folder", "bucket", "chest", "sack",
    "trunk", "bin", "pot", "cabinet", "purse",
)) + (_P("envelope", "an"),)
_O = lambda w, art="a": Noun(w, "object", article=art)
OBJECTS: tuple[Noun, ...] = tuple(_O(w) for w in (
    "key", "ring", "coin", "letter", "ticket", "map", "pen", "watch", "phone", "wallet", "badge", "stamp", "bell",
    "spoon", "candle", "brush", "card", "photo", "whistle", "button", "marble", "shell", "comb", "glove",
))
NAMES: tuple[Name, ...] = tuple(Name(w, "f") for w in (
    "Anna", "Emma", "Lucy", "Kate", "Jane", "Mary", "Sarah", "Ruth", "Alice", "Clara", "Julia", "Laura", "Helen",
    "Nina", "Maya", "Rachel", "Susan", "Linda", "Diana", "Wendy", "Carol", "Ellen", "Molly", "Erin", "Lisa",
)) + tuple(Name(w, "m") for w in (
    "Tom", "Ben", "Leo", "Paul", "Adam", "Hugo", "Owen", "Ivan", "Omar", "Carl", "Dan", "Eric", "Fred", "Greg",
    "Jack", "Luke", "Matt", "Nick", "Peter", "Ryan", "Sean", "Seth", "Tony", "Walt", "Kevin",
))
check_pools({"places": PLACES, "objects": OBJECTS, "names": NAMES})

# ── phrase sets (narration; every phrase neutral: none implies a change) ─────────────────────────────────────────────
_E = frozenset({"event", "narration"})
_S = frozenset({"state", "narration"})
PHRASES: dict[str, PhraseSet] = {s.name: s for s in (
    PhraseSet("setup", (Phrase("{people} are at home. In the room there is {places}.", _S),
                        Phrase("{people} are in a room with {places}.", _S),
                        Phrase("The room where {people} are has {places}.", _S))),
    PhraseSet("put", (Phrase("{a} puts the {o} in the {p}.", _E | {"location"}),
                      Phrase("{a} places the {o} in the {p}.", _E | {"location"}),
                      Phrase("{a} drops the {o} into the {p}.", _E | {"location"}))),
    PhraseSet("move", (Phrase("{a} takes the {o} out of the {s} and puts it in the {p}.", _E | {"location"}),
                       Phrase("{a} moves the {o} from the {s} to the {p}.", _E | {"location"}),
                       Phrase("{a} shifts the {o} from the {s} into the {p}.", _E | {"location"}))),
    PhraseSet("leave", (Phrase("{a} goes out.", _E), Phrase("{a} steps outside.", _E), Phrase("{a} leaves the room.", _E))),
    PhraseSet("return", (Phrase("{a} comes back.", _E), Phrase("{a} returns.", _E), Phrase("{a} comes back in.", _E))),
    PhraseSet("claim", (Phrase("{a} tells {l} that the {o} is in the {p}.", _E | {"speech", "location"}),
                        Phrase("{a} says to {l} that the {o} is in the {p}.", _E | {"speech", "location"}),
                        Phrase("{a} lets {l} know that the {o} is in the {p}.", _E | {"speech", "location"}))),
    PhraseSet("reveal", (Phrase("{a} opens the {p} and sees the {o} inside.", _E | {"location"}),
                         Phrase("{a} looks in the {p} and finds the {o}.", _E | {"location"}),
                         Phrase("{a} spots the {o} in the {p}.", _E | {"location"}))),
    PhraseSet("mention", (Phrase("{a} opens the {p} to look for something else.", _E),
                          Phrase("{a} reaches into the {p} for something else.", _E),
                          Phrase("{a} glances into the {p} while looking for something else.", _E))),
    PhraseSet("chat", (Phrase("{a} makes some tea.", _E), Phrase("{a} reads for a while.", _E),
                       Phrase("{a} hums a song.", _E))),
    # probe leads: fixed classes (neutral), paraphrased by rendering
    PhraseSet("q_situation", (Phrase("So the {o} is in the", _S | {"location"}), Phrase("The {o} is in the", _S | {"location"}),
                              Phrase("That means the {o} is in the", _S | {"location"}))),
    PhraseSet("q_belief", (Phrase("{a} will look for the {o} in the", frozenset({"belief", "narration"})),
                           Phrase("When {a} wants the {o}, {a} will look in the", frozenset({"belief", "narration"})),
                           Phrase("{a} expects to find the {o} in the", frozenset({"belief", "narration"})))),
    PhraseSet("q_knot_put", (Phrase("The one who put the {o} away was", _S), Phrase("The {o} was put away by", _S),
                             Phrase("The person who put the {o} there was", _S))),
    PhraseSet("q_knot_move", (Phrase("The one who moved the {o} was", _S), Phrase("The {o} was moved by", _S),
                              Phrase("The person who moved the {o} was", _S))),
    PhraseSet("q_knot_claim", (Phrase("The one who said the {o} was in the {p} was", _S | {"speech"}),
                               Phrase("The claim that the {o} was in the {p} came from", _S | {"speech"}),
                               Phrase("The person who said the {o} was in the {p} was", _S | {"speech"})),),
)}
FORBID: frozenset[str] = frozenset({"implies_change", "ownership", "matches_query"})


@dataclass
class Probe:
    kind: ProbeKind
    lead: str                         # a PHRASES key (q_*)
    gold: str
    candidates: list[str]             # bare words; scored as " " + word
    slots: dict[str, str] = field(default_factory=dict)


@dataclass
class Item:
    item_id: str
    arm: Arm
    people: list[str]
    obj: str
    places: list[str]                 # setup order (all three are candidates)
    events: list[dict[str, Any]]
    probes: list[Probe]
    meta: dict[str, Any] = field(default_factory=dict)

    def event_list(self) -> list[Event]:
        return [Event(**e) for e in self.events]


# ── arms ──────────────────────────────────────────────────────────────────────────────────────────────────────────────

def _skeleton(arm: Arm, A: str, B: str, p0: str, p1: str, p2: str, rng: random.Random) -> list[Event]:
    put = Event("put", A, place=p0)
    move = Event("move", B, place=p1, src=p0)
    filler = Event("chat", rng.choice([A, B]))
    if arm == "U":
        return [put, move, filler]
    if arm == "N":
        return [put, filler, Event("chat", B if filler.actor == A else A)]
    if arm == "R":
        return [put, move, Event("mention", rng.choice([A, B]), place=p0)]
    if arm == "B":
        return [put, Event("leave", A), move, Event("return", A)]
    if arm == "L":
        return [put, Event("leave", A), move, Event("return", A), Event("claim", B, place=p0, listener=A),
                Event("reveal", A, place=p1)]
    raise ValueError(f"unknown arm {arm!r}")


def build(item_id: str, arm: Arm, rng: random.Random) -> Item:
    women, men = [n for n in NAMES if n.gender == "f"], [n for n in NAMES if n.gender == "m"]
    A, B = (x.word for x in rng.sample([rng.choice(women), rng.choice(men)], 2))
    obj = draw(rng, OBJECTS, 1)[0].word
    places = [p.word for p in draw(rng, PLACES, 3)]
    p0, p1, p2 = rng.sample(places, 3)
    events = _skeleton(arm, A, B, p0, p1, p2, rng)
    st = check_story([A, B], places, events)
    probes = [Probe("situation", "q_situation", str(st.world), list(places), {"o": obj})]
    if arm in ("B", "L"):
        bel = st.belief[A]
        if arm == "B" and bel == st.world:
            raise AssertionError(f"{item_id}: B must leave A's belief behind the world")
        probes.append(Probe("belief", "q_belief", str(bel), list(places), {"o": obj, "a": A}))
    who = {"U": ("q_knot_move", B), "N": ("q_knot_put", A), "R": ("q_knot_move", B), "B": ("q_knot_move", B),
           "L": ("q_knot_claim", B)}[arm]
    knot_slots = {"o": obj, "p": p0} if arm == "L" else {"o": obj}
    probes.append(Probe("knot", who[0], who[1], [A, B], knot_slots))
    meta = {"world": st.world, "belief": dict(st.belief), "last_place_mention": st.place_mentions[-1],
            "gold_is_last_place_mention": st.place_mentions[-1] == st.world,
            "n_changes": sum(e.kind in CHANGES for e in events)}
    return Item(item_id, arm, [A, B], obj, places, [asdict(e) for e in events], probes, meta)


# ── rendering ─────────────────────────────────────────────────────────────────────────────────────────────────────────

def _place_list(places: Sequence[str]) -> str:
    art = {p.word: p for p in PLACES}
    parts = [art[w].indefinite() for w in places]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def render_story(item: Item, r: int) -> str:
    if not 0 <= r < N_RENDERINGS:
        raise ValueError(f"rendering {r} (have {N_RENDERINGS})")
    rng = random.Random(0)                     # unused: the rendering index pins every phrase
    pick = lambda name: PHRASES[name].pick(rng, forbid=FORBID, index=r)[1]
    out = [pick("setup").fill(people=f"{item.people[0]} and {item.people[1]}", places=_place_list(item.places))]
    for e in item.event_list():
        slots = {"a": e.actor, "o": item.obj, "p": e.place or "", "s": e.src or "", "l": e.listener or ""}
        ph = pick(e.kind)
        out.append(ph.fill(**{k: v for k, v in slots.items() if k in ph.slots}))
    return " ".join(out)


def probe_prompt(item: Item, probe: Probe, r: int, with_answer: bool = False) -> str:
    lead = PHRASES[probe.lead].pick(random.Random(0), forbid=FORBID, index=r)[1].fill(**probe.slots)
    text = f"{render_story(item, r)} {lead}"
    return f"{text} {probe.gold}." if with_answer else text


# ── generation, rows, gates ───────────────────────────────────────────────────────────────────────────────────────────

def generate(per_arm: int, seed: int = DEFAULT_SEED, arms: Sequence[Arm] = ARMS) -> list[Item]:
    out = []
    for arm in arms:
        rng = random.Random(f"situation:{seed}:{arm}")
        for k in range(per_arm):
            out.append(build(f"sit-{arm}-{k:03d}", arm, rng))
    return out


def rows(items: Sequence[Item], conditions: Sequence[str] = ("0shot",)) -> list[dict[str, Any]]:
    """Prompt rows: every item × probe × rendering (× condition; ``fair`` comes from :func:`fair_rows`)."""
    out = []
    for it in items:
        for pr in it.probes:
            for r in range(N_RENDERINGS):
                out.append({"id": f"{it.item_id}|{pr.kind}|r{r}|0shot", "prompt": probe_prompt(it, pr, r),
                            "candidates": [" " + c for c in pr.candidates], "gold": pr.gold, "arm": it.arm,
                            "probe": pr.kind, "rendering": r, "condition": "0shot", "item_id": it.item_id,
                            "gold_is_last_place_mention": it.meta["gold_is_last_place_mention"]
                            if pr.kind != "knot" else None})
    return out


def gate(item: Item) -> None:
    """Refuse an item whose stories or probes break the battery's rules (raises ``AssertionError``)."""
    st = check_story(item.people, item.places, item.event_list())
    sit = next(p for p in item.probes if p.kind == "situation")
    assert sit.gold == st.world, f"{item.item_id}: situation gold {sit.gold} != world {st.world}"
    for pr in item.probes:
        assert pr.gold in pr.candidates and len(set(pr.candidates)) == len(pr.candidates), item.item_id
    if item.arm == "R":
        assert item.meta["last_place_mention"] != st.world, f"{item.item_id}: R's last mention is the gold"
    if item.arm == "N":
        assert item.meta["n_changes"] == 1, f"{item.item_id}: N must not move the object"
    if item.arm == "L":
        evs = item.event_list()
        claim = next(e for e in evs if e.kind == "claim")
        assert claim.place != st.world, f"{item.item_id}: L's claim must be false"
        listener_saw_move = any(e.kind == "leave" and e.actor == claim.listener for e in evs[:[x.kind for x in evs].index("move")])
        assert listener_saw_move, f"{item.item_id}: L's listener must be away for the move (else the lie is moot)"
    if item.arm == "B":
        bel = next(p for p in item.probes if p.kind == "belief")
        assert bel.gold != st.world, f"{item.item_id}: B's belief must differ from the world"
    for r in range(N_RENDERINGS):
        text = render_story(item, r)
        for p in item.places:
            assert f" {p}" in text, f"{item.item_id}/r{r}: place {p} never mentioned"
        for w in (" now ", " still ", " again ", " anymore"):
            assert w not in f" {text} ", f"{item.item_id}/r{r}: change word {w.strip()!r}"


def emit(out: Path, per_arm: int = 100, seed: int = DEFAULT_SEED) -> Path:
    items = generate(per_arm, seed)
    for it in items:
        gate(it)
    out.mkdir(parents=True, exist_ok=True)
    (out / "items.jsonl").write_text("".join(json.dumps(asdict(it)) + "\n" for it in items))
    (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows(items)))
    return out / "rows.jsonl"


def check_tokens(tokenizer_keys: Sequence[str] = ()) -> dict[str, dict[str, int]]:
    """``{word: {tokenizer: n_tokens}}`` for every scored word (places, names) that is not one token after a leading
    space, over the grid tokenizers (``riders.items.GRID_TOKENIZERS``; all when ``tokenizer_keys`` is empty)."""
    from transformers import AutoTokenizer  # deferred: heavy import

    from binding_bench.riders.items import GRID_TOKENIZERS

    keys = list(tokenizer_keys) or list(GRID_TOKENIZERS)
    words = [p.word for p in PLACES] + [n.word for n in NAMES]
    bad: dict[str, dict[str, int]] = {}
    for k in keys:
        tok = AutoTokenizer.from_pretrained(GRID_TOKENIZERS[k])
        for w in words:
            n = len(tok(" " + w, add_special_tokens=False).input_ids)
            if n != 1:
                bad.setdefault(w, {})[k] = n
    return bad
