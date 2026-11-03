"""Situation-model items: arms, crossed factors, renderings, probes, prompt rows, gates.

An item is one short story (:mod:`binding_bench.situation.world`) with probes at its end, all over the same text:

  situation    "So the key is in the"                        → a place   (where the object IS)
  belief       "Ana will look for the key in the"             → a place   (where a person thinks it is)
  knot_first   "The one who stowed the key in the drawer was" → a person  (the agent of the earlier placement)
  knot_latest  "The one who carried the key to the pot was"   → a person  (the agent of the later move)

Arms (a fixed event skeleton each, filled from the pools; A puts the object away, B moves it):
  U  update        put by A, move by B                          situation = the new place; belief(A) = world, implicit
  N  no change     put by A, filler                             situation = the first place
  R  recency trap  U, then someone opens the OLD place for something else
  O  out of order  U narrated move first, then "Before that, A had put it in …" (the last placement narrated is not
                   the last one that happened)
  F  foiled move   put by A; B is about to move it elsewhere and doesn't
  W  watched       B moves it, and A sees B do it               belief(A) = world, perceived (explicit)
  B  absent        A leaves, B moves it, A returns              belief(A) = old place
  T  told truly    B, then B tells A where it now is            belief(A) = world, by testimony
  D  deceived      B, then B (with a motive) tells A it's where A left it      belief(A) = old place (the lie)

Belief is asked in W, T (belief = world) and B, D (belief ≠ world): perception and testimony, each true and false. U
asks it too, as a measure, not a gold: the story says only that A is present, so whether A saw the move is a convention
(a blind judge read it both ways). Its reading is the share of answers naming where the object is. Crossed factors, each drawn per item where the arm allows it (recorded in ``meta``):
  prelude     B first puts it in a third place and A moves it into the place the first knot asks about, so neither
              "the first actor" nor "the other one" finds a knot's gold
  postlude    after B's move, A is about to move it again and doesn't, so the story's last actor is not the mover
              (never in B or D: handling the object would show A where it is)
  distractor  the other person (not the knot's gold) later glances into the knot's queried place, so the name nearest
              that place's last mention is not the gold (never A into the object's place in B or D)

Every item asks one knot (first or latest, drawn). The knot questions use verbs no narration phrase uses ("stowed",
"carried"), so a question never matches its event word for word.

Renderings: 0–2 are narration, each slot with three paraphrases (rendering ``r`` takes paraphrase ``r`` of every slot);
3 is the room's transcript ("Anna: I'm putting the key in the drawer."), with the probe as a narrator line. Readings
keep the transcript apart from narration. ``fair`` (:func:`fair_rows`): 4 answered exemplars of the same probe kind
from other items, in the same rendering, no shared name, object or place, at most 2 of the item's arm and at most 2
with its gold position.

The set-level gate (:func:`gate_set`) runs the surface policies of :mod:`binding_bench.situation.shortcuts` and refuses
a set where any policy reaches :data:`SHORTCUT_CEILING` on any probe kind.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

from binding_bench.lexicon import Name, Noun, Phrase, PhraseSet, check_pools, draw
from binding_bench.situation.world import CHANGES, Event, check_story

GENERATION = 2
Arm = Literal["U", "N", "R", "O", "F", "W", "B", "T", "D"]
ARMS: tuple[Arm, ...] = ("U", "N", "R", "O", "F", "W", "B", "T", "D")
ARM_WEIGHT: dict[str, int] = {"O": 2, "F": 2}          # the arms that defeat "the last place put in" get twice the items
BELIEF_ARMS: frozenset[str] = frozenset({"U", "W", "T", "B", "D"})
BELIEF_IS_WORLD: frozenset[str] = frozenset({"U", "W", "T"})
MOVE_ARMS: frozenset[str] = frozenset({"U", "R", "O", "W", "B", "T", "D"})
ProbeKind = Literal["situation", "belief", "knot_first", "knot_latest"]
PROBE_KINDS: tuple[ProbeKind, ...] = ("situation", "belief", "knot_first", "knot_latest")
NARRATION = 3                     # renderings 0–2: narration paraphrase sets
N_RENDERINGS = 4                  # rendering 3: the same story as a transcript of the room
FAIR_K = 4
SEPARATOR = "\n\n* * *\n\n"
DEFAULT_SEED = 1103
SHORTCUT_CEILING = 0.6            # no surface policy may reach this κ on any probe kind of a generated set
P_FACTOR = 0.5                    # the draw rate of each crossed factor where the arm allows it

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
_D = frozenset({"event", "dialogue"})
PHRASES: dict[str, PhraseSet] = {s.name: s for s in (
    PhraseSet("setup", (Phrase("{people} are at home. In the room there is {places}.", _S),
                        Phrase("{people} are in a room with {places}.", _S),
                        Phrase("{people} are in a room. It has {places}.", _S))),
    PhraseSet("put", (Phrase("{a} puts the {o} in the {p}.", _E | {"location"}),
                      Phrase("{a} places the {o} in the {p}.", _E | {"location"}),
                      Phrase("{a} drops the {o} into the {p}.", _E | {"location"}))),
    PhraseSet("move", (Phrase("{a} takes the {o} out of the {s} and puts it in the {p}.", _E | {"location"}),
                       Phrase("{a} moves the {o} from the {s} to the {p}.", _E | {"location"}),
                       Phrase("{a} shifts the {o} from the {s} into the {p}.", _E | {"location"}))),
    # a placement narrated after a later one, in the past perfect
    PhraseSet("fb_put", (Phrase("Before that, {a} had put the {o} in the {p}.", _E | {"location"}),
                         Phrase("Earlier, {a} had placed the {o} in the {p}.", _E | {"location"}),
                         Phrase("Before this, {a} had dropped the {o} into the {p}.", _E | {"location"}))),
    PhraseSet("intend", (Phrase("{a} is about to move the {o} to the {p}, but decides against it.", _E),
                         Phrase("{a} picks up the {o} to take it to the {p}, then sets it down where it was.", _E),
                         Phrase("{a} almost moves the {o} into the {p}, but stops.", _E))),
    PhraseSet("watch", (Phrase("{a} sees {l} do it.", _E), Phrase("{a} watches {l} do it.", _E),
                        Phrase("{a} sees {l} move it.", _E))),
    PhraseSet("leave", (Phrase("{a} goes out.", _E), Phrase("{a} steps outside.", _E), Phrase("{a} leaves the room.", _E))),
    PhraseSet("return", (Phrase("{a} comes back.", _E), Phrase("{a} returns.", _E), Phrase("{a} comes back in.", _E))),
    PhraseSet("claim", (Phrase("{a} tells {l} that the {o} is in the {p}.", _E | {"speech", "location"}),
                        Phrase("{a} says to {l} that the {o} is in the {p}.", _E | {"speech", "location"}),
                        Phrase("{a} mentions to {l} that the {o} is in the {p}.", _E | {"speech", "location"}))),
    PhraseSet("motive", (Phrase("{a} wants to keep the {o} hidden from {l}.", _E),
                         Phrase("{a} would rather {l} did not find the {o}.", _E),
                         Phrase("{a} hopes to keep the {o} away from {l}.", _E))),
    PhraseSet("mention", (Phrase("{a} opens the {p} to get something else.", _E),
                          Phrase("{a} reaches into the {p} for something else.", _E),
                          Phrase("{a} glances into the {p} while searching for something else.", _E))),
    PhraseSet("chat", (Phrase("{a} makes some tea.", _E), Phrase("{a} reads for a while.", _E),
                       Phrase("{a} hums a song.", _E))),
    # the transcript rendering: one line per event, spoken by its actor (a claim is addressed to its listener)
    PhraseSet("d_setup", (Phrase("A chat between {people}. The room has {places}.", _S),)),
    PhraseSet("d_put", (Phrase("{a}: I'm putting the {o} in the {p}.", _D | {"location"}),)),
    PhraseSet("d_move", (Phrase("{a}: I'm moving the {o} from the {s} to the {p}.", _D | {"location"}),)),
    PhraseSet("d_fb_put", (Phrase("{a}: Before that, I had put the {o} in the {p}.", _D | {"location"}),)),
    PhraseSet("d_intend", (Phrase("{a}: I was going to move the {o} to the {p}, but never mind.", _D),)),
    PhraseSet("d_watch", (Phrase("{a}: I saw that, {l}.", _D),)),
    PhraseSet("d_leave", (Phrase("{a}: I'm heading out for a bit.", _D),)),
    PhraseSet("d_return", (Phrase("{a}: I'm back.", _D),)),
    PhraseSet("d_claim", (Phrase("{a}: {l}, the {o} is in the {p}.", _D | {"speech", "location"}),)),
    PhraseSet("d_motive", (Phrase("[{a} wants to keep the {o} hidden from {l}.]", _E),)),
    PhraseSet("d_mention", (Phrase("{a}: Let me get something else from the {p}.", _D),)),
    PhraseSet("d_chat", (Phrase("{a}: I'll make some tea.", _D), Phrase("{a}: I'm going to read for a while.", _D),
                         Phrase("{a}: Nice weather today.", _D))),
    # probe leads: fixed classes (neutral), paraphrased by rendering
    PhraseSet("q_situation", (Phrase("So the {o} is in the", _S | {"location"}), Phrase("The {o} is in the", _S | {"location"}),
                              Phrase("Where is the {o}? It is in the", _S | {"location"}))),
    PhraseSet("q_belief", (Phrase("{a} will look for the {o} in the", frozenset({"belief", "narration"})),
                           Phrase("{a} thinks the {o} is in the", frozenset({"belief", "narration"})),
                           Phrase("{a} expects to find the {o} in the", frozenset({"belief", "narration"})))),
    PhraseSet("q_knot_first", (Phrase("The one who stowed the {o} in the {p} was", _S),
                               Phrase("The {o} was stowed in the {p} by", _S),
                               Phrase("The person who stowed the {o} in the {p} was", _S))),
    PhraseSet("q_knot_latest", (Phrase("The one who carried the {o} to the {p} was", _S),
                                Phrase("The {o} was carried to the {p} by", _S),
                                Phrase("The person who carried the {o} to the {p} was", _S))),
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
    events: list[dict[str, Any]]      # in the order they happen (the simulator's order)
    probes: list[Probe]
    narration: list[int] = field(default_factory=list)   # event indices in the order they are told
    flashback: list[int] = field(default_factory=list)   # events told in the past perfect, after a later event
    meta: dict[str, Any] = field(default_factory=dict)

    def event_list(self) -> list[Event]:
        return [Event(**e) for e in self.events]

    def told(self) -> list[tuple[int, Event]]:
        evs = self.event_list()
        return [(i, evs[i]) for i in (self.narration or range(len(evs)))]


# ── arms ──────────────────────────────────────────────────────────────────────────────────────────────────────────────

def _skeleton(arm: Arm, A: str, B: str, p0: str, p1: str, p2: str, prelude: bool,
              rng: random.Random) -> tuple[list[Event], Optional[list[int]], list[int]]:
    """The arm's events (in the order they happen), its narration order (None: as they happen) and flashback events.
    ``prelude``: B first puts the object in ``p2`` and A moves it into ``p0`` (instead of A putting it there)."""
    first = [Event("put", B, place=p2), Event("move", A, place=p0, src=p2)] if prelude else [Event("put", A, place=p0)]
    move = Event("move", B, place=p1, src=p0)
    filler = Event("chat", rng.choice([A, B]))
    if arm == "U":
        return first + [move, filler], None, []
    if arm == "N":
        return first + [filler, Event("chat", B if filler.actor == A else A)], None, []
    if arm == "R":
        return first + [move, Event("mention", rng.choice([A, B]), place=p0)], None, []
    if arm == "O":
        evs = [Event("put", A, place=p0), move, filler]        # no prelude: the flashback is one past-perfect put
        return evs, [1, 0, 2], [0]
    if arm == "F":
        return first + [Event("intend", B, place=p1), filler], None, []
    if arm == "W":
        return first + [move, Event("watch", A, listener=B), filler], None, []
    away = first + [Event("leave", A), move, Event("return", A)]
    if arm == "B":
        return away, None, []
    if arm == "T":
        return away + [Event("claim", B, place=p1, listener=A)], None, []
    if arm == "D":
        return away + [Event("motive", B, listener=A), Event("claim", B, place=p0, listener=A)], None, []
    raise ValueError(f"unknown arm {arm!r}")


def build(item_id: str, arm: Arm, rng: random.Random) -> Item:
    women, men = [n for n in NAMES if n.gender == "f"], [n for n in NAMES if n.gender == "m"]
    A, B = (x.word for x in rng.sample([rng.choice(women), rng.choice(men)], 2))
    obj = draw(rng, OBJECTS, 1)[0].word
    places = [p.word for p in draw(rng, PLACES, 3)]
    p0, p1, p2 = rng.sample(places, 3)
    which = "latest" if arm in MOVE_ARMS and rng.random() < .5 else "first"
    prelude = arm not in ("N", "O") and rng.random() < P_FACTOR
    events, narration, flashback = _skeleton(arm, A, B, p0, p1, p2, prelude, rng)
    false_belief = arm in BELIEF_ARMS and arm not in BELIEF_IS_WORLD     # A must get no look at where the object is
    postlude = which == "latest" and arm != "O" and not false_belief and rng.random() < P_FACTOR
    if postlude:                      # A is about to move it again (to p0 or p2, never where it is) and doesn't
        events.append(Event("intend", A, place=rng.choice([p0, p2])))
    gold_p, qplace = (A, p0) if which == "first" else (B, p1)
    distractor = not (false_belief and which == "latest") and rng.random() < P_FACTOR
    if distractor:                    # the other person later glances into the queried place
        other = B if gold_p == A else A
        events.append(Event("mention", other, place=qplace))
    if narration is not None:
        narration = narration + list(range(len(narration), len(events)))
    st = check_story([A, B], places, events)
    people = rng.sample([A, B], 2)    # who is named first, and the knot candidates' order, are drawn
    probes = [Probe("situation", "q_situation", str(st.world), list(places), {"o": obj})]
    if arm in BELIEF_ARMS:
        bel = st.belief[A]
        if (bel == st.world) != (arm in BELIEF_IS_WORLD):
            raise AssertionError(f"{item_id}: {arm}: A's belief should "
                                 f"{'match' if arm in BELIEF_IS_WORLD else 'differ from'} the world")
        probes.append(Probe("belief", "q_belief", str(bel), list(places), {"o": obj, "a": A}))
    probes.append(Probe(f"knot_{which}", f"q_knot_{which}", gold_p, list(people), {"o": obj, "p": qplace}))  # type: ignore[arg-type]
    told = [events[i] for i in (narration or range(len(events)))]
    mentions = [x for e in told for x in ((e.src,) if e.kind == "move" else ()) + ((e.place,) if e.place else ())]
    meta = {"generation": GENERATION, "world": st.world, "belief": dict(st.belief), "last_place_mention": mentions[-1],
            "gold_is_last_place_mention": mentions[-1] == st.world, "n_changes": sum(e.kind in CHANGES for e in events),
            "belief_matches_world": st.belief[A] == st.world, "knot": which, "prelude": prelude,
            "postlude": postlude, "distractor": distractor, "out_of_order": bool(flashback)}
    return Item(item_id, arm, people, obj, places, [asdict(e) for e in events], probes,
                narration or [], flashback, meta)


# ── rendering ─────────────────────────────────────────────────────────────────────────────────────────────────────────

def _place_list(places: Sequence[str]) -> str:
    art = {p.word: p for p in PLACES}
    parts = [art[w].indefinite() for w in places]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def render_story(item: Item, r: int) -> str:
    if not 0 <= r < N_RENDERINGS:
        raise ValueError(f"rendering {r} (have {N_RENDERINGS})")
    dialogue = r >= NARRATION
    rng = random.Random(0)                     # unused: the rendering index pins every phrase
    pick = lambda name: (PHRASES["d_" + name].pick(rng, forbid=FORBID, index=0)[1] if dialogue
                         else PHRASES[name].pick(rng, forbid=FORBID, index=r)[1])
    out = [pick("setup").fill(people=f"{item.people[0]} and {item.people[1]}", places=_place_list(item.places))]
    n_chat = 0
    for i, e in item.told():
        slots = {"a": e.actor, "o": item.obj, "p": e.place or "", "s": e.src or "", "l": e.listener or ""}
        if e.kind == "chat":            # filler varies within a story, so two filler lines never repeat
            name = "d_chat" if dialogue else "chat"
            ph = PHRASES[name].phrases[(r + n_chat) % len(PHRASES[name].phrases)]
            n_chat += 1
        else:
            ph = pick(("fb_" if i in item.flashback else "") + e.kind)
        out.append(ph.fill(**{k: v for k, v in slots.items() if k in ph.slots}))
    return ("\n" if dialogue else " ").join(out)


def probe_prompt(item: Item, probe: Probe, r: int, with_answer: bool = False) -> str:
    """The story, then the probe's lead (narration: the same paragraph; transcript: a narrator line of its own; the
    transcript uses the first lead paraphrase)."""
    lead = PHRASES[probe.lead].pick(random.Random(0), forbid=FORBID, index=r if r < NARRATION else 0)[1]
    joiner = "\n" if r >= NARRATION else " "
    text = f"{render_story(item, r)}{joiner}{lead.fill(**probe.slots)}"
    return f"{text} {probe.gold}." if with_answer else text


# ── generation, rows, gates ───────────────────────────────────────────────────────────────────────────────────────────

def generate(per_arm: int, seed: int = DEFAULT_SEED, arms: Sequence[Arm] = ARMS) -> list[Item]:
    out = []
    for arm in arms:
        rng = random.Random(f"situation-g{GENERATION}:{seed}:{arm}")
        for k in range(per_arm * ARM_WEIGHT.get(arm, 1)):
            out.append(build(f"sit{GENERATION}-{arm}-{k:03d}", arm, rng))
    return out


def _row(it: Item, pr: Probe, r: int, cond: str, prompt: str) -> dict[str, Any]:
    return {"id": f"{it.item_id}|{pr.kind}|r{r}|{cond}", "prompt": prompt, "candidates": [" " + c for c in pr.candidates],
            "gold": pr.gold, "arm": it.arm, "probe": pr.kind, "rendering": r, "condition": cond, "item_id": it.item_id,
            "gold_is_last_place_mention": it.meta["gold_is_last_place_mention"] if pr.kind in ("situation", "belief") else None}


def rows(items: Sequence[Item]) -> list[dict[str, Any]]:
    """The 0shot rows: every item × probe × rendering (``fair`` comes from :func:`fair_rows`)."""
    return [_row(it, pr, r, "0shot", probe_prompt(it, pr, r)) for it in items for pr in it.probes for r in range(N_RENDERINGS)]


def _words(item: Item) -> set[str]:
    return set(item.people) | {item.obj} | set(item.places)


def fair_rows(items: Sequence[Item]) -> list[dict[str, Any]]:
    """The ``fair`` rows: per item × probe × rendering, 4 answered exemplars of the same probe kind (see the module
    docstring), drawn once per (item, probe kind) so every rendering shows the same exemplar stories."""
    out = []
    pool = [(it, pr) for it in items for pr in it.probes]
    for it in items:
        used = _words(it)
        for pr in it.probes:
            gpos = pr.candidates.index(pr.gold)
            cand = [(o, q) for o, q in pool if q.kind == pr.kind and o.item_id != it.item_id and not (_words(o) & used)]
            rng = random.Random(f"situation-fair:{it.item_id}:{pr.kind}")
            for _ in range(5000):
                pick = rng.sample(cand, FAIR_K)
                if (sum(o.arm == it.arm for o, _ in pick) <= 2
                        and sum(q.candidates.index(q.gold) == gpos for _, q in pick) <= 2):
                    break
            else:
                raise ValueError(f"{it.item_id}/{pr.kind}: no fair exemplar set")
            for r in range(N_RENDERINGS):
                shots = [probe_prompt(o, q, r, with_answer=True) for o, q in pick]
                out.append(_row(it, pr, r, "fair", SEPARATOR.join(shots + [probe_prompt(it, pr, r)])))
    return out


def gate(item: Item) -> None:
    """Refuse an item whose story or probes break the battery's rules (raises ``AssertionError``)."""
    evs = item.event_list()
    st = check_story(item.people, item.places, evs)
    sit = next(p for p in item.probes if p.kind == "situation")
    assert sit.gold == st.world, f"{item.item_id}: situation gold {sit.gold} != world {st.world}"
    for pr in item.probes:
        assert pr.gold in pr.candidates and len(set(pr.candidates)) == len(pr.candidates), item.item_id
    if item.narration:
        assert sorted(item.narration) == list(range(len(evs))), f"{item.item_id}: narration must tell every event once"
    told = [e for _, e in item.told()]
    placements = [e for e in told if e.kind in ("put", "move", "intend")]
    if item.arm in ("O", "F"):
        assert placements[-1].place != st.world, f"{item.item_id}: {item.arm}'s last told placement must not be the world"
    if item.arm == "R":
        assert item.meta["last_place_mention"] != st.world or item.meta["distractor"], f"{item.item_id}: R's last mention is the gold"
    if item.arm == "N":
        assert item.meta["n_changes"] == 1, f"{item.item_id}: N must not move the object"
    if item.arm == "D":
        claim = next(e for e in evs if e.kind == "claim")
        assert claim.place != st.world, f"{item.item_id}: D's claim must be false"
    if item.arm == "T":
        claim = next(e for e in evs if e.kind == "claim")
        assert claim.place == st.world, f"{item.item_id}: T's claim must be true"
    if item.arm in BELIEF_ARMS and item.arm not in BELIEF_IS_WORLD:
        # after returning, A (who left) never looks into or handles the object's place: a look would show it
        a = next(e.actor for e in evs if e.kind == "leave")
        back = max(i for i, e in enumerate(evs) if e.kind == "return")
        assert not any(e.actor == a and e.kind in ("mention", "intend") and e.place == st.world for e in evs[back + 1:]), \
            f"{item.item_id}: A looks into or handles the object's place after returning"
    if item.arm in ("B", "D", "T"):
        mv = next(i for i, e in enumerate(evs) if e.kind == "move" and e.actor != next(x for x in evs if x.kind == "leave").actor)
        assert any(e.kind == "leave" for e in evs[:mv]), f"{item.item_id}: A must be away for the move"
    bel = [p for p in item.probes if p.kind == "belief"]
    if item.arm in BELIEF_ARMS:
        assert len(bel) == 1 and (bel[0].gold == st.world) == (item.arm in BELIEF_IS_WORLD), item.item_id
    else:
        assert not bel, f"{item.item_id}: {item.arm} asks no belief"
    knot = next(p for p in item.probes if p.kind.startswith("knot"))
    q = knot.slots["p"]
    doers = {e.actor for e in evs if e.kind in ("put", "move") and e.place == q}
    assert doers == {knot.gold}, f"{item.item_id}: only the gold placed the object in the queried place ({doers})"
    for r in range(N_RENDERINGS):
        text = render_story(item, r)
        for p in item.places:
            assert f" {p}" in text, f"{item.item_id}/r{r}: place {p} never mentioned"
        for w in (" now ", " still ", " again ", " anymore", " stowed", " stow", " carried", " carry"):
            assert w not in f" {text} ", f"{item.item_id}/r{r}: banned word {w.strip()!r}"


def gate_set(items: Sequence[Item]) -> dict[str, dict[str, float]]:
    """Refuse a set where any surface policy reaches :data:`SHORTCUT_CEILING` κ on any probe kind; return the report."""
    from binding_bench.situation.shortcuts import report

    rep = report(items)
    over = {k: (p, x) for k, ps in rep.items() for p, x in ps.items() if x >= SHORTCUT_CEILING}
    if over:
        raise AssertionError(f"surface policies at or above κ {SHORTCUT_CEILING}: {over}")
    return rep


def emit(out: Path, per_arm: int = 100, seed: int = DEFAULT_SEED) -> Path:
    items = generate(per_arm, seed)
    for it in items:
        gate(it)
    rep = gate_set(items)
    out.mkdir(parents=True, exist_ok=True)
    (out / "items.jsonl").write_text("".join(json.dumps(asdict(it)) + "\n" for it in items))
    (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows(items) + fair_rows(items)))
    (out / "shortcuts.json").write_text(json.dumps(rep, indent=1))
    return out / "rows.jsonl"


def check_tokens(tokenizers: Sequence[str] = ()) -> dict[str, dict[str, int]]:
    """``{word: {tokenizer: n_tokens}}`` for every scored word (places, names) that is not one token after a leading
    space. ``tokenizers``: grid keys (``riders.items.GRID_TOKENIZERS``) or Hugging Face names / paths; all grid
    tokenizers when empty."""
    from transformers import AutoTokenizer  # deferred: heavy import

    from binding_bench.riders.items import GRID_TOKENIZERS

    keys = list(tokenizers) or list(GRID_TOKENIZERS)
    words = [p.word for p in PLACES] + [n.word for n in NAMES]
    bad: dict[str, dict[str, int]] = {}
    for k in keys:
        tok = AutoTokenizer.from_pretrained(GRID_TOKENIZERS.get(k, k))
        for w in words:
            n = len(tok(" " + w, add_special_tokens=False).input_ids)
            if n != 1:
                bad.setdefault(w, {})[k] = n
    return bad
