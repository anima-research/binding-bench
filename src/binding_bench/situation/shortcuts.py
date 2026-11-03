"""Surface policies for the situation battery: rules that answer a probe from the told story's surface, with no model
and no simulation. Each is read in narration order, as a reader meets the text: a flashback counts where it is told,
and a foiled move ("is about to move the key to the pot") counts as a move, because its words are a move's.

A policy that cannot answer abstains and is scored at chance. :func:`report` gives each policy's chance-corrected
accuracy, (accuracy − chance) / (1 − chance), per probe kind; the generator's set gate refuses a set where any reaches
``items.SHORTCUT_CEILING``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Callable, Optional, Sequence

if TYPE_CHECKING:
    from binding_bench.situation.items import Item, Probe

PLACING = ("put", "move", "intend")
ACTING = ("put", "move", "intend", "claim")


def _told(it: "Item") -> list:
    return [e for _, e in it.told()]


def _mentions(it: "Item") -> list[tuple[str, str]]:
    """(actor, place) for every place mention in told order (a move mentions its source, then its destination)."""
    out = []
    for e in _told(it):
        if e.kind == "move" and e.src:
            out.append((e.actor, e.src))
        if e.place:
            out.append((e.actor, e.place))
    return out


def _placements(it: "Item") -> list:
    return [e for e in _told(it) if e.kind in PLACING]


PlacePolicy = Callable[["Item", "Probe"], Optional[str]]
PLACE_POLICIES: dict[str, PlacePolicy] = {
    "last_place_put_in": lambda it, pr: _placements(it)[-1].place if _placements(it) else None,
    "first_place_put_in": lambda it, pr: _placements(it)[0].place if _placements(it) else None,
    "last_place_mentioned": lambda it, pr: _mentions(it)[-1][1] if _mentions(it) else None,
    "first_place_listed": lambda it, pr: it.places[0],
}
PERSON_POLICIES: dict[str, PlacePolicy] = {
    "first_actor": lambda it, pr: next((e.actor for e in _told(it) if e.kind in ACTING), None),
    "last_actor": lambda it, pr: next((e.actor for e in reversed(_told(it)) if e.kind in ACTING), None),
    "first_named": lambda it, pr: it.people[0],
    "second_named": lambda it, pr: it.people[1],
    # the name beside the queried place's last mention: the actor of the last told event that names it
    "nearest_to_queried_place": lambda it, pr: next((a for a, p in reversed(_mentions(it)) if p == pr.slots.get("p")), None),
}


def _other(pol: PlacePolicy) -> PlacePolicy:
    """The complement of a person policy: with two candidates, a rule that is reliably wrong is a rule."""
    def f(it: "Item", pr: "Probe") -> Optional[str]:
        pick = pol(it, pr)
        rest = [c for c in pr.candidates if c != pick]
        return rest[0] if pick in pr.candidates and len(rest) == 1 else None
    return f


PERSON_POLICIES.update({f"not_{k}": _other(v) for k, v in list(PERSON_POLICIES.items())})


def policies_for(kind: str) -> dict[str, PlacePolicy]:
    return PERSON_POLICIES if kind.startswith("knot") else PLACE_POLICIES


def report(items: Sequence["Item"]) -> dict[str, dict[str, float]]:
    """``{probe kind: {policy: κ}}`` over the items (0shot; one answer per item and probe)."""
    acc: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for it in items:
        for pr in it.probes:
            chance = 1 / len(pr.candidates)
            for name, pol in policies_for(pr.kind).items():
                pick = pol(it, pr)
                acc[pr.kind][name].append((chance if pick not in pr.candidates else float(pick == pr.gold), chance))
    out: dict[str, dict[str, float]] = {}
    for kind, ps in acc.items():
        out[kind] = {}
        for name, xs in ps.items():
            c = sum(x for x, _ in xs) / len(xs)
            h = sum(y for _, y in xs) / len(xs)
            out[kind][name] = round((c - h) / (1 - h), 4)
    return out
