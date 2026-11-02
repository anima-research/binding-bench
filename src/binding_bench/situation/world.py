"""The situation simulator: where an object is, and where each person believes it is, through a short story.

One object, two or three people, three places (containers that all take "in", so a probe's preposition never
narrows the candidates). Events:

  put      actor puts the object in a place                    world := place; everyone present sees it
  move     actor moves it from one place to another            world := dst; everyone present sees it
  leave    actor goes out                                      they stop seeing events
  return   actor comes back                                    (beliefs don't update by returning)
  claim    speaker tells listener it is in a place             listener believes it, true or not
  reveal   a person sees where it really is                    their belief := world
  mention  a person uses a place for something else            nothing changes (the recency trap)
  chat     neutral filler                                      nothing changes

The simulator is the only source of golds: :func:`simulate` replays the events and :func:`check_story` refuses a
story whose events contradict themselves (a move from where the object isn't, an absent actor acting, …).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

EventKind = Literal["put", "move", "leave", "return", "claim", "reveal", "mention", "chat"]
CHANGES: frozenset[str] = frozenset({"put", "move"})


class StoryError(ValueError):
    """A story whose events cannot all happen."""


@dataclass(frozen=True)
class Event:
    kind: EventKind
    actor: str
    place: Optional[str] = None      # put / mention / claim / reveal: the place; move: the destination
    src: Optional[str] = None        # move: where it was
    listener: Optional[str] = None   # claim: who is told


@dataclass
class State:
    world: Optional[str] = None
    belief: dict[str, Optional[str]] = field(default_factory=dict)
    present: set[str] = field(default_factory=set)
    place_mentions: list[str] = field(default_factory=list)    # every place mention, in story order


def simulate(people: list[str], events: list[Event]) -> State:
    """Replay ``events`` from a start where everyone is present and nothing is placed."""
    st = State(belief={p: None for p in people}, present=set(people))
    for k, ev in enumerate(events):
        where = f"event {k} ({ev.kind} by {ev.actor})"
        if ev.actor not in st.belief:
            raise StoryError(f"{where}: {ev.actor!r} is not in the story")
        acting = ev.kind not in ("return",)
        if acting and ev.actor not in st.present:
            raise StoryError(f"{where}: {ev.actor} is not there")
        if ev.kind == "put":
            if st.world is not None or ev.place is None:
                raise StoryError(f"{where}: put needs an unplaced object and a place")
            st.world = ev.place
            for p in st.present:
                st.belief[p] = ev.place
        elif ev.kind == "move":
            if ev.src != st.world or ev.place is None or ev.place == ev.src:
                raise StoryError(f"{where}: move {ev.src} -> {ev.place}, but the object is in {st.world}")
            st.world = ev.place
            for p in st.present:
                st.belief[p] = ev.place
        elif ev.kind == "leave":
            st.present.discard(ev.actor)
        elif ev.kind == "return":
            if ev.actor in st.present:
                raise StoryError(f"{where}: {ev.actor} never left")
            st.present.add(ev.actor)
        elif ev.kind == "claim":
            if ev.listener not in st.belief or ev.listener == ev.actor or ev.listener not in st.present:
                raise StoryError(f"{where}: claim to {ev.listener!r}, who isn't there to hear it")
            st.belief[ev.listener] = ev.place
        elif ev.kind == "reveal":
            if st.world is None:
                raise StoryError(f"{where}: nothing to reveal")
            if ev.place != st.world:
                raise StoryError(f"{where}: a reveal shows where the object is ({st.world}), not {ev.place}")
            st.belief[ev.actor] = st.world
        elif ev.kind == "mention":
            if ev.place is None:
                raise StoryError(f"{where}: mention needs a place")
        elif ev.kind != "chat":
            raise StoryError(f"{where}: unknown event kind")
        if ev.kind == "move" and ev.src is not None:
            st.place_mentions.append(ev.src)
        if ev.place is not None:
            st.place_mentions.append(ev.place)
    return st


def check_story(people: list[str], places: list[str], events: list[Event]) -> State:
    """:func:`simulate`, plus: every place an event names is one of ``places``, and the object ends placed."""
    for ev in events:
        for p in (ev.place, ev.src):
            if p is not None and p not in places:
                raise StoryError(f"{ev.kind} names {p!r}, not one of the story's places {places}")
    st = simulate(people, events)
    if st.world is None:
        raise StoryError("the object is never placed")
    return st
