"""Typed word pools and tagged paraphrase sets: the shared machinery a rendering-aware generator draws from.

A generator never writes a free string. It asks for a word with the features a slot needs, and for a phrase from a
paraphrase set whose tags fit the context, so a combination that would read wrong cannot be built:

* :class:`Noun` carries its grammar (count or mass, the article it takes, the preposition a place takes) and its
  kind (``object``, ``place``); :class:`Name` carries a gender and a frequency band, so an arm can draw from one band.
* :class:`Phrase` is a template with named slots and a set of tags saying what it presupposes or states
  (``implies_change``, ``holding``, ``ownership``, ``location``, ``matches_query``, …).
* :class:`PhraseSet` is one slot's paraphrases. :meth:`PhraseSet.pick` takes the tags a context requires and the tags
  it forbids, and refuses when no phrase fits, rather than falling back to one that doesn't.

The rule the tags enforce comes from the template audit: a phrase that implies a change appears only where the
simulator records one, and a query lead's class is fixed per battery so the lead never signals the answer.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from typing import Iterable, Literal, Mapping, Optional, Sequence

Kind = Literal["object", "place"]
Band = Literal["common", "mid", "rare"]
Gender = Literal["f", "m"]

TAGS: frozenset[str] = frozenset({
    "implies_change",    # "now", "still", "again", "anymore": the sentence presupposes a change or its absence
    "state",             # states a standing fact (vs an event)
    "event",             # reports something happening
    "holding",           # possession as holding / having in hand
    "ownership",         # possession as owning ("mine", "belongs to")
    "location",          # where something is
    "matches_query",     # uses the probe's own verb / frame word for word
    "belief",            # what a person thinks or expects
    "speech",            # a character says it
    "narration",         # register: narrator prose
    "dialogue",          # register: a character's line
})


class LexiconError(ValueError):
    """A request no word or phrase in the lexicon can satisfy."""


@dataclass(frozen=True)
class Noun:
    word: str
    kind: Kind
    count: bool = True                 # singular count noun ("a cup") vs mass / plural form ("the glue", "the dice")
    article: Optional[str] = None      # "a" / "an" for count nouns; None for mass
    prep: str = "in"                   # a place's preposition ("in the drawer", "on the shelf")
    band: Band = "mid"

    def __post_init__(self) -> None:
        if not self.word or any(c in string.whitespace for c in self.word):
            raise LexiconError(f"noun {self.word!r}: one word, no spaces")
        if self.count and self.article not in ("a", "an"):
            raise LexiconError(f"count noun {self.word!r} needs an article (a / an)")
        if not self.count and self.article is not None:
            raise LexiconError(f"mass noun {self.word!r} takes no indefinite article")
        if self.prep not in ("in", "on", "under", "behind"):
            raise LexiconError(f"noun {self.word!r}: unknown preposition {self.prep!r}")

    def indefinite(self) -> str:
        if not self.count:
            raise LexiconError(f"{self.word!r} takes no indefinite article")
        return f"{self.article} {self.word}"


@dataclass(frozen=True)
class Name:
    word: str
    gender: Gender
    band: Band = "mid"


@dataclass(frozen=True)
class Phrase:
    """A template over ``{slot}`` fields, with the tags of what it presupposes or states."""
    template: str
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        unknown = set(self.tags) - TAGS
        if unknown:
            raise LexiconError(f"phrase {self.template!r}: unknown tags {sorted(unknown)}")

    @property
    def slots(self) -> tuple[str, ...]:
        return tuple(f for _, f, _, _ in string.Formatter().parse(self.template) if f)

    def fill(self, **values: str) -> str:
        missing = set(self.slots) - set(values)
        if missing:
            raise LexiconError(f"phrase {self.template!r}: missing slots {sorted(missing)}")
        return self.template.format(**values)


@dataclass(frozen=True)
class PhraseSet:
    """One slot's paraphrases. Every phrase in a set has the same slots, so any can stand in for any other."""
    name: str
    phrases: tuple[Phrase, ...]

    def __post_init__(self) -> None:
        if not self.phrases:
            raise LexiconError(f"phrase set {self.name!r} is empty")
        slots = {frozenset(p.slots) for p in self.phrases}
        if len(slots) != 1:
            raise LexiconError(f"phrase set {self.name!r}: phrases disagree on slots {sorted(map(sorted, slots))}")

    def fitting(self, require: Iterable[str] = (), forbid: Iterable[str] = ()) -> tuple[int, ...]:
        req, forb = frozenset(require), frozenset(forbid)
        for t in req | forb:
            if t not in TAGS:
                raise LexiconError(f"unknown tag {t!r}")
        return tuple(i for i, p in enumerate(self.phrases) if req <= p.tags and not (forb & p.tags))

    def pick(self, rng: random.Random, require: Iterable[str] = (), forbid: Iterable[str] = (),
             index: Optional[int] = None) -> tuple[int, Phrase]:
        """A fitting phrase (``index`` pins one: a rendering that fixes the paraphrase across an item). Refuses when
        nothing fits, or when the pinned phrase doesn't."""
        ok = self.fitting(require, forbid)
        if not ok:
            raise LexiconError(f"phrase set {self.name!r}: no phrase with {sorted(require)} and without "
                               f"{sorted(forbid)}")
        if index is not None:
            if index not in ok:
                raise LexiconError(f"phrase set {self.name!r}: phrase {index} does not fit {sorted(require)} / "
                                   f"not {sorted(forbid)}")
            return index, self.phrases[index]
        i = rng.choice(ok)
        return i, self.phrases[i]


def draw(rng: random.Random, pool: Sequence[Noun] | Sequence[Name], k: int, *, band: Optional[Band] = None,
         exclude: Iterable[str] = (), **features: object) -> list:
    """``k`` distinct entries of ``pool`` with the given features (and band), none of ``exclude``."""
    ex = set(exclude)
    ok = [w for w in pool if w.word not in ex and (band is None or w.band == band)
          and all(getattr(w, f) == v for f, v in features.items())]
    if len(ok) < k:
        raise LexiconError(f"only {len(ok)} entries with band={band} {features} (need {k})")
    return rng.sample(ok, k)


def check_pools(pools: Mapping[str, Sequence[Noun] | Sequence[Name]]) -> None:
    """Every word appears once across all pools (no word is both an object and a place, or a name and a noun)."""
    seen: dict[str, str] = {}
    for pname, pool in pools.items():
        for w in pool:
            key = w.word.lower()
            if key in seen:
                raise LexiconError(f"{w.word!r} is in both {seen[key]!r} and {pname!r}")
            seen[key] = pname
