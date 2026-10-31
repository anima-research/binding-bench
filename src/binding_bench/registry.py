"""The benchmark registry: one row per eval, with what it measures and how far to trust each reading of it.

:data:`binding_bench.binding.batteries.BATTERIES` pins the bytes (file and render digests). This table holds the
validity facts beside them, keyed by eval:

* ``construct``: what the eval claims to measure, in the benchmark's construct vocabulary (:data:`CONSTRUCTS`).
* ``batteries``: the pinned batteries that carry it (none for riders, whose items are generated at scoring time).
* ``recipe``: how its prompts are made again, byte for byte: a generator in this repository, or for an older
  bindprobes version the generator that drew it (:data:`HISTORICAL_RECIPES`).
* ``clean``: the readings that count when two models are compared; every other reading is a diagnostic.
* ``floor``: the reading that bounds it from below (bare items, no worked examples), when there is one.
* ``flags``: audit findings, each scoped to the panel it was measured on. A flag says a reading is not a clean
  measure of the construct; it never says the model is weak.
* ``audit``: how far the rendering audit has reached this eval.

An eval changes only on validity grounds (rendering invariance, insensitivity to off-construct levers, item
shortcuts), never because a model scores low; a re-rendering has to shrink the spread across renderings, not raise
the score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Optional

Audit = Literal["audited", "item-audited", "pending"]

CONSTRUCTS: dict[str, str] = {
    "static-lookup": "retrieve which entity holds which attribute when nothing changes",
    "update": "track an entity's state through changes (transfers, reassignments, mutations)",
    "set-binding": "bind several attributes or members to one entity",
    "deixis": "resolve I / you / my to the right person in a dialogue",
    "stage-gauge": "the swap-shift gauges of the staircase: whether a pronoun's answer moves with the speaker",
    "capacity": "how many bindings fit in context before retrieval fails (K50)",
    "fluid": "learn a new rule from the examples in the prompt",
    "crystallized": "recall a known fact in the same few-shot frame (the control for fluid)",
    "form-vs-update": "the cost of updating a binding against forming it, paired per item",
}


@dataclass(frozen=True)
class Flag:
    reading: str        # the rendering or condition the finding is about
    finding: str
    panel: str          # where it holds, model-relative: "6/6 eligible models", "tsumugi"
    test: str           # the test that found it


@dataclass(frozen=True)
class Eval:
    name: str
    construct: str
    source: Literal["binding", "bindprobes", "riders"]
    what: str
    batteries: tuple[str, ...] = field(default_factory=tuple)
    recipe: str = ""
    clean: tuple[str, ...] = field(default_factory=tuple)
    floor: str | None = None
    flags: tuple[Flag, ...] = field(default_factory=tuple)
    audit: Audit = "pending"


# bp-v3.2's recipe: the v3 families at their current versions, emitted with this seed and per-cell sizes
# (binding_bench.bindprobes.items.emit); the emit is byte-identical to the pinned file.
BP_V32_RECIPE: Mapping[str, object] = {"seed": 2031, "per_cell": {"setbind": 64, "answer": 160, "adjacency": 225, "codevarmut": 100}}

_BP_CLEAN = ("0shot", "fair")   # bare items are the floor; `fair` (four exemplars balanced on every cell component) is the few-shot reading


@dataclass(frozen=True)
class EmitRecipe:
    """A bindprobes battery as kotodama's bindprobes generator emitted it, before the families' current versions: run
    ``kotodama.eval.bindprobes.items.emit(out, seed, list(per_cell), per_cell)`` from kotodama at ``kotodama_commit``
    (https://github.com/anima-research/kotodama), then keep only the items whose cell is in ``keep_cells`` when that
    is set, in emitted order. ``tools/replay_recipe.py`` does this and checks the pin."""
    kotodama_commit: str
    seed: int
    per_cell: Mapping[str, int]            # its key order is the families argument's order
    keep_cells: tuple[tuple[object, ...], ...] = ()


# The generators' current family versions draw different items, so these batteries come back only from the code that
# drew them. Each reproduces its pinned file byte for byte.
HISTORICAL_RECIPES: Mapping[str, EmitRecipe] = {
    "bp-profile": EmitRecipe("44cd0bc", 2026, {"attr": 110, "codevar": 100, "dialogue": 100}),
    "bp-deixis-v2": EmitRecipe("7a5a672", 2026, {"deixis": 60}),
    "bp-deixis-lite": EmitRecipe("8b86281", 113, {"deixis": 200}, ((3, 1, "owner"), (3, 2, "owner"))),
    "bp-v3.1": EmitRecipe("795b324", 2031, {"setbind": 64, "answer": 160, "adjacency": 150, "codevarmut": 100}),
}
_HISTORICAL = "HISTORICAL_RECIPES[{}] (tools/replay_recipe.py)"
_STAGE = "python -m binding_bench.binding gen-stage (swapdyn and stage5-decomp from bp-deixis-v2's items)"

REGISTRY: dict[str, Eval] = {e.name: e for e in (
    # ---- bindprobes: one family per construct, read under every condition of bindprobes.items.CONDITIONS
    Eval("bp-attr", "static-lookup", "bindprobes", "who holds which attribute, in prose", ("bp-profile",),
         _HISTORICAL.format("bp-profile"), _BP_CLEAN, "0shot",
         (Flag("0shot", "format-loaded: worked examples lift it by 8 to 52 points", "8-model panel",
               "rendering spread, 0shot vs fair"),),
         "audited"),
    Eval("bp-codevar", "update", "bindprobes", "a variable's value after reassignments, in code", ("bp-profile",),
         _HISTORICAL.format("bp-profile"), _BP_CLEAN, "0shot",
         (Flag("3shot", "shortcut: exemplars that match the item's cell can be copied", "8-model panel",
               "mixed-shot comparison"),),
         "audited"),
    Eval("bp-dialogue", "update", "bindprobes", "who has what after objects change hands in a dialogue", ("bp-profile",),
         _HISTORICAL.format("bp-profile"), _BP_CLEAN, "0shot", (), "audited"),
    Eval("bp-deixis", "deixis", "bindprobes", "person deixis: deictic, third-person and twin variants",
         ("bp-deixis-v2", "bp-deixis-lite"), _HISTORICAL.format("bp-deixis-v2 / bp-deixis-lite"), _BP_CLEAN, "0shot", (), "audited"),
    Eval("bp-setbind", "set-binding", "bindprobes", "several members bound to one set", ("bp-v3.2", "bp-v3.1"),
         "bp-v3.2: BP_V32_RECIPE; bp-v3.1: " + _HISTORICAL.format("bp-v3.1"), _BP_CLEAN, "0shot", (), "audited"),
    Eval("bp-answer", "deixis", "bindprobes", "production deixis: who answers, from the reply's own words",
         ("bp-v3.2", "bp-v3.1"), "bp-v3.2: BP_V32_RECIPE; bp-v3.1: " + _HISTORICAL.format("bp-v3.1"), _BP_CLEAN, "0shot",
         (Flag("3shot", "exemplars mislead: below 0shot", "8/8 models", "condition comparison"),),
         "audited"),
    Eval("bp-adjacency", "deixis", "bindprobes", "structural you: the reply's addressee", ("bp-v3.2", "bp-v3.1"),
         "bp-v3.2: BP_V32_RECIPE; bp-v3.1: " + _HISTORICAL.format("bp-v3.1"), _BP_CLEAN, "0shot", (), "audited"),
    Eval("bp-codevarmut", "update", "bindprobes", "aliased lists: a mutation seen through another name",
         ("bp-v3.2", "bp-v3.1"), "bp-v3.2: BP_V32_RECIPE; bp-v3.1: " + _HISTORICAL.format("bp-v3.1"), _BP_CLEAN, "0shot",
         (Flag("3shot", "shortcut: cell-matched exemplars teach the answer's role; flipping their answers collapses "
                        "the score", "6/6 eligible models", "flipped-exemplar test"),),
         "audited"),
    # ---- the rows batteries of the binding suite
    Eval("gauges", "stage-gauge", "binding", "static and dynamic swap shifts: the staircase's gauges",
         ("joinlite-v1", "swapdyn-v1"), _STAGE, ("0shot",), "0shot",
         (Flag("joinlite third-person query", "off-construct: does not group with the other deixis readings under fair",
               "construct matrix, 8 and 20 models", "construct matrix"),
          Flag("joinlite third-person query (`Now X is holding the`)",
               "off-construct through its opening \"Now\", which reads as a change of state: strong models answer with an "
               "object that could have moved. Without \"Now\" (`X is holding the`, chat) Llama-3.1-8B goes 37 -> 91 and "
               "OLMo-2-7B 52 -> 91, and the slice loads with the other static slices; that wording needs a new battery "
               "version. The I / you queries' opening \"Right now\" carries no such cue: dropping it moves no "
               "I / you slice by 20 points and no swap-gauge reading, on 8/8 models",
               "4/8 models (OLMo-2-1B, Qwen2.5-7B, Llama-3.1-8B, OLMo-2-7B)",
               "rendering spread, frame (chat, scene) x query (with, without \"Now\")")),
         "audited"),
    Eval("stage5-decomp", "stage-gauge", "binding", "stage-5 decomposition: third / deictic / named transfer / vocative",
         ("stage5-decomp-v1", "stage5-decomp-core"), _STAGE, ("0shot",), "0shot",
         (Flag("deictic variant", "byte-identical to swapdyn-v1's original rows: one measurement, not two", "all models",
               "item comparison"),),
         "audited"),
    Eval("frame-ladder", "deixis", "binding", "the cold two-speaker 'you' with one cue per rung: scene line, an addressee tag "
         "on every line, the tag's format without the addressee, transfers narrated, no I / you",
         ("frame-ladder-v1",), _STAGE, ("0shot, per rung",), "0shot", (), "pending"),
    Eval("capacity", "capacity", "binding", "binding capacity, K50 after arXiv 2609.30634", ("capacity-v1",),
         "python -m binding_bench.binding gen-capacity", ("K50 on the `Name: task` frame, named with it",), None,
         (Flag("K50", "format-loaded: the same bindings as prose sentences cost 21 to 25 points at no filler, and K50 "
                      "moves with the frame (Llama-3.1-8B >64 -> 37.6, OLMo-2-7B >64 -> 17.5)",
               "3/8 models (SmolLM2-1.7B, Llama-3.1-8B, OLMo-2-7B)", "rendering spread, `Name: task` vs prose"),),
         "audited"),
    Eval("fluid", "fluid", "binding", "fluid F1-F4: nonce morphology, symbol algebra, rebinding chains, sequence induction",
         ("fluid-v1-morph", "fluid-v1-algebra", "fluid-v2-algebra", "fluid-v1-rebind", "fluid-v2-rebind", "fluid-v1-seqind"),
         "python -m binding_bench.binding gen-fluid", ("ladder AUC", "F4: ladder AUC beside the in-context n-gram's (.26)"),
         None,
         (Flag("F4 ladder AUC", "copy-solvable subset: an in-context n-gram (longest suffix match over the prompt) scores "
                                "AUC .26, from period rows read after a motif token (n-gram accuracy 1.00); mirror rows "
                                "defeat it. Every panel model is above it; Pythia-1.4B by .11, near chance off the "
                                "copy-solvable rows", "8/8 models above the baseline", "in-context n-gram baseline"),
          Flag("F1 vowel-change errors", "errors copy the stem syllable unchanged (the `unchanged` foil) far above the "
                                         "uniform share (.79 to 1.00 vs .25). Accuracy is invariant to the gold "
                                         "syllable's corpus frequency (resolved on 2/8; errors never follow frequency)",
               "8/8 models", "in-context n-gram foil; token-frequency invariance"),
          Flag("F2 errors", "errors retrieve the answer of an example equation that shares the query's operator and "
                            "right operand, above the uniform share (.29 to .39 vs .20); no operand control covers it. "
                            "Accuracy is invariant to the gold word's corpus frequency (0/8)",
               "8/8 models", "in-context n-gram foil; token-frequency invariance"),
          Flag("F3 prose errors", "errors pick the target's first value, from the introduction line, above the uniform "
                                  "share (.37 to .50 vs .25)",
               "4/8 models (Qwen3-1.7B, Pythia-1.4B, SmolLM2-1.7B, Llama-3.1-8B)", "in-context n-gram foil")),
         "audited"),
    Eval("crystallized", "crystallized", "binding", "C0: known facts in the fluid few-shot frame", ("cryst-v1",),
         "python -m binding_bench.binding gen-fluid", ("shots 1 / 3 / 6",), None,
         (Flag("shots 0", "frame-loaded: C0's bare rung moves by 24 points or more across the fluid families' frames. "
                          "At shots 6 and 3 C0 holds in every frame, and a frame-matched C0 moves no family's fluid - "
                          "crystallized gap by 20 points", "8/8 models", "C0 in every fluid family's frame"),),
         "audited"),
    Eval("formupdate", "form-vs-update", "binding", "forming vs updating a binding, paired per base item",
         ("formupdate-v3", "formupdate-v2", "formupdate-v1"), "python -m binding_bench.binding gen-formupdate",
         ("paired cost, per register (formupdate-v3)",), None,
         (Flag("formupdate-v2 update cost (B - C), pooled over registers",
               "register-dependent: it moves by 20 points or more between prose and code, and in prose it reverses sign "
               "(updating the queried holder is easier than updating the others)",
               "6/8 models; sign reversed on 4/8 (tsumugi, OLMo-2-1B, Qwen2.5-7B, OLMo-2-7B)",
               "item-matched register swap"),
          Flag("formupdate-v2 update cost at U = 1, prose",
               "shortcut: C puts the gold on the line next to a query that repeats the holder, so C reads as a copy of "
               "that line. One neutral line before the query (formupdate-v3, the same items) raises the prose U = 1 cost "
               "by 20 points or more (tsumugi -57 -> -6, Llama-3.1-8B -54 -> 0); OLMo-2-1B, Qwen2.5-7B and OLMo-2-7B "
               "stay below -45 there",
               "6/8 models", "neutral line before the query, v2 vs v3 on the same items"),
          Flag("formupdate-v2 form accuracy (A), pooled over registers",
               "register-dependent: code above prose by 20 points or more",
               "3/8 models (OLMo-2-7B, Llama-3.1-8B, Pythia-1.4B)", "item-matched register swap")),
         "audited"),
    # ---- riders: generated at scoring time from binding_bench.riders (the in-loop slice: riders.lite)
    Eval("riders", "update", "riders", "entity-state tracking: boxes and operations, the same items in four registers",
         (), "binding_bench.riders.lite.select_items (seeded); no prompts file", ("0shot, per register", "3shot, code"),
         "0shot",
         (Flag("0shot", "register-dependent: the registers spread by 20 points or more", "5/8 models",
               "four-register spread"),
          Flag("pooled prose (narrative + screenplay + chat)", "pooling hides a format term of up to 40 points", "4/8 models",
               "four-register spread"),
          Flag("3shot, prose registers", "fixed exemplars hurt prose and help code", "chat 8/8, screenplay 6/8, narrative 5/8; "
               "code helped 8/8", "condition comparison")),
         "audited"),
)}



@dataclass(frozen=True)
class SliceReliability:
    """How well one slice separates models on one panel: split-half reliability across the panel's models (items halved
    by a hash of their id, Spearman-Brown corrected), the SD of chance-normalized accuracy across models, and the share
    of models whose accuracy CI includes chance. These describe the panel, not the eval: a slice at chance on most of
    it is a floor of these models, and ``large_model_check`` names the check that tells the two apart."""
    eval: str
    reading: str
    reliability: float
    sd: float
    floor_share: float
    large_model_check: Optional[str] = None     # set while a large-model check of this slice is pending or done

    @property
    def informative_on_panel(self) -> bool:
        """Reliability >= .70 and at chance on at most half the panel: usable as a marker on THIS panel."""
        return self.reliability >= 0.70 and self.floor_share <= 0.50


RELIABILITY_PANEL = "20 base models, 0.5B-8B (Llama 3 / 3.1 / 3.2, OLMo 1 / 2, Pythia, Qwen 2.5 / 3, SmolLM2, tsumugi)"
# A slice at chance on most of the 0.5-8B panel is checked on a large base model before anything is said about the
# slice itself.
LARGE_MODEL_CHECK = "pending: score on a large base model (a 405B-class model) before reading the panel floor as the eval's"
# The construct-matrix slices on RELIABILITY_PANEL: bindprobes slices read under `fair`, prompt rows as scored.
PANEL_RELIABILITY: dict[str, SliceReliability] = {r.reading: r for r in (
    SliceReliability("bp-attr", "attr", 1.00, 0.239, 0.00),
    SliceReliability("capacity", "capacity K<=4, no filler", 0.99, 0.162, 0.00),
    SliceReliability("formupdate", "formupdate-v2 A", 1.00, 0.211, 0.00),
    SliceReliability("bp-setbind", "setbind, one object", 0.96, 0.265, 0.00),
    SliceReliability("gauges", "joinlite third-person query", 0.98, 0.207, 0.00),
    SliceReliability("bp-codevar", "codevar", 0.99, 0.193, 0.20),
    SliceReliability("bp-dialogue", "dialogue", 0.99, 0.191, 0.00),
    SliceReliability("bp-codevarmut", "codevarmut", 0.99, 0.198, 0.15),
    SliceReliability("bp-setbind", "setbind, two transfers", 0.98, 0.212, 0.00),
    SliceReliability("fluid", "F3 rebind, R >= 1", 0.98, 0.111, 0.05),
    SliceReliability("formupdate", "formupdate-v2 C", 0.98, 0.159, 0.00),
    SliceReliability("bp-deixis", "deixis-v2 addressee, deictic", 0.46, 0.049, 0.85,
                     "done: above chance at scale, Llama-3.1 base 8B +.06 (chance), 70B +.45, 405B +.60 [+.53, +.67]; "
                     "the panel floor is the small models'"),
    SliceReliability("gauges", "joinlite addressee", 0.98, 0.271, 0.00),
    SliceReliability("gauges", "swapdyn addressee, 0shot", 0.73, 0.062, 0.60,
                     "done: at chance on Llama-3.1-405B base (+.07 [-.03, +.17]); the same items lift with 3 shots "
                     "(+.30); pending: a validity review of the 0-shot frame before the slice is read"),
)}

def evals_for(battery: str) -> list[Eval]:
    """The registry rows a pinned battery carries."""
    return [e for e in REGISTRY.values() if battery in e.batteries]
