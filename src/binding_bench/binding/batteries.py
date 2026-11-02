"""The battery registry: what each battery scores, where its frozen prompts live, and which reader reads it.

Prompt files live under one prompts directory (``--prompts-dir``, else ``$KOTODAMA_BINDING_PROMPTS``; machine paths
belong in site.env). Every file is pinned by sha256, and a run refuses a file that does not match its pin, so a
number in the record always names the exact strings it was scored on. ``capacity-v1`` can be regenerated
(``python -m binding_bench.binding gen-capacity``), and the pin then checks that the regeneration is byte-identical;
so can every fluid battery (``gen-fluid``, :mod:`binding_bench.binding.fluid`) and both form-update versions
(``gen-formupdate``, :mod:`binding_bench.binding.formupdate`), and the stage gauges (``gen-stage``,
:mod:`binding_bench.binding.stagegauges`). The current bindprobes batteries come back from this package's generator
(``binding_bench.registry.BP_RECIPES``); bp-v3.1, which an older family version drew, from the generator that drew
it (``binding_bench.registry.HISTORICAL_RECIPES``). The ``fluid`` and ``formupdate`` groups name the current
versions, ``fluid-v1``, ``formupdate-v1`` and ``formupdate-v2`` the earlier ones. Files under ``r2/`` are the
current rendering of their battery; a file the battery pinned before stays at its earlier path, so a grid scored on
it still names its exact strings.

A bindprobes file stores item specs, and its prompts are rendered at scoring time, so its file pin alone does
not fix the strings: each bindprobes battery also carries ``render_sha256``, the digest of every prompt it
renders (:func:`binding_bench.bindprobes.items.render_sha256`), and a run refuses a file whose rendering under
the current code differs (:func:`check_render`). The families render each item with the templates of the
generation that built it, so a new wording leaves pinned files as they were scored.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

PROMPTS_ENV = "KOTODAMA_BINDING_PROMPTS"


@dataclass(frozen=True)
class Battery:
    name: str
    kind: Literal["rows", "bindprobes"]
    file: str                                   # relative to the prompts dir
    sha256: str
    reader: Optional[str]                       # readers.READERS key (None: read with bindprobes aggregate)
    what: str
    families: tuple[str, ...] = field(default_factory=tuple)     # bindprobes only
    conditions: tuple[str, ...] = field(default_factory=tuple)   # bindprobes only
    render_sha256: Optional[str] = None                          # bindprobes only: the rendered prompts


BATTERIES: dict[str, Battery] = {b.name: b for b in (
    Battery("joinlite-v1", "rows", "r2/joinlite-v1.jsonl",
            "82cd75aa52e389fbf9886181ed3e5cabaeccc9a5eda51838d96e6d692b8b048d", "joinlite",
            "stage gauges, static: lookup, I / you, P2 speaker swap"),
    Battery("swapdyn-v1", "rows", "r2/swapdyn-v1.jsonl",
            "fb508474604a56ce8b5b09def73a3f951c06a481758d2825ac54c5ca2b33b0c4", "swapdyn",
            "stage gauge, dynamic: speaker swap on transfer dialogues (0shot / 3shot)"),
    Battery("stage5-decomp-v1", "rows", "r2/stage5-decomp-v1.jsonl",
            "54773a3f926f26b4fbeb38556f9019246a3b8318c59fdbdc8f41fe1f16ce3887", "decomp",
            "stage-5 decomposition, all 6 variants × 0shot / 3shot"),
    Battery("stage5-decomp-core", "rows", "r2/stage5-decomp-core.jsonl",
            "2770697337b266dca03bc5d51526faed103f09022ca494b0dea0a7292abfe094", "decomp",
            "stage-5 decomposition, core variants (deictic / third / named_transfer)"),
    Battery("frame-ladder-v1", "rows", "r2/frame-ladder-v1.jsonl",
            "15ea952e32497a2320d9fb611acecc6181ae7bae65594c8f117e277533d60432", "frame",
            "frame ladder: the swapdyn 0-shot items with one cue per rung (R0 as built … R4 no I / you)"),
    Battery("capacity-v1", "rows", "r2/capacity-v1.jsonl",
            "602e08196d2c66e868c9b4b57138e8f9e6149a11a2520a0d0385f61310ca3e89", "capacity",
            "binding capacity (K50, after arXiv 2609.30634)"),
    Battery("fluid-v1-morph", "rows", "r2/fluid-v1-morph.jsonl",
            "c54e445339f73b429764fbf2cad1c6e31ffa089a9878770e2602a3bd9af6b43c", "fluid-morph",
            "fluid F1: nonce morphology (rule tier × examples)"),
    Battery("fluid-v1-algebra", "rows", "fluid-v1-algebra.jsonl",
            "81442d46aa9d0356d54f0676cef7e912d5f1ace36058f2fe74f41bfe8560ecf4", "fluid-algebra",
            "fluid F2: symbol algebra (operators × composition depth)"),
    Battery("fluid-v1-rebind", "rows", "fluid-v1-rebind.jsonl",
            "7431ebb1dc74dca1ab1c656456a7371bdf0d0300317dbeff0a84149626f3a543", "fluid-rebind",
            "fluid F3: rebinding chains, prose + code (reassignments × distractors; stale-value rate)"),
    Battery("fluid-v2-algebra", "rows", "r2/fluid-v2-algebra.jsonl",
            "5ef5202687676757d3032e2f6b25663a53ae2a5175092874409c306acc2f04d0", "fluid-algebra",
            "fluid F2 v2: symbol algebra, gold's place among the query operands controlled, one answer per row"),
    Battery("fluid-v2-rebind", "rows", "fluid-v2-rebind.jsonl",
            "4643c37d0366629beafa334bf051cadc515e5247e6c8a9a3801900ee8015d76c", "fluid-rebind",
            "fluid F3 v2: rebinding chains, recency control on the most recent candidate"),
    Battery("fluid-v1-seqind", "rows", "r2/fluid-v1-seqind.jsonl",
            "04c6c4b8e67cf4dd427e8299d3ce199404b415270d820a2bc6220c7f2a6dfb4e", "fluid-seqind",
            "fluid F4: nonce sequence induction (pattern length × gap)"),
    Battery("cryst-v1", "rows", "r2/cryst-v1.jsonl",
            "cdfdb91a7e57acd1a3d63b3283663485bac99c899ce027aacf49bafd7c181267", "cryst",
            "crystallized control C0: facts in the fluid few-shot frame (shots 6 / 3 / 1 / 0)"),
    Battery("formupdate-v1", "rows", "formupdate-v1.jsonl",
            "7222046c4c63a4eca769212c6cca327eb64401d4909db58e675dc16a31a17d1f", "formupdate",
            "forming vs updating: static / update-other / update-target, paired per base item (K × U × register)"),
    Battery("formupdate-v2", "rows", "formupdate-v2.jsonl",
            "b99fbc549f8cc2cc8d773fcb578320988867f4499ca88601d1ee73f73061efbe", "formupdate",
            "forming vs updating v2: the gold's first / middle / last mention position matched across A / B / C"),
    Battery("formupdate-v3", "rows", "r2/formupdate-v3.jsonl",
            "f70dad391bed59dbe161379f78d0c8e4bd7ce31225c32d0cc58919c87a0ed897", "formupdate",
            "forming vs updating v3: v2's items with one neutral line before the query, so no value sits next to it"),
    Battery("bp-deixis-lite", "bindprobes", "r2/bindprobes-deixis-lite/items.jsonl",
            "19549a5fd29ee4d470a8cc77db3f0c964726a020f6f000132f1757a548d30e4d", None,
            "bindprobes deixis-lite (0shot / 3shot)", ("deixis",), ("0shot", "3shot"),
            render_sha256="d30fa81fc4e581db6adaad17e50f53d8bdd341a35200bdd42c2702366647a2f5"),
    Battery("bp-profile", "bindprobes", "r2/bindprobes-items/items.jsonl",
            "191dd8cc1d94b56e1548bf7e359febd6fbb1821997d4eaf376fbcb96069926e5", None,
            "bindprobes profile (attr / codevar / dialogue …, all conditions)",
            render_sha256="cc4c9c28f518db2e58e43bb72dcfdd068cdce0563139409fdafd012a012b6468"),
    Battery("bp-v3.1", "bindprobes", "bindprobes-v3.1/items.jsonl",
            "bc3496d3c68990b8a615f485ca38e43b917deca1315cc6207990454be1330f37", None,
            "bindprobes v3.1: setbind / answer / adjacency / codevarmut",
            ("setbind", "answer", "adjacency", "codevarmut"),
            render_sha256="765f81db6c0d77fc7f833fb91c846116d610a61ef4c4304b17416f3a02fe41e1"),
    Battery("bp-v3.2", "bindprobes", "r2/bindprobes-v3.2/items.jsonl",
            "959b54969ac30401b3823f4dc2996b5c29f98a43a1e77867d166fcfe9bb6f88c", None,
            "bindprobes v3.2: setbind / answer / adjacency / codevarmut, controlled recency and role shortcuts",
            ("setbind", "answer", "adjacency", "codevarmut"),
            render_sha256="4a0d7bfcfb73baa9de87dff57048c23988858bceca5ffa9669455d0813f13610"),
    Battery("bp-deixis-v2", "bindprobes", "r2/bindprobes-deixis-v2-full/items.jsonl",
            "2a68cb383ca488287641b5fa6bbe08bf63f0a33c70421e9a326b3e9b71ac7b61", None,
            "bindprobes deixis v2 full", ("deixis",),
            render_sha256="454076829819f0a846917863a80b9bf5eed8e79f62944c583130f6e36e8336dc"),
)}

@dataclass(frozen=True)
class Pin:
    """Bytes a battery name was pinned to: its file under the prompts dir, sha256 and (bindprobes) render digest."""
    file: str
    sha256: str
    render_sha256: Optional[str] = None


# Every earlier pin of a battery whose bytes were upgraded in place under the same name, oldest first. A number
# recorded against one of these names before the upgrade resolves here to its exact file; the file stays at its path.
PREVIOUS_PINS: dict[str, tuple[Pin, ...]] = {
    "capacity-v1": (Pin("capacity-v1.jsonl", "c7241ec026e40054c21bef0ec44465848e3758da40ff7b183a7ada4a89e3c88d", None),),
    "joinlite-v1": (Pin("joinlite-v1.jsonl", "1e20c12961803a8a459b302d7ece30c34c7d4535f871330dba51020699facb90", None),),
    "swapdyn-v1": (Pin("swapdyn-v1.jsonl", "8ddab1262a2fa340e559fbbcd966b2c20a2acdfade2bb888e55034851810bdd3", None),),
    "stage5-decomp-v1": (Pin("stage5-decomp-v1.jsonl", "c08b40150b84e3e2e6d5e9918a8b7c6b5986230545f730b06b1cd2d8bc3db666", None),),
    "stage5-decomp-core": (Pin("stage5-decomp-core.jsonl", "f653040de69157ef6edb75e4702e00473ce229bc0e25dfe8cdc457a09ac3727e", None),),
    "frame-ladder-v1": (Pin("frame-ladder-v1.jsonl", "ba60a0789e74a131351780e87012d74ad0e13eb5a91c537b4849749707f8efba", None),),
    "fluid-v1-morph": (Pin("fluid-v1-morph.jsonl", "ec73660da59ba0893f5a65c6a7675105b1dbcc7c17166da255c9f2c6ccfaa341", None),),
    "fluid-v2-algebra": (Pin("fluid-v2-algebra.jsonl", "7b5bc3983583b6a87e6e26c9771f10d51ed38174c1eacc953e8ca8048cf45add", None),),
    "fluid-v1-seqind": (Pin("fluid-v1-seqind.jsonl", "b2a1e6b54040500363909cf4d669f84ef968a60df6353127599e9f041a46e15a", None),),
    "cryst-v1": (Pin("cryst-v1.jsonl", "e73e55699384eab1665b0dc7501a5f96f3ab793aeeca475bdc8089946783e5a2", None),),
    "formupdate-v3": (Pin("formupdate-v3.jsonl", "5cb5f48f34f10746c5c4427c15a8b4190c0b775018cbc2da681054cd89b4be87", None),),
    "bp-deixis-lite": (Pin("bindprobes-deixis-lite/items.jsonl", "229e7ac377ff2ec5ef298ecdee5c27667f9c97d86ba2337fb5e411aac9167c3e", '71fd836f43ed7732a17d70c738d4bacba7f6ae74b56692e438f45fe6eda4111a'),),
    "bp-profile": (Pin("bindprobes-items/items.jsonl", "7bfab8bff499667846fd9c9522cfbb4147476626bdd37bb7d702a2f55f4f9776", '58e1b17bf606dcfa6bcb336b86c4aca205bc546e550f5b92c65481da7818421b'),),
    "bp-v3.2": (Pin("bindprobes-v3.2/items.jsonl", "ee5e827bd4698623e96b906480a8652999c7d8884d69f783e29b072d8cfc13e5", '1d12e4f3a8702379aceef5b18593e8f10850054c4e27081dda73e62dd8abca3b'),
                Pin("r2/bindprobes-v3.2/items-r2a.jsonl", "7eb9da741d9df65176c7006bfaf5709beff42d4b2bca2c2d304da149b9d83e6f", "d6e2669b3e2deb72842bfac1338b88165d91a40267d9cba8299dccff216756c6")),
    "bp-deixis-v2": (Pin("bindprobes-deixis-v2-full/items.jsonl", "08151e66c4798288c7b1f471aaeb46d07e3d21acd5d279e8d8c137ba3bd35234", '2b3330a1c32ac92e8c3bbba8e4a8d50b1d8d3ca999c837347fbe95ae763c2345'),),
}


# Named sets for the CLI: `--batteries gauges,capacity` etc.
GROUPS: dict[str, tuple[str, ...]] = {
    "gauges": ("joinlite-v1", "swapdyn-v1"),
    "decomp": ("stage5-decomp-v1",),
    "frame": ("frame-ladder-v1",),
    "bindprobes": ("bp-deixis-lite", "bp-profile", "bp-v3.2", "bp-deixis-v2"),
    "scorecard": ("joinlite-v1", "swapdyn-v1", "stage5-decomp-v1", "capacity-v1"),
    "fluid": ("fluid-v1-morph", "fluid-v2-algebra", "fluid-v2-rebind", "fluid-v1-seqind", "cryst-v1"),
    "fluid-v1": ("fluid-v1-morph", "fluid-v1-algebra", "fluid-v1-rebind", "fluid-v1-seqind", "cryst-v1"),
    "formupdate": ("formupdate-v3",),
    "formupdate-v2": ("formupdate-v2",),
    "formupdate-v1": ("formupdate-v1",),
    "formupdate-v3": ("formupdate-v3",),
    "all": tuple(BATTERIES),
}


class PinError(RuntimeError):
    """A prompts file whose bytes do not match its registry pin."""


def resolve(names: str) -> list[Battery]:
    """``a,b,group`` → batteries, in order, deduplicated."""
    out: list[Battery] = []
    for n in filter(None, (x.strip() for x in names.split(","))):
        for m in GROUPS.get(n, (n,)):
            if m not in BATTERIES:
                raise KeyError(f"unknown battery {m!r} (have: {', '.join(BATTERIES)}; groups: {', '.join(GROUPS)})")
            if BATTERIES[m] not in out:
                out.append(BATTERIES[m])
    return out


def prompts_dir(arg: Optional[str]) -> Path:
    d = arg or os.environ.get(PROMPTS_ENV)
    if not d:
        raise SystemExit(f"no prompts dir: pass --prompts-dir or set ${PROMPTS_ENV} (site.env)")
    return Path(d)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_render(b: Battery, path: Path) -> None:
    """Raise :class:`PinError` unless a bindprobes file renders to its ``render_sha256`` pin (every item, every
    condition, under the current code); prompt-row batteries store their strings, so their file pin covers them."""
    if b.kind != "bindprobes":
        return
    if b.render_sha256 is None:
        raise PinError(f"{b.name}: a bindprobes battery needs a render_sha256 pin")
    from binding_bench.bindprobes.items import load_items, render_sha256

    try:
        got = render_sha256(load_items(Path(path)))
    except (ValueError, KeyError, TypeError) as exc:
        raise PinError(f"{b.name}: {path} does not render under this code: {exc}") from exc
    if got != b.render_sha256:
        raise PinError(f"{b.name}: {path} renders to sha256 {got[:16]}… ≠ render pin {b.render_sha256[:16]}… "
                       "(the prompt strings changed)")


def path_for(b: Battery, root: Path, check: bool = True) -> Path:
    p = Path(root) / b.file
    if not p.exists():
        raise FileNotFoundError(f"{b.name}: prompts file missing: {p}")
    if check and (got := sha256(p)) != b.sha256:
        raise PinError(f"{b.name}: {p} sha256 {got[:16]}… ≠ pin {b.sha256[:16]}… (the strings changed)")
    return p
