"""Model specs for the binding suite: a Hugging Face model, or a checkpoint format a plugin provides, loaded once per run.

Spec strings (CLI ``--model``):
    hf:MODEL                (a Hugging Face repo id or a local model directory)
    koto:CHECKPOINT[?size=1b-tsumugi][&bounds=0,1,7,…][&attn_res=config|on|off]

``hf:`` is built in. Every other kind is loaded by a plugin: a callable ``(ModelSpec, device) -> LoadedModel``. A
plugin is found in one of three places, and finding two different ones for a kind is refused:

* :func:`register_loader`, called by the program before it loads a model;
* ``$BINDING_BENCH_LOADERS``, a comma list of ``kind=module:attr`` (for a codebase on ``PYTHONPATH`` that is not
  installed as a package);
* the entry-point group named by :data:`LOADER_GROUP`, with the kind as its name (for an installed package).

kotodama provides ``koto``, which loads a kotodama checkpoint with the same forward its stage-gauge rows were
scored with.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

import torch

LogitsFn = Callable[[torch.Tensor], torch.Tensor]


class SpecError(ValueError):
    """A model spec string that cannot be parsed."""


@dataclass(frozen=True)
class ModelSpec:
    kind: Literal["koto", "hf"]
    path: str
    size: str = "1b-tsumugi"
    bounds: Optional[str] = None
    attn_res: Literal["config", "on", "off"] = "config"

    @classmethod
    def parse(cls, spec: str) -> "ModelSpec":
        kind, sep, rest = spec.partition(":")
        if not sep or not rest:
            raise SpecError(f"model spec must be koto:PATH[?k=v&…] or hf:PATH, got {spec!r}")
        path, _, query = rest.partition("?")
        if kind == "hf":
            if query:
                raise SpecError(f"hf specs take no options, got {spec!r}")
            return cls(kind="hf", path=path)
        if kind != "koto":
            raise SpecError(f"model spec kind must be koto or hf, got {kind!r}")
        kw: dict[str, str] = {}
        for part in filter(None, query.split("&")):
            k, eq, v = part.partition("=")
            if not eq or k not in ("size", "bounds", "attn_res"):
                raise SpecError(f"bad koto option {part!r} in {spec!r} (size= bounds= attn_res=)")
            kw[k] = v
        if kw.get("attn_res", "config") not in ("config", "on", "off"):
            raise SpecError(f"attn_res must be config|on|off in {spec!r}")
        return cls(kind="koto", path=path, size=kw.get("size", cls.size), bounds=kw.get("bounds"),
                   attn_res=kw.get("attn_res", "config"))  # type: ignore[arg-type]


@dataclass
class LoadedModel:
    spec: ModelSpec
    logits_fn: LogitsFn
    tokenizer: Any
    device: torch.device
    max_seq_len: Optional[int]


VRAM_BUDGET_ENV = "HEIMDALL_VRAM_BUDGET_GB"


def cap_cuda_memory() -> Optional[float]:
    """Hold this process to its declared GPU budget (``$HEIMDALL_VRAM_BUDGET_GB``, set by the scheduler for a job
    that declares one): the caching allocator then frees and retries before it grows past the budget, so models that
    share a GPU stay within their shares, and a model that truly needs more fails with its own out-of-memory error.
    Returns the fraction set on each visible device, or None when no budget is declared."""
    raw = os.environ.get(VRAM_BUDGET_ENV, "")
    try:
        budget = float(raw) if raw else 0.0
    except ValueError as e:
        raise ValueError(f"{VRAM_BUDGET_ENV}={raw!r} is not a number of GB") from e
    if budget <= 0 or not torch.cuda.is_available():
        return None
    frac = 1.0
    for i in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(i).total_memory
        frac = min(1.0, budget * 1024 ** 3 / total)
        torch.cuda.set_per_process_memory_fraction(frac, i)
    return frac


def load(spec: ModelSpec, device: str = "cuda") -> LoadedModel:
    if device != "cpu":
        cap_cuda_memory()
    if device == "cpu":   # torch's thread pools ignore OMP_NUM_THREADS; cap them on a shared node
        torch.set_num_threads(int(os.environ.get("SCORE_THREADS", "8")))
        torch.set_num_interop_threads(1)
    if spec.kind == "koto":
        return _load_koto(spec, device)
    return _load_hf(spec, device)


LOADER_GROUP = "binding_bench.loaders"
LOADERS_ENV = "BINDING_BENCH_LOADERS"

Loader = Callable[["ModelSpec", str], "LoadedModel"]
_REGISTERED: dict[str, Loader] = {}


class LoaderMissing(RuntimeError):
    """A model spec whose kind no plugin loads, or whose kind two different plugins claim."""


def register_loader(kind: str, loader: Loader) -> None:
    """Make ``loader`` the plugin for ``kind:`` specs in this process. Registering the same callable again is a no-op;
    a different one for a registered kind is refused."""
    if kind == "hf":
        raise LoaderMissing("hf: is built in and takes no plugin")
    held = _REGISTERED.get(kind)
    if held is not None and held is not loader:
        raise LoaderMissing(f"a different loader is already registered for {kind!r}: {held!r}")
    _REGISTERED[kind] = loader


def _resolve(target: str) -> Loader:
    from importlib import import_module

    module, sep, attr = target.partition(":")
    if not sep or not module or not attr:
        raise LoaderMissing(f"${LOADERS_ENV} entry {target!r} must be module:attr")
    obj: Any = import_module(module)
    for part in attr.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise LoaderMissing(f"${LOADERS_ENV} target {target!r} is not callable")
    return obj


def _env_loaders() -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in filter(None, (x.strip() for x in os.environ.get(LOADERS_ENV, "").split(","))):
        kind, eq, target = entry.partition("=")
        if not eq or not kind or not target:
            raise LoaderMissing(f"${LOADERS_ENV} entry {entry!r} must be kind=module:attr")
        if kind in out and out[kind] != target:
            raise LoaderMissing(f"${LOADERS_ENV} names two loaders for {kind!r}: {out[kind]}, {target}")
        out[kind] = target
    return out


def loader_for(kind: str) -> Loader:
    """The one plugin that loads ``kind:`` specs, from the registry, the environment or the entry points."""
    from importlib.metadata import entry_points

    found: dict[str, Loader] = {}
    if kind in _REGISTERED:
        found["register_loader"] = _REGISTERED[kind]
    env = _env_loaders()
    if kind in env:
        found[f"${LOADERS_ENV}"] = _resolve(env[kind])
    eps = [ep for ep in entry_points(group=LOADER_GROUP) if ep.name == kind]
    if len(eps) > 1:
        raise LoaderMissing(f"{len(eps)} entry points for {kind!r} in {LOADER_GROUP!r}: "
                            + ", ".join(ep.value for ep in eps))
    if eps:
        found[f"entry point {eps[0].value}"] = eps[0].load()
    distinct = {id(fn): fn for fn in found.values()}
    if not distinct:
        raise LoaderMissing(f"no loader for {kind}: specs; register one with register_loader, name it in "
                            f"${LOADERS_ENV} ({kind}=module:attr), or install a package that registers {kind!r} "
                            f"in the {LOADER_GROUP!r} entry-point group (kotodama provides koto)")
    if len(distinct) > 1:
        raise LoaderMissing(f"different loaders for {kind!r}: " + ", ".join(found))
    return next(iter(distinct.values()))


def _load_koto(spec: ModelSpec, device: str) -> LoadedModel:
    loader = loader_for(spec.kind)
    loaded = loader(spec, device)
    if not isinstance(loaded, LoadedModel):
        raise LoaderMissing(f"the {spec.kind!r} loader {loader!r} returned {type(loaded).__name__}, not a LoadedModel")
    return loaded


def _load_hf(spec: ModelSpec, device: str) -> LoadedModel:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(spec.path)
    dt = torch.float32 if device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(spec.path, torch_dtype=dt,
                                                 device_map=None if device == "cpu" else "auto")
    if device == "cpu":
        model = model.to("cpu")
    model.eval()

    def fwd(ids: torch.Tensor) -> torch.Tensor:
        return model(input_ids=ids).logits

    msl = getattr(model.config, "max_position_embeddings", None)
    return LoadedModel(spec, fwd, tok, next(model.parameters()).device, msl)
