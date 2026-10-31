"""The rider grid: the four-register entity-state binding battery.

Judge-free and logprob-only. ``items`` generates N-boxes × M-updates binding
items (plus difficulty-0 gate items) and renders them in four registers (code,
narrative, screenplay, chat); ``grid`` scores them against a live server's
``/logprobs``; ``aggregate`` builds the gate + margin/rank/accuracy tables;
``smoothing`` reads the attractor-pull index off the same grids. ``lite`` is the in-loop slice (balanced item
subset, re-lettered shots, summary); ``errors`` types what a wrong row picked.

CLI: ``python -m binding_bench.riders {gen,verify-names,selftest,run,aggregate,smoothing}``.
"""

# Submodule names (items, grid, aggregate, smoothing) are NOT shadowed by
# re-exported functions: ``from binding_bench.riders import aggregate`` is the module.
from binding_bench.riders import aggregate, errors, grid, items, lite, smoothing
from binding_bench.riders.aggregate import cc_acc, write_tables
from binding_bench.riders.grid import default_endpoint, run_grid, score_row
from binding_bench.riders.items import (RENDERERS, Item, build_prompt, emit, generate,
                                        load_items)
from binding_bench.riders.smoothing import smoothing_index

__all__ = [
    "RENDERERS",
    "Item",
    "aggregate",
    "errors",
    "grid",
    "items",
    "lite",
    "smoothing",
    "build_prompt",
    "cc_acc",
    "default_endpoint",
    "emit",
    "generate",
    "load_items",
    "run_grid",
    "score_row",
    "smoothing_index",
    "write_tables",
]
