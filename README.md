# binding-bench

binding-bench is a benchmark of binding and in-context learning for language models. Its batteries test three
things:

- whether a model tracks who holds what;
- whether it resolves who *I* and *you* refer to in a dialogue;
- whether it learns a new rule from the examples in its prompt (fluid in-context learning), against recall of known
  facts in the same frame (crystallized).

Every battery is a frozen, sha256-pinned set of prompts. Every model is scored the same way: by the log-probability
that it gives each candidate answer.

## Where the pieces are

| piece | module |
|---|---|
| what each eval measures, its recipe, clean readings and audit flags | `binding_bench.registry` |
| pinned batteries, prompt-file checks, the suite CLI and readers | `binding_bench.binding` (`batteries.py`, `__main__.py`, `readers.py`) |
| bindprobes families and their item generator | `binding_bench.bindprobes` |
| riders: entity-state tracking in four registers, and a small in-loop slice | `binding_bench.riders` (`lite.py` for the slice) |
| the closed-set scorer | `binding_bench.score`, with `binding_bench.binding.score` for prompt rows |
| the tokenizer that sizes items, and the ChatML frame of the chat register | `binding_bench.tokenizer`, `binding_bench.chatml` |
| model loading: `hf:` built in, other checkpoint formats as plugins | `binding_bench.binding.models` |

The package imports no model codebase. Other checkpoint formats arrive as loader plugins, which register in the
`binding_bench.loaders` group of entry points. `tests/test_bench_boundary.py` checks this boundary. It also checks
that every pinned battery has a registry row.

## Reading a score

Each registry row names its **clean readings**: the conditions that count when two models are compared. For the
bindprobes families these are `0shot` (bare items with no worked examples, which is the floor) and `fair` (four
worked examples balanced on every component of the item's cell). Other conditions stay available as diagnostics.
A **flag** marks a reading that is not a clean measure of its construct on the panel that it names. An example is a
few-shot condition whose exemplars can be copied. A flag is a finding about the reading, not about the model.

An eval changes only on validity grounds: a rendering that moves the score, an off-construct lever that moves it,
or a shortcut in the items. A new rendering has to reduce the spread across renderings, not raise the score. A new
version of a battery is a new name and a new pin beside the old one. The prior work that the benchmark builds on is
mapped in `docs/PRIOR-ART.md`.

## Commands

```
uv pip install -e '.[dev]'
python -m binding_bench.binding list                                       # batteries and named groups
python -m binding_bench.binding run --model hf:Qwen/Qwen3-1.7B-Base --name qwen3-1.7b --batteries scorecard --out scores/
python -m binding_bench.binding report --scores scores/                   # the scorecard of the prompt-row batteries
python -m binding_bench.binding gen-capacity --out capacity-v1.jsonl      # regenerates the pinned file byte for byte
python -m binding_bench.binding gen-fluid --out DIR
python -m binding_bench.binding gen-formupdate --out DIR
python -m binding_bench.binding gen-stage --out DIR [--prompts-dir DIR]  # joinlite, swapdyn, stage5-decomp
python -m tools.replay_recipe --kotodama CLONE --battery bp-profile      # an older bindprobes battery
```

Prompt files live under one directory (`--prompts-dir`, or `$KOTODAMA_BINDING_PROMPTS`). A run refuses a file whose
bytes, or whose rendered prompts, differ from the pin.
Every pinned file can be rebuilt byte for byte. The generators above write most of them. Earlier versions of
the bindprobes families drew a few batteries, and the registry records the public kotodama commit of each one.
The replay tool rebuilds those batteries with that commit's generator.
