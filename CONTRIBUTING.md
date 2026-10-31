# Contributing

Two kinds of contribution arrive here, and they are handled differently.

**Code** is a pull request against this repository. The gates described below decide whether it merges.

**Claims** are not code. A contribution that asserts a finding arrives as an issue with the runs and receipts
behind it. Examples are that a model binds better, that an eval measures its construct, or that a number holds. It is
replicated where warranted, and a public page states it only once it has been. Code that *enables* a claim (a battery,
a reader, an audit tool) is welcome in a pull request. The claim itself travels the other road.

## How this repository is organised

Each fact has one home, and every other place reads it from there:

- what each battery's prompt file and rendered prompts hash to: `src/binding_bench/binding/batteries.py`;
- what each eval measures, which readings are clean, and the audit flags: `src/binding_bench/registry.py`;
- the tokenizer that sizes and checks items: `src/binding_bench/tokenizer.py`;
- the ChatML frame of the riders chat register: `src/binding_bench/chatml.py`;
- the closed-set scorer: `src/binding_bench/score.py`.

A constant restated as a literal somewhere else is a second source of truth, and the two drift. Machine paths,
hostnames and keys never enter a tracked file: the prompts directory comes from `--prompts-dir` or
`$KOTODAMA_BINDING_PROMPTS`. A checkpoint format other than `hf:` arrives as a loader plugin in the
`binding_bench.loaders` entry-point group (`src/binding_bench/binding/models.py`), never as an import of another codebase.

## The benchmark's own rules

These hold for every battery and every reading, and a pull request that breaks one is refused.

- **A battery is its bytes.** Every battery is pinned by the sha256 of its prompt file and, for bindprobes, of every
  prompt it renders. A run refuses a file that differs from its pin. A new version of a battery is a new name and a
  new pin beside the old one, which stays scoreable. A pin is never changed in place.
- **Every battery has a recipe.** Its registry row names how its bytes come back. Most batteries come from a
  generator in this repository, and their tests regenerate each pin. A bindprobes battery drawn by an older
  version of a family has a historical recipe instead, which names the public kotodama commit whose generator
  drew it (`HISTORICAL_RECIPES`). `tools/replay_recipe.py` rebuilds it and checks the pin. A new battery
  arrives with its generator.
- **An eval changes only on validity grounds:** a rendering that moves the score, an off-construct lever that moves
  it, or an item-level shortcut. A low score is never a reason. A new rendering has to shrink the spread across
  renderings, not raise the score.
- **Every test is registered before any row is scored.** An amendment is a dated addendum below the registration,
  never an edit above it.
- **A flag is model-relative.** A registry flag names the panel it was measured on and the test that found it. An
  eval-level verdict needs the flag on at least three models of different families.
- **The few-shot reading is `fair`, and 0-shot is the floor.** `fair` uses four worked examples balanced on every
  component of the item's cell. Exemplars that match the item's cell can teach the answer's role, and the
  flipped-exemplar test shows that they do.

## The documentation rule

Comments, docstrings, and the message that a `raise` or a logging call prints obey two rules. Both are mechanically
checked. Neither applies to wire format (a fixture, a dict key, a filename, a banked field name), because changing it
would break reading what is already banked.

### 1. State what is true now

A comment describes the code as it stands. It does not narrate how the code got there: no `previously`, `used to`,
`changed in`, `we now`, `no longer`. There is no bare `TODO`, `FIXME` or `HACK`. A known gap is an explicit refusal, a
test that pins the current behaviour, or an issue. **No dates**, on an edit or on evidence: git records when a line
changed, and a dated measurement belongs in the record that holds it. The code needs the standing fact that it depends
on, in the present tense.

### 2. Every referent must be reachable from this repository

A reader who holds only this repository must be able to look up whatever a comment names. That can be a module here,
a file in this tree, a symbol this package defines, or a public URL. A private path, a planning document, a
conversation, a machine, a host or a sibling repository is not a referent. Neither is a provenance citation: the `§`
of a document that is not here, a pre-registration, or a commit hash. When the substance is short, state it inline.
When it is long and public, link it. When it is long and private, restate the part that the code depends on.

A claim in a docstring must match the code under it.

## Pages a stranger reads

The README, the pages under `docs/`, this file, and every module docstring that those pages link to are read by a
technically literate stranger. The journal, the notes and `CLAUDE.md` are working records and are not public pages. They follow a stricter voice. Pages are written in the third person and the timeless
present, and they never narrate their own edits. Uncertainty is stated as a fact about evidence ("measured on eight
models"). A claim travels with its scope. A number is verbatim from the run that produced it, with n and the
comparison inline. Prior work that grounds a claim is cited, and `docs/PRIOR-ART.md` keeps the map.

A page is also held to stay at least as legible as it is. `tools/check_legibility.py` counts the features that make
prose hard to follow on a first read: long sentences, long distances between linked words, object relatives without
"that", stacked nouns, things as the subjects of mind verbs, and nouns used as verbs. The features come from Clark,
Martin & Mahowald, [*Why is AI-generated language so hard to understand?*](https://arxiv.org/abs/2610.29162), and each
rate is compared with English Wikipedia. `tools/legibility_baseline.json` records each page's numbers. A page fails when
a feature rises more than ten percent above its baseline, plus one occurrence. The check parses with the spaCy
versions that `tools/legibility_requirements.txt` pins, so it runs in an environment of its own:

```bash
uv venv .venv-legibility && uv pip install --python .venv-legibility -r tools/legibility_requirements.txt
.venv-legibility/bin/python -m tools.check_legibility                            # every public page
.venv-legibility/bin/python -m tools.check_legibility --page README.md --flags   # one page, every flagged sentence
```

A page that improves passes, and `--update-baseline` writes the new level in. A regression enters the baseline only
with a reason: `--update-baseline --reason "..."`.

## The gates

A change merges when these pass. Each one is a command that runs from the repository root.

| gate | what it checks | command |
|---|---|---|
| tests | the CPU suite (pytest-xdist from the `dev` extra) | `python -m pytest -n auto tests/` |
| G2 test retention | a change that shrinks the suite shrinks the code by at least as many lines | `python -m tools.check_test_retention --base main` |
| G3 documentation | the two rules above, and the public voice | `python -m tools.check_timelessness --root src/binding_bench tools tests`, `python -m tools.check_referents --root src/binding_bench tools tests` and `python -m tools.check_legibility` |
| G4 import closure | every module reachable from a test or a tool; no orphans | `python -m tools.check_import_closure --package src/binding_bench --roots tests tools` |
| parity | the 10 regenerable batteries regenerate to their pins, and bindprobes renders match the golden file | in the test suite |

G2 counts physical lines of Python under `tests/` (the suite) and everywhere else (the code), at the branch and at the
point it left `main`. Both pattern files, `tools/timelessness_allowlist.txt` and `tools/referents_allowlist.txt`, carry
a reason above every entry, and an entry goes when its reason expires. A gate is waived only in writing, with the
reason beside the receipt. The failure is recorded, and the gate is never rewritten.

## Publishing

Before a push to `main`, the gates pass and a person reads the diff for what no checker can enumerate: host and
machine names, private paths, keys and tokens, account names, and pointers into private research.
