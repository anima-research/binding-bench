"""padbind: binding over distance. register-v3's seen scenes, with natural text between the scene and its query.

A row is a register-v3 seen row (static 4-shot lookup and static "I", orig and swap; set binding, fair) with filler
inserted inside the queried scene: after its last context line, before the query line. So the worked examples and the
scene come first, then the filler, and the query is the prompt's last line. The total length is padded to roughly
:data:`PADS` tokens under :data:`TOKENIZER`: the filler holds ``pad −`` :data:`PROMPT_ALLOWANCE` tokens, counted
from the pool's per-document token counts, so a row lands just under its pad.

Two filler conditions, the same documents in the same order:

* ``span``: the documents joined by blank lines, one continuous span inside the scene's own document (the regime a
  model trained with document masking over long documents has seen);
* ``docs``: the documents separated by :data:`EOS`, before and after, so the query sits in a different document
  from the scene (descriptive: whether an unmasked read reaches across document boundaries).

Filler is drawn from a pinned pool of public FineWeb-Edu documents (``tools/build_padbind_pool.py``). Each scene (and
each set item) gets its own stream from a seeded start, skipping documents that mention any of its candidates or
names, so filler never adds a distractor. A scene's rows share one filler, so orig and swap differ only in the query,
and a shorter pad's filler is the start of a longer one's.

Rows carry ``register`` = ``seen`` (no filler) or ``pad<P>-<condition>``, so :func:`readers.register` reads every pad
against the unpadded scenes, paired item by item.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from binding_bench.binding import register as rg

Row = dict[str, Any]

EOS = "<|endoftext|>"                       # GPT-2-family end of text, as SmolLM2's tokenizer reads it in text
POOL_REPO = "HuggingFaceFW/fineweb-edu"
POOL_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
POOL_FILE_IN_REPO = "sample/10BT/000_00000.parquet"
POOL_MAX_ROW_GROUPS = 10                    # row groups read in order (1,000 documents each) until the pool is full
POOL_DOCS = 3000
POOL_MIN_CHARS, POOL_MAX_CHARS = 1000, 8000
POOL_FILE = "padbind-pool-v1.jsonl"         # relative to the prompts dir
POOL_SHA256 = "03dfd7835f75b219ffd5ff6fd08005a5fd0565b39b2ce0a5770511da4bdb21e7"
TOKENIZER = "HuggingFaceTB/SmolLM2-135M"    # counts each pool document's tokens (the pool's ``tokens`` field)
PROMPT_ALLOWANCE = 560                      # tokens left for the scene and query (297–386) and for the joins' slack (±150): every row fits its pad
PADS = (1024, 4096, 8192, 16384)
CONDITIONS = ("span", "docs")


@dataclass(frozen=True)
class Version:
    name: str
    source: str                             # the register battery whose seen rows are padded
    scenes: int                             # static scenes (each gives lookup + static "I" orig and swap)
    set_per_cell: int


VERSIONS = {"padbind-v1": Version("padbind-v1", "register-v3", 200, 20)}


def load_pool(path: Path) -> list[Row]:
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != POOL_SHA256:
        raise ValueError(f"{path}: sha256 {got} is not the pinned padbind pool {POOL_SHA256}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def scene_words(rows: Sequence[Row]) -> set[str]:
    """The candidates and the names of a scene's (or set item's) final block: what its filler must never mention."""
    words: set[str] = set()
    for r in rows:
        words |= {c.strip() for c in r["candidates"]}
        last = r["prompt"].rsplit(rg.SEEN_SEPARATOR, 1)[-1]
        words |= set(re.findall(r"\b[A-Z][a-z]+\b", last)) - {"A", "The", "Right", "So", "Now"}
    return words


def _banned(rows: Sequence[Row]) -> re.Pattern[str]:
    return re.compile(r"\b(" + "|".join(sorted(map(re.escape, scene_words(rows)))) + r")\b", re.I)


def _stream(pool: Sequence[Row], key: str, banned: re.Pattern[str], tokens: int) -> list[Row]:
    """Whole documents from a seeded start, skipping banned ones, until their tokens (one more per document for its
    separator) reach ``tokens``; the last one is cut in proportion, at a word boundary, so the total is just under."""
    rng = random.Random(key)
    i = rng.randrange(len(pool))
    out: list[Row] = []
    total, seen = 0, 0
    while total < tokens:
        if seen >= len(pool):
            raise ValueError(f"{key}: the pool runs out before {tokens} tokens of clean filler")
        d = pool[i % len(pool)]
        i, seen = i + 1, seen + 1
        if banned.search(d["text"]):
            continue
        out.append(d)
        total += d["tokens"] + 1
    if out:
        last = out[-1]
        keep = max(1, last["tokens"] - (total - tokens))
        cut = last["text"][: len(last["text"]) * keep // last["tokens"]]
        cut = cut[: cut.rfind(" ")] if " " in cut else cut
        out[-1] = {"id": last["id"], "text": cut, "tokens": keep}
    return out


def _insert(prompt: str, docs: Sequence[Row], condition: str) -> str:
    head, sep, last = prompt.rpartition(rg.SEEN_SEPARATOR)
    *context, query = last.split("\n")
    if condition == "span":
        body = "\n".join(context) + "\n" + "\n\n".join(d["text"] for d in docs) + "\n" + query
    elif condition == "docs":
        body = "\n".join(context) + EOS + EOS.join(d["text"] for d in docs) + EOS + query
    else:
        raise ValueError(f"unknown filler condition {condition!r}")
    return head + sep + body


def sources(v: Version) -> tuple[list[list[Row]], list[list[Row]]]:
    """The seen rows to pad, grouped by scene (static) and by set item, in a seeded sample."""
    src = rg.VERSIONS[v.source]
    rng = random.Random(f"{v.name}:sample")
    st = [r for r in rg.static_rows(src, rg.SEED) if r["register"] == "seen" and r["condition"] == src.static_cond
          and (r["qkind"], r["arm"]) in {("third", "orig"), ("self", "orig"), ("self", "swap")}]
    by_scene: dict[str, list[Row]] = {}
    for r in st:
        by_scene.setdefault(r["base_id"], []).append(r)
    scenes = [by_scene[b] for b in sorted(rng.sample(sorted(by_scene), v.scenes))]
    se = [r for r in rg.set_rows(src, rg.SEED) if r["register"] == "seen" and r["condition"] == "fair"]
    by_cell: dict[str, list[Row]] = {}
    for r in se:
        by_cell.setdefault(json.dumps(r["cell"]), []).append(r)
    items = [[r] for c in sorted(by_cell) for r in sorted(rng.sample(by_cell[c], v.set_per_cell),
                                                          key=lambda x: x["base_id"])]
    return scenes, items


def padbind_rows(name: str, pool_path: Path) -> list[Row]:
    v = VERSIONS[name]
    pool = load_pool(pool_path)
    scenes, items = sources(v)
    rows: list[Row] = []
    for unit in scenes + items:
        for r in unit:
            rows.append({**r, "id": f"{r['id']}|pad0", "pad": 0, "filler": None})
        banned = _banned(unit)
        key = f"{v.name}:{unit[0]['base_id']}"
        longest = _stream(pool, key, banned, max(PADS) - PROMPT_ALLOWANCE)
        for pad in PADS:
            docs = _stream(pool, key, banned, pad - PROMPT_ALLOWANCE)
            assert [d["id"] for d in docs] == [d["id"] for d in longest[: len(docs)]], "fillers must nest"
            for cond in CONDITIONS:
                for r in unit:
                    rows.append({**r, "id": f"{r['id']}|pad{pad}|{cond}", "register": f"pad{pad}-{cond}",
                                 "prompt": _insert(r["prompt"], docs, cond), "pad": pad, "filler": cond,
                                 "filler_tokens": sum(d["tokens"] + 1 for d in docs),
                                 "filler_docs": [d["id"] for d in docs]})
    return rows
