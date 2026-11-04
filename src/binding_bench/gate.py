"""The refusal gate: a training corpus may share a battery's generating family, never its surface.

A corpus that contains the bench's prompts, sentence frames or word pools trains on the instrument, and the
instrument stops measuring. The gate checks a corpus against everything the bench renders and refuses it on any hit:

  line      a corpus line (16+ characters) equal to a line of a rendered bench prompt
  template  a corpus line (4+ words) whose skeleton equals a bench line's skeleton. A skeleton replaces every
            pool word (the bench's, and the record's own fillers) with ``_`` and every number with ``#``, so a
            sentence frame matches whatever fills it
  pool      a record's declared fillers (``fillers``: the names, objects and places it slotted in) that are bench
            pool words
  ngram     a word 13-gram (lowercased) that occurs in a rendered bench prompt

Bench text: every pinned rows battery's prompts, every bindprobes battery rendered at 0shot, 3shot and
3shot_collide, the riders grid (banked recipe) in every renderer at 0 and 3 shots, and any extra rows files given
(the situation rows). Bench pool: every candidate of every item, plus the name, object, place and holder pools the
generators draw from; one-letter words (chemical symbols, code variables) and the answer family's "me" / "you" are
left out.

Corpus: JSONL records ``{"text": ..., "fillers": [...]}`` (``fillers`` optional). The manifest records the corpus
sha256, the pin table's digest, each check's count and its first examples, and the verdict.

    python -m binding_bench.gate --corpus FILE.jsonl [--prompts-dir DIR] [--rows situation=FILE] --manifest OUT.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

from binding_bench.binding import batteries as B

WORD = re.compile(r"[A-Za-z][A-Za-z']*|\d+")
NGRAM = 13
MIN_LINE_CHARS = 16
MIN_TEMPLATE_WORDS = 4
N_EXAMPLES = 5
# candidates that are the construct's own words, not slot fillers (answer's me / you)
NOT_FILLERS = frozenset({"me", "you"})


def _words(line: str) -> list[str]:
    return WORD.findall(line)


def skeleton(line: str, pool: frozenset[str]) -> str:
    return " ".join("#" if w.isdigit() else "_" if w.lower() in pool else w.lower() for w in _words(line))


def _bench_prompts(root: Path, rows_files: Sequence[Path]) -> Iterator[str]:
    from binding_bench.bindprobes.items import build_prompt, load_items
    from binding_bench.riders import items as ri

    for b in B.BATTERIES.values():
        path = B.path_for(b, root)
        if b.kind == "bindprobes":
            for it in load_items(path):
                for cond in ("0shot", "3shot", "3shot_collide"):
                    yield build_prompt(it, cond)
        else:
            for ln in path.open():
                yield json.loads(ln)["prompt"]
    for f in rows_files:
        for ln in f.open():
            yield json.loads(ln)["prompt"]
    its, shots = ri.generate(list(ri.BANKED_VERIFIED_NAMES), 200, 200, ri.DEFAULT_SEED)
    for it in its:
        for rd in ri.ALL_RENDERERS:
            if rd != "chat":
                yield ri.build_prompt(it, rd, shots, "raw")["prompt"]


def _bench_pool(root: Path) -> frozenset[str]:
    from binding_bench import shortcuts as S
    from binding_bench.binding import capacity, fluid, stagegauges
    from binding_bench.bindprobes import pools
    from binding_bench.riders import items as ri
    from binding_bench.situation import items as si

    words: set[str] = set()
    for b in B.BATTERIES.values():
        for p in S.battery_probes(b, root):
            words.update(c.lower() for c in p.candidates)
    words.update(w.lower() for w in (*pools.FEMALE_CANDIDATES, *pools.MALE_CANDIDATES, *capacity.NAMES,
                                     *capacity.TASKS, *stagegauges.THIRD_NAMES, *stagegauges.EXTRA_OBJS,
                                     *fluid.VALUES, *fluid.HOLDERS, *ri.OBJECT_CANDIDATES))
    words.update(x.word.lower() for x in (*si.NAMES, *si.PLACES, *si.OBJECTS))
    return frozenset(w for w in words if WORD.fullmatch(w) and len(w) > 1 and w not in NOT_FILLERS)


@dataclass
class BenchIndex:
    lines: set[str]
    skeletons: set[str]
    ngrams: set[bytes]
    pool: frozenset[str]
    pins_digest: str

    @staticmethod
    def build(root: Path, rows_files: Sequence[Path] = ()) -> "BenchIndex":
        pool = _bench_pool(root)
        lines: set[str] = set(); skels: set[str] = set(); grams: set[bytes] = set()
        for prompt in _bench_prompts(root, rows_files):
            for ln in prompt.splitlines():
                s = ln.strip()
                if len(s) >= MIN_LINE_CHARS:
                    lines.add(s)
                if len(_words(s)) >= MIN_TEMPLATE_WORDS:
                    skels.add(skeleton(s, pool))
            grams.update(_grams(prompt))
        pins = json.dumps(sorted((b.name, b.sha256, b.render_sha256) for b in B.BATTERIES.values()))
        return BenchIndex(lines, skels, grams, pool, hashlib.sha256(pins.encode()).hexdigest())


def _grams(text: str) -> Iterator[bytes]:
    w = [x.lower() for x in _words(text)]
    for i in range(len(w) - NGRAM + 1):
        yield hashlib.blake2b(" ".join(w[i:i + NGRAM]).encode(), digest_size=12).digest()


@dataclass
class Check:
    count: int = 0
    examples: list[str] = field(default_factory=list)

    def hit(self, what: str) -> None:
        self.count += 1
        if len(self.examples) < N_EXAMPLES:
            self.examples.append(what)


@dataclass
class Report:
    records: int
    line: Check
    template: Check
    pool: Check
    ngram: Check

    @property
    def refused(self) -> bool:
        return any(c.count for c in (self.line, self.template, self.pool, self.ngram))


def check(records: Iterable[dict], idx: BenchIndex) -> Report:
    rep = Report(0, Check(), Check(), Check(), Check())
    for k, rec in enumerate(records):
        if not isinstance(rec.get("text"), str):
            raise ValueError(f"record {k}: no text")
        rep.records += 1
        fillers = [str(f) for f in rec.get("fillers", [])]
        for f in fillers:
            if f.lower() in idx.pool:
                rep.pool.hit(f"record {k}: {f}")
        local = idx.pool | frozenset(f.lower() for f in fillers)
        for ln in rec["text"].splitlines():
            s = ln.strip()
            if len(s) >= MIN_LINE_CHARS and s in idx.lines:
                rep.line.hit(f"record {k}: {s}")
            if len(_words(s)) >= MIN_TEMPLATE_WORDS and skeleton(s, local) in idx.skeletons:
                rep.template.hit(f"record {k}: {s}")
        if any(g in idx.ngrams for g in _grams(rec["text"])):
            rep.ngram.hit(f"record {k}")
    return rep


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--prompts-dir", default=None)
    ap.add_argument("--rows", action="append", default=[], help="NAME=FILE: an extra rows file (the situation rows)")
    ap.add_argument("--manifest", type=Path, required=True)
    a = ap.parse_args(argv)
    rows = [Path(x.split("=", 1)[1]) for x in a.rows]
    idx = BenchIndex.build(B.prompts_dir(a.prompts_dir), rows)
    with a.corpus.open() as f:
        rep = check((json.loads(ln) for ln in f if ln.strip()), idx)
    manifest = {"corpus": str(a.corpus), "corpus_sha256": _sha256(a.corpus), "pins_digest": idx.pins_digest,
                "extra_rows": {str(p): _sha256(p) for p in rows}, "bench_lines": len(idx.lines),
                "bench_skeletons": len(idx.skeletons), "bench_pool": len(idx.pool),
                "records": rep.records, "verdict": "REFUSED" if rep.refused else "PASS",
                **{k: asdict(getattr(rep, k)) for k in ("line", "template", "pool", "ngram")}}
    a.manifest.write_text(json.dumps(manifest, indent=1))
    for k in ("line", "template", "pool", "ngram"):
        c = getattr(rep, k)
        print(f"{k:9} {c.count:7}  {'; '.join(c.examples[:2])}")
    print(f"verdict: {manifest['verdict']} ({rep.records} records; manifest {a.manifest})")
    return 1 if rep.refused else 0


if __name__ == "__main__":
    sys.exit(main())
