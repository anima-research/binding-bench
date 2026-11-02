"""Situation-model battery CLI.

    python -m binding_bench.situation gen --out DIR [--per-arm 100] [--seed 1103]
    python -m binding_bench.situation check-tokens [--tokenizers smollm2,pythia]
    python -m binding_bench.situation nli --items DIR/items.jsonl --model NLI_MODEL --out REPORT_JSON
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from binding_bench.situation import items


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m binding_bench.situation", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen", help="write items.jsonl + rows.jsonl (every item gated)")
    g.add_argument("--out", required=True)
    g.add_argument("--per-arm", type=int, default=100)
    g.add_argument("--seed", type=int, default=items.DEFAULT_SEED)
    t = sub.add_parser("check-tokens", help="scored words that are not one token in a grid tokenizer")
    t.add_argument("--tokenizers", default="")
    n = sub.add_parser("nli", help="the entailment screen over an items file")
    n.add_argument("--items", required=True)
    n.add_argument("--model", required=True)
    n.add_argument("--out", required=True)
    n.add_argument("--margin", type=float, default=0.2)
    a = ap.parse_args(argv)
    if a.cmd == "nli":
        from binding_bench.situation import nli
        rep = nli.screen(nli.load_items(Path(a.items)), a.model, a.margin)
        nli.write(rep, Path(a.out))
        print(json.dumps(rep["summary"], indent=1))
        return 0
    if a.cmd == "gen":
        print(items.emit(Path(a.out), a.per_arm, a.seed))
        return 0
    bad = items.check_tokens([k for k in a.tokenizers.split(",") if k])
    print(json.dumps(bad, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
