"""Rebuild a bindprobes battery from its historical recipe and check it against the pin.

    python -m tools.replay_recipe --kotodama PATH_TO_A_KOTODAMA_CLONE --battery bp-profile [--out DIR] [--keep]

The recipe (``binding_bench.registry.HISTORICAL_RECIPES``) names a public kotodama commit. The tool checks that commit
out into a temporary git worktree of the given clone, runs its bindprobes generator in a separate interpreter (so the
old code never mixes with this package's), keeps the recipe's cells, and compares the file's sha256 with the
battery's pin. Exit 0 on a match, 1 on a mismatch, 2 on a refusal (no such battery or commit).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

from binding_bench.binding import batteries as bt
from binding_bench.registry import HISTORICAL_RECIPES, EmitRecipe

_EMIT = """
import sys
from pathlib import Path
from kotodama.eval.bindprobes import items
seed, out, fams = int(sys.argv[1]), Path(sys.argv[2]), sys.argv[3].split(",")
per_cell = dict(zip(fams, map(int, sys.argv[4].split(","))))
print(items.emit(out, seed, fams, per_cell))
"""


def keep(lines: Iterable[str], cells: Sequence[Sequence[object]]) -> list[str]:
    """The item lines whose ``cell`` is one of ``cells`` (all lines when ``cells`` is empty), in order."""
    wanted = {tuple(c) for c in cells}
    return [ln for ln in lines if not wanted or tuple(json.loads(ln)["cell"]) in wanted]


def replay(recipe: EmitRecipe, kotodama: Path, out: Path, python: str = sys.executable) -> Path:
    """Write the recipe's items file under ``out`` and return its path."""
    with tempfile.TemporaryDirectory(prefix="replay-") as tmp:
        wt = Path(tmp) / "kotodama"
        subprocess.run(["git", "-C", str(kotodama), "worktree", "add", "--detach", str(wt), recipe.kotodama_commit],
                       check=True, capture_output=True, text=True)
        try:
            fams = list(recipe.per_cell)
            res = subprocess.run([python, "-I", "-c", f"import sys; sys.path.insert(0, {str(wt / 'src')!r})\n" + _EMIT,
                                  str(recipe.seed), str(Path(tmp) / "emit"), ",".join(fams),
                                  ",".join(str(recipe.per_cell[f]) for f in fams)],
                                 check=True, capture_output=True, text=True)
            emitted = Path(res.stdout.strip().splitlines()[-1])
            lines = keep(emitted.read_text().splitlines(keepends=True), recipe.keep_cells)
        finally:
            subprocess.run(["git", "-C", str(kotodama), "worktree", "remove", "--force", str(wt)],
                           check=False, capture_output=True)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "items.jsonl"
    dest.write_text("".join(lines))
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kotodama", required=True, help="a clone of the kotodama repository that holds the commit")
    ap.add_argument("--battery", required=True, choices=sorted(HISTORICAL_RECIPES))
    ap.add_argument("--out", default=None, help="where to write <battery>/items.jsonl (default: a temporary directory)")
    a = ap.parse_args(argv)
    recipe = HISTORICAL_RECIPES[a.battery]
    try:
        with tempfile.TemporaryDirectory() as tmp:
            dest = replay(recipe, Path(a.kotodama), Path(a.out or tmp) / a.battery)
            got = hashlib.sha256(dest.read_bytes()).hexdigest()
            n = sum(1 for _ in dest.open())
    except subprocess.CalledProcessError as exc:
        print(f"{a.battery}: replay refused: {' '.join(exc.cmd[:6])}…\n{exc.stderr.strip()[-600:]}", file=sys.stderr)
        return 2
    pin = bt.BATTERIES[a.battery].sha256
    print(f"{a.battery}: {n} items at kotodama {recipe.kotodama_commit}; sha256 {got} "
          f"{'= pin' if got == pin else '≠ pin ' + pin[:16] + '…'}")
    return 0 if got == pin else 1


if __name__ == "__main__":
    sys.exit(main())
