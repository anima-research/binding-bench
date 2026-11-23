"""Build padbind's filler pool from a pinned public FineWeb-Edu slice.

    python -m tools.build_padbind_pool <out>.jsonl

Reads row groups of one parquet file of ``HuggingFaceFW/fineweb-edu`` at a pinned revision, in order (HTTP range
reads; the file itself is about 2 GB), and keeps documents that pass the filter in file order until the pool holds
:data:`binding_bench.binding.padbind.POOL_DOCS` of them, and writes ``{"id", "text", "tokens"}`` lines (``tokens``: the document's length under
:data:`binding_bench.binding.padbind.TOKENIZER`). The pool's sha256 is
pinned in :mod:`binding_bench.binding.padbind`; the battery generator refuses any other file.

Filter: 1,000–8,000 characters; no ``* * *`` (the battery's episode separator), no ``<|endoftext|>``, fewer than three
lines that read as a chat turn (``Name: text``), and printable text only. Needs ``pyarrow`` and ``huggingface_hub``
(``pip install binding-bench[padbind-pool]``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from binding_bench.binding import padbind as pb

CHAT_LINE = re.compile(r"^[A-Z][a-z]+: ", re.M)


def keep(text: str) -> bool:
    return (pb.POOL_MIN_CHARS <= len(text) <= pb.POOL_MAX_CHARS and "* * *" not in text
            and pb.EOS not in text and len(CHAT_LINE.findall(text)) < 3
            and all(c.isprintable() or c in "\n\t" for c in text))


def main() -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(pb.TOKENIZER)
    fs = HfFileSystem()
    path = f"datasets/{pb.POOL_REPO}@{pb.POOL_REVISION}/{pb.POOL_FILE_IN_REPO}"
    docs: list[dict[str, str]] = []
    with fs.open(path, "rb") as f:
        pf = pq.ParquetFile(f)
        for g in range(min(pb.POOL_MAX_ROW_GROUPS, pf.num_row_groups)):
            table = pf.read_row_group(g, columns=["id", "text"])
            for i, t in zip(table.column("id").to_pylist(), table.column("text").to_pylist()):
                if keep(t) and len(docs) < pb.POOL_DOCS:
                    docs.append({"id": i, "text": t, "tokens": len(tok.encode(t))})
            if len(docs) == pb.POOL_DOCS:
                break
    if len(docs) < pb.POOL_DOCS:
        raise SystemExit(f"only {len(docs)} documents passed the filter; the pool needs {pb.POOL_DOCS}")
    a.out.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs), encoding="utf-8")
    print(f"{a.out}: {len(docs)} documents, {sum(len(d['text']) for d in docs):,} characters, "
          f"sha256 {hashlib.sha256(a.out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
