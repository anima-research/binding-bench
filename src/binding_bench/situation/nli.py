"""The entailment screen: does each story entail its gold and not the other candidates?

Per item, probe and rendering, an NLI model reads the story as the premise and one hypothesis per candidate:

  situation  "The {o} is in the {place}."
  belief     "{a} thinks the {o} is in the {place}."
  knot       "{name} put the {o} in the {place}." (first) / "{name} moved the {o} to the {place}." (latest)

The story passes when the gold's hypothesis has the highest entailment probability, by at least ``margin``. A flag
is a pointer for a human or LLM reader, not a verdict. The screen gates the knot probes, and checks that no story
supports a candidate it never puts the object in. It does not gate situation or belief probes: a sentence-level NLI
model entails every place the object has ever been (DeBERTa-v3-large-mnli: "in the closet" .997 and "in the pot" .999
after a move from the closet to the pot), so it reads stories at the knot level. The simulator is the gold source
for those. The model is any Hugging Face sequence-classification NLI checkpoint whose labels include "entailment".

    python -m binding_bench.situation nli --items ITEMS_JSONL --model PATH_OR_NAME --out REPORT_JSON
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from binding_bench.situation.items import N_RENDERINGS, Item, Probe, render_story

KNOT_VERB = {"q_knot_first": "{n} put the {o} in the {p}.", "q_knot_latest": "{n} moved the {o} to the {p}."}


def hypotheses(item: Item, probe: Probe) -> list[str]:
    if probe.kind == "situation":
        return [f"The {item.obj} is in the {c}." for c in probe.candidates]
    if probe.kind == "belief":
        return [f"{probe.slots['a']} thinks the {item.obj} is in the {c}." for c in probe.candidates]
    t = KNOT_VERB[probe.lead]
    return [t.format(n=c, o=item.obj, p=probe.slots.get("p", "")) for c in probe.candidates]


def load_items(path: Path) -> list[Item]:
    out = []
    for ln in path.read_text().splitlines():
        if ln.strip():
            d = json.loads(ln)
            d["probes"] = [Probe(**p) for p in d["probes"]]
            out.append(Item(**d))
    return out


def screen(items: Sequence[Item], model: str, margin: float = 0.2, batch: int = 32,
           device: str = "cuda") -> dict[str, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    clf = AutoModelForSequenceClassification.from_pretrained(model).to(device).eval()
    labels = {v.lower(): int(k) for k, v in clf.config.id2label.items()}
    if "entailment" not in labels:
        raise ValueError(f"{model}: no 'entailment' label in {clf.config.id2label}")
    ent = labels["entailment"]
    jobs = []
    for it in items:
        for pr in it.probes:
            for r in range(N_RENDERINGS):
                prem = render_story(it, r)
                for c, h in zip(pr.candidates, hypotheses(it, pr)):
                    jobs.append((it.item_id, it.arm, pr.kind, r, c, prem, h))
    probs: list[float] = []
    with torch.inference_mode():
        for i in range(0, len(jobs), batch):
            chunk = jobs[i:i + batch]
            enc = tok([j[5] for j in chunk], [j[6] for j in chunk], return_tensors="pt", padding=True,
                      truncation=True, max_length=512).to(device)
            probs += clf(**enc).logits.softmax(-1)[:, ent].tolist()
    by: dict[tuple, dict[str, float]] = defaultdict(dict)
    for j, p in zip(jobs, probs):
        by[j[:4]][j[4]] = p
    gold = {(it.item_id, pr.kind): pr.gold for it in items for pr in it.probes}
    flags, tally = [], Counter()
    for (iid, arm, kind, r), ps in by.items():
        g = gold[(iid, kind)]
        rest = max(v for c, v in ps.items() if c != g)
        ok = ps[g] - rest >= margin
        tally[(arm, kind, ok)] += 1
        if not ok:
            flags.append({"item_id": iid, "arm": arm, "probe": kind, "rendering": r, "gold": g, "p": ps})
    summary = {f"{a}|{k}": {"pass": tally[(a, k, True)], "flag": tally[(a, k, False)]}
               for a, k in sorted({(a, k) for a, k, _ in tally})}
    return {"model": model, "margin": margin, "summary": summary, "flags": flags}


def write(report: dict[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
