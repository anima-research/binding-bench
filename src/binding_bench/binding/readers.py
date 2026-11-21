"""Per-battery readers (each battery computes a different statistic) and the one scorecard built from them.

Readers take scored rows (``score.prompt_row`` shape) for ONE model and return plain dicts:
  joinlite      static stages: lookup (3rd-person) accuracy, I / you accuracy (P2 and P3), and the P2 speaker-swap
                shift with a paired-bootstrap CI (the stage gauge: > +.02 at the CI's low end binds, < −.02 at its high
                end inverts)
  swapdyn       dynamic stage: the speaker-swap shift on the transfer prompts, per condition (0shot / 3shot) and
                query kind (self / addressee)
  decomp        stage-5 decomposition: accuracy per variant × condition (layer 1 = 0-shot ``third``)
  capacity      K50 / AUC / D=256 (:func:`binding_bench.binding.capacity.summarize`)
  fluid-*, cryst  (but fluid-supplied) per-family ladder midpoint / AUC, F3 stale-value rate, C0 accuracy by shots
                (:func:`binding_bench.binding.fluid.summarize`)
  fluid-supplied  accuracy against chance per family × rung of the supplied-rule rows (their inferred twins are read
                by the inferred batteries)
  formupdate    form acc(A), interference and update costs (raw and normalized) with paired-bootstrap intervals,
                C's stale rate (:func:`binding_bench.binding.formupdate.summarize`)
The arithmetic is that of the readers the recorded gauge numbers came from, so a suite number equals the recorded
one for the same rows.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Mapping, Sequence

Row = Mapping[str, Any]
BIND = 0.02


def _pnorm(r: Row, obj: str) -> float:
    lps = r["cand_lps"]
    mx = max(lps.values())
    return math.exp(lps[obj] - mx) / sum(math.exp(v - mx) for v in lps.values())


def _ok(r: Row, gold_key: str = "gold") -> bool:
    return max(r["cand_lps"], key=r["cand_lps"].get) == r[gold_key]


def boot_ci(xs: Sequence[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    k = len(xs)
    means = sorted(sum(rng.choice(xs) for _ in range(k)) / k for _ in range(n))
    return means[int(.025 * n)], means[int(.975 * n)]


def reading(lo: float, hi: float) -> str:
    return "binds" if lo > BIND else ("INVERTS" if hi < -BIND else "ignores")


def _shift(d: Sequence[float]) -> dict[str, Any]:
    mu = sum(d) / len(d)
    lo, hi = boot_ci(d)
    return {"n": len(d), "shift": mu, "ci": [lo, hi], "reading": reading(lo, hi)}


def joinlite(rows: Sequence[Row]) -> dict[str, Any]:
    by = {(r["scene"], r["qkind"], r["arm"]): r for r in rows}
    out: dict[str, Any] = {}
    for P in (2, 3):
        sc = sorted({r["scene"] for r in rows if r["P"] == P})
        if not sc:
            continue

        def acc(q: str, arm: str = "orig") -> float:
            return sum(_ok(by[(s, q, arm)]) for s in sc if (s, q, arm) in by) / len(sc)

        third_ok = [s for s in sc if _ok(by[(s, "third", "orig")])]
        cell: dict[str, Any] = {"scenes": len(sc), "lookup": acc("third"), "I": acc("self"), "you": acc("addressee"),
                                "you_given_lookup": sum(_ok(by[(s, "addressee", "orig")]) for s in third_ok)
                                / max(1, len(third_ok)), "chance": 1 / (4 if P == 3 else 3)}
        if P == 2:
            for q in ("self", "addressee"):
                d = [_pnorm(by[(s, q, "orig")], by[(s, q, "orig")]["gold"])
                     - _pnorm(by[(s, q, "swap")], by[(s, q, "orig")]["gold"]) for s in sc]
                cell[f"swap_{q}"] = _shift(d)
            inv = sum(max(by[(s, "addressee", "orig")]["cand_lps"], key=by[(s, "addressee", "orig")]["cand_lps"].get)
                      == by[(s, "self", "orig")]["gold"] for s in sc)
            cell["you_to_speaker_obj"] = inv / len(sc)
        out[f"P{P}"] = cell
    return out


def swapdyn(rows: Sequence[Row]) -> dict[str, Any]:
    pair: dict[tuple[str, str, str], dict[str, Row]] = defaultdict(dict)
    for r in rows:
        if r.get("multi_token"):
            continue
        pair[(r["condition"], r["qkind"], r["base_id"])][r["arm"]] = r
    out: dict[str, Any] = {}
    for cond in ("0shot", "3shot"):
        for q in ("addressee", "self"):
            ps = [v for (c, qq, _), v in pair.items() if c == cond and qq == q and "orig" in v and "swap" in v]
            if not ps:
                continue
            po = [_pnorm(v["orig"], v["orig"]["gold_orig"]) for v in ps]
            pw = [_pnorm(v["swap"], v["orig"]["gold_orig"]) for v in ps]
            cell = _shift([x - y for x, y in zip(po, pw)])
            cell.update(p_orig=sum(po) / len(po), p_swap=sum(pw) / len(pw),
                        argmax_orig=sum(_ok(v["orig"], "gold_orig") for v in ps) / len(ps),
                        argmax_swap=sum(_ok(v["swap"], "gold_orig") for v in ps) / len(ps))
            out[f"{cond}/{q}"] = cell
    return out


def decomp(rows: Sequence[Row]) -> dict[str, Any]:
    by: dict[tuple[str, str], list[bool]] = defaultdict(list)
    chance: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        key = (r["variant"], r["condition"])
        by[key].append(_ok(r))
        chance[key].append(1 / len(r["candidates"]))
    return {f"{v}/{c}": {"n": len(x), "acc": sum(x) / len(x), "chance": sum(chance[(v, c)]) / len(x)}
            for (v, c), x in sorted(by.items())}


def capacity(rows: Sequence[Row]) -> dict[str, Any]:
    from binding_bench.binding.capacity import summarize

    s = summarize(rows)
    return {"ceiling": s.ceiling, "K50": s.fit.k50, "K50_censored": s.fit.censored, "K50_label": s.fit.label(),
            "beta": s.fit.beta, "auc": s.auc, "acc@8": s.acc8, "ceiling_d256": s.ceiling_d256,
            "K50_d256_fixed": s.fit_d256_fixed.label() if s.fit_d256_fixed else None,
            "K50_d256_own": s.fit_d256_own.label() if s.fit_d256_own else None,
            "acc@8_d256": s.acc8_d256, "usable": s.usable, "multi_token_rows": s.multi_token_rows,
            "curve": s.curve, "K50_listed": s.fit_listed.label() if s.fit_listed else None,
            "K50_by_rendering": {k: f.label() for k, f in s.by_rendering.items()}, "K50_spread": s.spread}


def fluid(rows: Sequence[Row]) -> dict[str, Any]:
    from binding_bench.binding.fluid import summarize

    return summarize(rows).as_dict()


def formupdate(rows: Sequence[Row]) -> dict[str, Any]:
    from binding_bench.binding.formupdate import summarize

    return summarize(rows).as_dict()


def frame(rows: Sequence[Row]) -> dict[str, Any]:
    """frame-ladder-v1: accuracy against chance per rung × query kind (0-shot, one cue added per rung)."""
    by: dict[tuple[str, str], list[bool]] = defaultdict(list)
    chance: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        key = (r["rung"], r["qkind"])
        by[key].append(_ok(r))
        chance[key].append(1 / len(r["candidates"]))
    return {f"{g}/{q}": {"n": len(x), "acc": sum(x) / len(x), "chance": sum(chance[(g, q)]) / len(x)}
            for (g, q), x in sorted(by.items())}


def fluid_supplied(rows: Sequence[Row]) -> dict[str, Any]:
    """fluid-v2-supplied: accuracy against chance per family × rung (the inferred twins are read by their own
    batteries; the paired difference needs both)."""
    by: dict[tuple[str, int], list[bool]] = defaultdict(list)
    chance: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in rows:
        key = (r["family"], int(r["rung"]))
        by[key].append(_ok(r))
        chance[key].append(1 / len(r["candidates"]))
    return {f"{f}/r{g}": {"n": len(x), "acc": sum(x) / len(x), "chance": sum(chance[(f, g)]) / len(x)}
            for (f, g), x in sorted(by.items())}


def _paired_ratio(u: Mapping[str, float], s: Mapping[str, float], n: int = 2000, seed: int = 0) -> dict[str, Any]:
    """U − S and R = mean U / mean S over the items both carry, with a paired bootstrap over items."""
    ids = sorted(set(u) & set(s))
    rng = random.Random(seed)
    du = [u[i] - s[i] for i in ids]
    lo, hi = boot_ci(du, n, seed)
    mu, ms = sum(u[i] for i in ids) / len(ids), sum(s[i] for i in ids) / len(ids)
    rs = []
    for _ in range(n):
        b = [rng.choice(ids) for _ in ids]
        bs = sum(s[i] for i in b)
        if bs != 0:
            rs.append(sum(u[i] for i in b) / bs)
    rs.sort()
    return {"n": len(ids), "diff": mu - ms, "diff_ci": [lo, hi], "ratio": mu / ms if ms else None,
            "ratio_ci": [rs[int(.025 * len(rs))], rs[int(.975 * len(rs))]] if rs else None}


def register(rows: Sequence[Row]) -> dict[str, Any]:
    """register-v1 (:mod:`binding_bench.binding.register`): per construct × register × condition its gauge reading
    (static, dynamic: the speaker-swap shift of the self and addressee queries, and static's third-person lookup
    accuracy; set: chance-corrected soft accuracy, the held set's share of the candidates' probability, and hit@1),
    then for each unseen register the paired transfer against ``seen`` at the same condition: U − S and
    R = U / S. R is reported only where S's whole interval clears the bar (:data:`BIND`); below it R is noise."""
    out: dict[str, Any] = {}
    per_item: dict[tuple[str, str, str, str], dict[str, float]] = {}
    st = [r for r in rows if r["construct"] == "static" and not r.get("multi_token")]
    by = {(r["register"], r["condition"], r["base_id"], r["qkind"], r["arm"]): r for r in st}
    for reg, cond in sorted({(r["register"], r["condition"]) for r in st}):
        ids = sorted({r["base_id"] for r in st if r["register"] == reg and r["condition"] == cond})
        third = [by[(reg, cond, i, "third", "orig")] for i in ids if (reg, cond, i, "third", "orig") in by]
        cell: dict[str, Any] = {"scenes": len(ids), "lookup": sum(_ok(r) for r in third) / max(1, len(third))}
        for q in ("self", "addressee"):
            d = {i: _pnorm(by[(reg, cond, i, q, "orig")], by[(reg, cond, i, q, "orig")]["gold"])
                 - _pnorm(by[(reg, cond, i, q, "swap")], by[(reg, cond, i, q, "orig")]["gold"])
                 for i in ids if (reg, cond, i, q, "orig") in by and (reg, cond, i, q, "swap") in by}
            cell[f"swap_{q}"] = _shift(list(d.values()))
            per_item[("static", f"swap_{q}", reg, cond)] = d
        out[f"static/{reg}/{cond}"] = cell
    dy = [r for r in rows if r["construct"] == "dynamic" and not r.get("multi_token")]
    pair: dict[tuple[str, str, str, str], dict[str, Row]] = defaultdict(dict)
    for r in dy:
        pair[(r["register"], r["condition"], r["qkind"], r["base_id"])][r["arm"]] = r
    for reg, cond in sorted({(r["register"], r["condition"]) for r in dy}):
        cell = {}
        for q in ("self", "addressee"):
            d = {b: _pnorm(v["orig"], v["orig"]["gold_orig"]) - _pnorm(v["swap"], v["orig"]["gold_orig"])
                 for (rg, c, qq, b), v in pair.items() if rg == reg and c == cond and qq == q and "orig" in v and "swap" in v}
            cell[f"swap_{q}"] = _shift(list(d.values()))
            per_item[("dynamic", f"swap_{q}", reg, cond)] = d
        out[f"dynamic/{reg}/{cond}"] = cell
    se = [r for r in rows if r["construct"] == "set" and not r.get("multi_token")]
    for reg, cond in sorted({(r["register"], r["condition"]) for r in se}):
        soft: dict[str, float] = {}
        hits: list[float] = []
        chance: list[float] = []
        for r in se:
            if r["register"] != reg or r["condition"] != cond:
                continue
            n, k = len(r["candidates"]), len(r["gold_set"])
            p_set = sum(_pnorm(r, o) for o in r["gold_set"])
            soft[r["base_id"]] = (p_set - k / n) / (1 - k / n)
            hits.append(float(max(r["cand_lps"], key=r["cand_lps"].get) in r["gold_set"]))
            chance.append(k / n)
        hit, ch = sum(hits) / len(hits), sum(chance) / len(chance)
        out[f"set/{reg}/{cond}"] = {"items": len(soft), "soft": _shift(list(soft.values())), "hit1": hit,
                                    "hit1_kappa": (hit - ch) / (1 - ch)}
        per_item[("set", "soft", reg, cond)] = soft
    transfer: dict[str, Any] = {}
    for (con, reading, reg, cond), u in sorted(per_item.items()):
        if reg == "seen" or ("seen" not in {k[2] for k in per_item} or (con, reading, "seen", cond) not in per_item):
            continue
        s = per_item[(con, reading, "seen", cond)]
        t = _paired_ratio(u, s)
        s_lo = boot_ci(list(s.values()))[0]
        if s_lo <= BIND:
            t["ratio"], t["ratio_ci"], t["ratio_note"] = None, None, f"seen interval not above {BIND}: R undefined"
        transfer[f"{con}/{reading}/{reg}/{cond}"] = t
    out["transfer"] = transfer
    return out


READERS = {"joinlite": joinlite, "swapdyn": swapdyn, "decomp": decomp, "frame": frame, "capacity": capacity,
           "fluid-morph": fluid, "fluid-algebra": fluid, "fluid-rebind": fluid, "fluid-seqind": fluid,
           "cryst": fluid, "formupdate": formupdate, "fluid-supplied": fluid_supplied,
           "register": register}


def _f(x: Any, fmt: str = "{:+.3f}") -> str:
    return "—" if x is None else fmt.format(x)


SCORECARD_COLUMNS = ("static I", "static you", "dyn-you 0shot", "dyn-you 3shot", "decomp third",
                     "decomp deictic", "capacity K50", "K50 D256")


def scorecard_row(results: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """One scorecard row from per-battery reader results (keyed by reader name)."""
    jl, sd, dc, cp = (results.get(k) or {} for k in ("joinlite", "swapdyn", "decomp", "capacity"))
    p2 = jl.get("P2", {})

    def shift(cell: Mapping[str, Any]) -> str:
        if not cell:
            return "—"
        return f"{cell['shift']:+.3f} {cell['reading'][:4]}"

    return {
        "static I": shift(p2.get("swap_self", {})),
        "static you": shift(p2.get("swap_addressee", {})),
        "dyn-you 0shot": shift(sd.get("0shot/addressee", {})),
        "dyn-you 3shot": shift(sd.get("3shot/addressee", {})),
        "decomp third": _f(dc.get("third/0shot", {}).get("acc"), "{:.3f}"),
        "decomp deictic": _f(dc.get("deictic/0shot", {}).get("acc"), "{:.3f}"),
        "capacity K50": cp.get("K50_label", "—"),
        "K50 D256": cp.get("K50_d256_fixed") or "—",
    }
