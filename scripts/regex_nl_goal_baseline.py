"""Regex baseline for NL-to-goal parsing on PlanBench.

One hand-written clause parser per domain (PARSERS); an instance is
accepted iff F1(parsed, gold) >= threshold (default 0.9). Reads
runs/n1_alias_prompted_<domain>.json and writes runs/regex_baseline_<domain>.json.
Usage: python -m scripts.regex_nl_goal_baseline --domain all --out_dir runs
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Callable


BW_COLOR_TO_LETTER = {
    "red": "a", "blue": "b", "orange": "c",
    "yellow": "d", "green": "e", "white": "f",
    "purple": "g", "pink": "h", "black": "i", "cyan": "j",
}


def _split_clauses(nl: str) -> list[str]:
    s = nl.strip().rstrip(".")
    parts = re.split(r"\s*,\s*and\s+|\s+and\s+|\s*,\s*", s)
    return [p.strip() for p in parts if p.strip()]


def parse_blocksworld(nl: str) -> set[str]:
    atoms: set[str] = set()
    for c in _split_clauses(nl):
        m = re.match(
            r"the\s+(\w+)\s+block\s+is\s+on\s+top\s+of\s+the\s+(\w+)\s+block",
            c, re.IGNORECASE)
        if m:
            x = BW_COLOR_TO_LETTER.get(m.group(1).lower())
            y = BW_COLOR_TO_LETTER.get(m.group(2).lower())
            if x and y:
                atoms.add(f"on({x},{y})")
            continue
        m = re.match(
            r"the\s+(\w+)\s+block\s+is\s+on\s+the\s+table",
            c, re.IGNORECASE)
        if m:
            x = BW_COLOR_TO_LETTER.get(m.group(1).lower())
            if x:
                atoms.add(f"ontable({x})")
            continue
        m = re.match(
            r"the\s+(\w+)\s+block\s+is\s+clear",
            c, re.IGNORECASE)
        if m:
            x = BW_COLOR_TO_LETTER.get(m.group(1).lower())
            if x:
                atoms.add(f"clear({x})")
    return atoms


def parse_depots(nl: str) -> set[str]:
    atoms: set[str] = set()
    for c in _split_clauses(nl):
        m = re.match(r"(\S+)\s+is\s+on\s+(\S+)$", c, re.IGNORECASE)
        if m:
            atoms.add(f"on({m.group(1)},{m.group(2)})")
            continue
        m = re.match(r"(\S+)\s+is\s+in\s+(\S+)$", c, re.IGNORECASE)
        if m:
            atoms.add(f"in({m.group(1)},{m.group(2)})")
            continue
        m = re.match(r"(\S+)\s+is\s+at\s+(\S+)$", c, re.IGNORECASE)
        if m:
            atoms.add(f"at({m.group(1)},{m.group(2)})")
    return atoms


def _logistics_compact(name: str) -> str:
    # package_0 -> p0; location_0_0 -> l0-0; truck_0 -> t0; airplane_0 -> a0
    m = re.match(r"package_(\d+)", name)
    if m:
        return f"p{m.group(1)}"
    m = re.match(r"location_(\d+)_(\d+)", name)
    if m:
        return f"l{m.group(1)}-{m.group(2)}"
    m = re.match(r"truck_(\d+)", name)
    if m:
        return f"t{m.group(1)}"
    m = re.match(r"airplane_(\d+)", name)
    if m:
        return f"a{m.group(1)}"
    m = re.match(r"city_(\d+)", name)
    if m:
        return f"c{m.group(1)}"
    return name


def parse_logistics(nl: str) -> set[str]:
    atoms: set[str] = set()
    for c in _split_clauses(nl):
        m = re.match(r"(\S+)\s+is\s+at\s+(\S+)$", c, re.IGNORECASE)
        if m:
            x = _logistics_compact(m.group(1))
            y = _logistics_compact(m.group(2))
            atoms.add(f"at({x},{y})")
            continue
        m = re.match(r"(\S+)\s+is\s+in\s+(\S+)$", c, re.IGNORECASE)
        if m:
            x = _logistics_compact(m.group(1))
            y = _logistics_compact(m.group(2))
            atoms.add(f"in({x},{y})")
    return atoms


def parse_mystery_bw(nl: str) -> set[str]:
    atoms: set[str] = set()
    for c in _split_clauses(nl):
        m = re.match(
            r"object\s+(\S+)\s+craves\s+object\s+(\S+)$",
            c, re.IGNORECASE)
        if m:
            atoms.add(f"craves({m.group(1)},{m.group(2)})")
            continue
        m = re.match(
            r"object\s+(\S+)\s+harmony$", c, re.IGNORECASE)
        if m:
            atoms.add(f"harmony({m.group(1)})")
    return atoms


PARSERS: dict[str, Callable[[str], set[str]]] = {
    "blocksworld_3": parse_blocksworld,
    "depots": parse_depots,
    "logistics": parse_logistics,
    "mystery_blocksworld_3": parse_mystery_bw,
}


def f1(pred: set[str], gt: set[str]) -> float:
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    tp = len(pred & gt)
    prec = tp / len(pred)
    rec = tp / len(gt)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def run_domain(domain: str, threshold: float = 0.9) -> dict:
    src = f"runs/n1_alias_prompted_{domain}.json"
    j = json.load(open(src))
    parser = PARSERS[domain]
    rows = []
    n_accept = 0
    n_perfect = 0
    f1_sum = 0.0
    for r in j["rows"]:
        nl = r.get("nl_goal") or r.get("nl") or ""
        gt_atoms = set(r.get("gt_atoms", []))
        pred = parser(nl)
        score = f1(pred, gt_atoms)
        accepted = score >= threshold
        if accepted:
            n_accept += 1
        if score == 1.0:
            n_perfect += 1
        f1_sum += score
        rows.append({
            "inst": r.get("inst"),
            "nl_goal": nl,
            "gt_atoms": sorted(gt_atoms),
            "regex_parsed": sorted(pred),
            "f1": score,
            "accepted": accepted,
        })
    n = len(rows)
    return {
        "domain": domain,
        "n_total": n,
        "n_accept_at_0.9": n_accept,
        "n_perfect_f1": n_perfect,
        "accept_rate": n_accept / max(n, 1),
        "mean_f1": f1_sum / max(n, 1),
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="all",
                    choices=list(PARSERS.keys()) + ["all"])
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--out_dir", default="runs")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    domains = [args.domain] if args.domain != "all" else list(PARSERS.keys())

    pooled_accept = 0
    pooled_total = 0
    pooled_perfect = 0
    print(f"\n{'domain':<24} {'N':>4} {'acc@0.9':>10} {'perfect':>9} {'mean_F1':>8}")
    print("-" * 60)
    for d in domains:
        res = run_domain(d, threshold=args.threshold)
        with open(f"{args.out_dir}/regex_baseline_{d}.json", "w") as f:
            json.dump(res, f, indent=2)
        pooled_accept += res["n_accept_at_0.9"]
        pooled_total += res["n_total"]
        pooled_perfect += res["n_perfect_f1"]
        print(f"{d:<24} {res['n_total']:>4} "
              f"{res['n_accept_at_0.9']:>5}/{res['n_total']:<4} "
              f"{res['n_perfect_f1']:>9} "
              f"{res['mean_f1']:>8.3f}")
    print("-" * 60)
    print(f"{'POOLED':<24} {pooled_total:>4} "
          f"{pooled_accept:>5}/{pooled_total:<4} "
          f"{pooled_perfect:>9}")
    print()
    print(f"=> regex baseline: {pooled_accept}/{pooled_total} = "
          f"{100*pooled_accept/max(pooled_total,1):.1f}% "
          f"accept at F1>={args.threshold}")


if __name__ == "__main__":
    main()
