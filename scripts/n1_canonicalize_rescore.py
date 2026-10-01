"""Rescore the logistics NL-goal parse run after canonicalising object names.

Verbose names in LLM proposals (package_0, location_1_0) are mapped to the
PDDL aliases (p0, l1-0) and the verifier is re-applied; the LLM is not
re-run. Reads runs/n1_parse_corrected_logistics.json and writes
runs/n1_parse_corrected_canon_logistics.json.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


def build_alias_map(obj_set: set, domain_hint: str = "") -> dict:
    """Build verbose→short alias map for common naming patterns.

    Detects patterns by looking at obj_set:
      pN     ← package_N
      lX-Y   ← location_X_Y
      tN     ← truck_N
      aN     ← airplane_N
      cN     ← city_N
      crateN ← (no alias, short = long)
      depotN ← (no alias, short = long)
    """
    alias = {}
    for o in obj_set:
        m = re.match(r"^p(\d+)$", o)
        if m: alias[f"package_{m.group(1)}"] = o
        m = re.match(r"^l(\d+)-(\d+)$", o)
        if m: alias[f"location_{m.group(1)}_{m.group(2)}"] = o
        m = re.match(r"^t(\d+)$", o)
        if m: alias[f"truck_{m.group(1)}"] = o
        m = re.match(r"^a(\d+)$", o)
        if m: alias[f"airplane_{m.group(1)}"] = o
        m = re.match(r"^c(\d+)$", o)
        if m: alias[f"city_{m.group(1)}"] = o
    return alias


def canonicalize_atom(s: str, alias: dict, obj_set: set) -> str:
    """Rewrite arg names in atom string using alias map."""
    s = s.strip().replace(" ", "")
    m = re.match(r"^([\w-]+)\(([^)]*)\)$", s)
    if not m:
        return s
    head, args_str = m.group(1), m.group(2)
    args = [a.strip() for a in args_str.split(",") if a.strip()]
    new_args = []
    for a in args:
        if a in obj_set:
            new_args.append(a)
        elif a in alias:
            new_args.append(alias[a])
        else:
            new_args.append(a)  # leave as-is; will fail verifier
    return f"{head}({','.join(new_args)})"


def rescore_f1(parsed_atoms: set, gt_atoms: set) -> float:
    """F1 between parsed and GT."""
    if not parsed_atoms or not gt_atoms:
        return 0.0
    tp = len(parsed_atoms & gt_atoms)
    if tp == 0: return 0.0
    prec = tp / len(parsed_atoms)
    rec = tp / len(gt_atoms)
    return 2 * prec * rec / max(prec + rec, 1e-9)


def main():
    cfg = "logistics"
    src = f"runs/n1_parse_corrected_{cfg}.json"
    out = f"runs/n1_parse_corrected_canon_{cfg}.json"

    d = json.load(open(src))
    rows = d["rows"]

    # The object set is reconstructed per row from the GT atoms.
    n_acc_post = 0
    n_acc_pre = 0
    canon_examples = []
    for r in rows:
        if r.get("llm_accepted"): n_acc_pre += 1
        proposal = r.get("llm_proposal", [])
        if not proposal: continue
        obj_set = set()
        for a in r.get("gt_atoms", []):
            m = re.match(r"^([\w-]+)\(([^)]*)\)$", a)
            if m:
                for x in m.group(2).split(","):
                    x = x.strip()
                    if x: obj_set.add(x)
        if not obj_set: continue
        alias = build_alias_map(obj_set)
        canon = set()
        for a in proposal:
            if isinstance(a, str):
                c = canonicalize_atom(a, alias, obj_set)
                canon.add(c)
        gt_set = set(r.get("gt_atoms", []))
        f1 = rescore_f1(canon, gt_set)
        accepted_post = f1 >= 0.9
        if accepted_post: n_acc_post += 1
        if accepted_post and not r.get("llm_accepted"):
            canon_examples.append({
                "inst": r["inst"],
                "before": proposal,
                "after": sorted(canon),
                "gt": sorted(gt_set),
                "f1_post": round(f1, 3),
            })

    n = len(rows)
    print(f"=== Logistics N1 rescore with canonicalization ===")
    print(f"Pre-canon  accept: {n_acc_pre}/{n} = {100*n_acc_pre/n:.1f}%")
    print(f"Post-canon accept: {n_acc_post}/{n} = {100*n_acc_post/n:.1f}%")
    print(f"NEW accepts via canonicalization: {n_acc_post - n_acc_pre}")
    print()
    for ex in canon_examples[:5]:
        print(f"  inst {ex['inst']} ({ex['f1_post']}):")
        print(f"    before: {ex['before']}")
        print(f"    after:  {ex['after']}")
    Path(out).write_text(json.dumps({
        "n_pre": n_acc_pre, "n_post": n_acc_post, "n_total": n,
        "new_accepts": n_acc_post - n_acc_pre,
        "examples": canon_examples,
    }, indent=2))
    print(f"\nwrote: {out}")


if __name__ == "__main__":
    main()
