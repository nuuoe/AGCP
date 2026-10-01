"""Categorize NL-goal-parse failures by error type.

Reads runs/n1_parse_uncorrected_<config>.json for each config and writes
runs/n1_failure_modes.json with per-category counts and examples.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


CONFIGS = ["blocksworld_3", "mystery_blocksworld_3", "logistics",
            "depots"]

ATOM_RE = re.compile(r"^([\w-]+)\(([^)]*)\)$")


def parse_atom(s: str):
    s = s.strip().replace(" ", "")
    m = ATOM_RE.match(s)
    if not m: return None
    head, args = m.group(1), m.group(2)
    a = [x for x in args.split(",") if x]
    return head, tuple(a)


def categorize(gt_atoms, proposal_atoms, predicate_vocab, object_ids):
    """Assign one error category to a proposal.

    CORRECT (exact match), ARG_SWAP (same atoms, args permuted),
    HALLUCINATED_HEAD / HALLUCINATED_OBJ (unknown predicate / object),
    WRONG_ARITY, ALIAS (right heads and arities, wrong object ids),
    MISSING_ATOMS / EXTRA_ATOMS (strict subset / superset of GT),
    EMPTY, OTHER.
    """
    gt = [parse_atom(a) for a in gt_atoms]
    gt = [a for a in gt if a is not None]
    gt_set = set(gt)
    p = [parse_atom(a) for a in proposal_atoms]
    p = [a for a in p if a is not None]
    p_set = set(p)

    if not p:
        return "EMPTY"
    # exact-match (accepted under canonical form)
    if gt_set == p_set:
        return "CORRECT"

    # ARG_SWAP: same heads, same arg multiset per atom, but
    # the (head, arg-tuple) doesn't match
    gt_multiset = sorted([(h, tuple(sorted(a))) for (h, a) in gt])
    p_multiset = sorted([(h, tuple(sorted(a))) for (h, a) in p])
    if gt_multiset == p_multiset and gt != p:
        return "ARG_SWAP"

    # HALLUCINATED_HEAD
    if predicate_vocab is not None:
        for h, _ in p:
            if h not in predicate_vocab:
                return "HALLUCINATED_HEAD"

    # HALLUCINATED_OBJ
    if object_ids is not None:
        for _, args in p:
            for a in args:
                if a not in object_ids:
                    return "HALLUCINATED_OBJ"

    # WRONG_ARITY (compare to gt heads' arities)
    gt_arity = {h: len(a) for (h, a) in gt}
    for h, args in p:
        if h in gt_arity and len(args) != gt_arity[h]:
            return "WRONG_ARITY"

    # ALIAS: heads and arities fit but no atom matches, so the object ids
    # are wrong.
    heads_match = all(
        (h in gt_arity) for (h, _) in p)
    if heads_match and not (gt_set & p_set):
        return "ALIAS"

    # MISSING / EXTRA
    if p_set < gt_set:
        return "MISSING_ATOMS"
    if p_set > gt_set:
        return "EXTRA_ATOMS"

    return "OTHER"


def main():
    out = {}
    examples = {}
    for cfg in CONFIGS:
        path = f"runs/n1_parse_uncorrected_{cfg}.json"
        if not Path(path).exists():
            continue
        d = json.load(open(path))
        rows = d.get("rows", [])
        # The valid vocabulary is taken from gt_atoms across all rows,
        # since the domain was not logged per row.
        all_heads = set()
        all_args = set()
        for r in rows:
            for s in r.get("gt_atoms", []):
                pa = parse_atom(s)
                if pa:
                    h, args = pa
                    all_heads.add(h)
                    for a in args: all_args.add(a)

        ctr = Counter()
        for r in rows:
            cat = categorize(
                r.get("gt_atoms", []),
                r.get("llm_proposal", []),
                predicate_vocab=all_heads,
                object_ids=all_args,
            )
            ctr[cat] += 1
            examples.setdefault(cfg, {}).setdefault(cat, []).append({
                "inst": r.get("inst"),
                "nl_goal": r.get("nl_goal", "")[:140],
                "gt_atoms": r.get("gt_atoms", []),
                "llm_proposal": r.get("llm_proposal", []),
            })
        out[cfg] = dict(ctr)
        print(f"=== {cfg}  N={len(rows)} ===")
        for cat in ["CORRECT", "ALIAS", "ARG_SWAP",
                     "HALLUCINATED_HEAD", "HALLUCINATED_OBJ",
                     "WRONG_ARITY", "MISSING_ATOMS",
                     "EXTRA_ATOMS", "EMPTY", "OTHER"]:
            n = ctr.get(cat, 0)
            if n: print(f"  {cat:18s} {n:3d}  ({100*n/len(rows):.1f}%)")

    out_path = Path("runs/n1_failure_modes.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"counts": out, "examples": examples}, f, indent=2)
    print(f"\nwrote: {out_path}")


if __name__ == "__main__":
    main()
