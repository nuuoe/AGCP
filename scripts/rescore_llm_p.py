"""Rescore LLM+P outputs with a stricter success criterion.

llm_p_baseline.py counts a zero-length plan as success, which admits
emitted PDDL whose init already equals the goal. Here success requires
a pyperplan plan of length > 0. Inputs: LLM+P result JSON files;
output: runs/llm_p_rescored_nonempty.json (keys kept for downstream scripts).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def rescore(path: str) -> dict:
    d = json.load(open(path))
    results = d.get("results", [])
    n_total = len(results)
    n_orig_ok = sum(1 for r in results if r.get("llm_success"))
    n_strict = sum(1 for r in results
                    if r.get("llm_success") and
                    (r.get("plan_length") or 0) > 0)
    n_trivial = n_orig_ok - n_strict
    return {
        "path": path,
        "n_total": n_total,
        "n_orig_success": n_orig_ok,
        "n_strict_success": n_strict,
        "n_trivial_init_eq_goal": n_trivial,
        "orig_rate": n_orig_ok / max(n_total, 1),
        "strict_rate": n_strict / max(n_total, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", default=[
        "runs/llm_p_depots_lexsorted.json",
        "runs/llm_p_depots_matched_n30.json",
    ])
    ap.add_argument("--out",
                    default="runs/llm_p_rescored_nonempty.json")
    args = ap.parse_args()

    summary = []
    print(f"\n{'file':<50} {'orig':>10} {'strict':>10} {'trivial':>10}")
    print("-" * 80)
    for p in args.inputs:
        if not Path(p).exists():
            print(f"{p}: SKIP (not found)")
            continue
        r = rescore(p)
        summary.append(r)
        print(f"{Path(p).name:<50} "
              f"{r['n_orig_success']:>3}/{r['n_total']:<3}({100*r['orig_rate']:>3.0f}%) "
              f"{r['n_strict_success']:>3}/{r['n_total']:<3}({100*r['strict_rate']:>3.0f}%) "
              f"{r['n_trivial_init_eq_goal']:>10}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "criterion": "strict = orig_success AND plan_length > 0",
            "summary": summary,
        }, f, indent=2)
    print(f"\nwrote: {args.out}")


if __name__ == "__main__":
    main()
