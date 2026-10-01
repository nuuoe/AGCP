"""LLM-removal ablation: solve PlanBench T1 instances with classical search alone.

Parses each domain/instance with pyperplan and runs A* (hAdd), falling
back to BFS; no LLM is involved. Writes per-instance solve results to --out.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan_bench_root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_instances", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, ".")
    from pyperplan.search import astar_search
    from pyperplan.heuristics.relaxation import hAddHeuristic
    from pyperplan.planner import _parse, _ground

    pb_root = args.plan_bench_root
    cfg_yaml = os.path.join(pb_root, "configs", f"{args.config}.yaml")
    import yaml
    cfg = yaml.safe_load(open(cfg_yaml))
    dom_path = os.path.join(pb_root, "instances", cfg["domain_file"])

    rows = []
    n_ok = 0
    for inst_id in range(2, 2 + args.n_instances):
        inst_path = os.path.join(
            pb_root, "instances", cfg["instance_dir"],
            cfg["instances_template"].format(inst_id))
        if not os.path.exists(inst_path):
            continue
        t0 = time.time()
        try:
            problem = _parse(dom_path, inst_path)
            task = _ground(problem,
                            remove_statics_from_initial_state=True,
                            remove_irrelevant_operators=False)
            sol = astar_search(task, hAddHeuristic(task))
            ok = sol is not None
            n_actions = len(sol) if sol else 0
        except Exception as e:
            # fall back to BFS in case heuristic crashes on statics
            try:
                from pyperplan.search import breadth_first_search
                sol = breadth_first_search(task)
                ok = sol is not None
                n_actions = len(sol) if sol else 0
            except Exception:
                ok = False
                n_actions = 0
                sol = None
        t1 = time.time()
        if ok: n_ok += 1
        rows.append({
            "inst": inst_id,
            "solved": ok,
            "n_actions": n_actions,
            "wallclock_sec": t1 - t0,
        })
        print(f"  inst {inst_id:2d}  "
              f"{'OK' if ok else 'fail'}  "
              f"plan_len={n_actions}  "
              f"wall={t1-t0:.1f}s")

    n = len(rows)
    print(f"\n=== LLM-removal ablation / {args.config} ===")
    print(f"Pure classical (no LLM): {n_ok}/{n} = {100*n_ok/max(n,1):.1f}%")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "config": args.config,
            "method": "pure_classical_no_llm",
            "n_solved": n_ok,
            "n_total": n,
            "rows": rows,
        }, f, indent=2)
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
