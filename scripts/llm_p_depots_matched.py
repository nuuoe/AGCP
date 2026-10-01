"""LLM+P on the Depots instance range used by AGCP (2..31 by default).

Runs the scripts/llm_p_baseline.py pipeline on instance-<i>.pddl for i
in [--first, --last_inclusive], so the comparison is paired on the same
30 instances. Output: runs/llm_p_depots_matched_n30.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from scripts.llm_p_baseline import (
    extract_problem_nl, build_llm_p_prompt, call_llm,
    extract_pddl, try_solve,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pb_root", default="external/LLMs-Planning/plan-bench")
    ap.add_argument("--first", type=int, default=2)
    ap.add_argument("--last_inclusive", type=int, default=31)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="runs/llm_p_depots_matched_n30.json")
    args = ap.parse_args()

    domain_path = os.path.join(args.pb_root,
                                  "instances/depots/generated_domain.pddl")
    inst_dir = os.path.join(args.pb_root,
                              "instances/depots/generated_basic")
    domain_text = open(domain_path).read()

    instances = []
    for i in range(args.first, args.last_inclusive + 1):
        p = os.path.join(inst_dir, f"instance-{i}.pddl")
        if os.path.isfile(p):
            instances.append((i, p))
    print(f"Matched-set LLM+P on Depots — {len(instances)} instances "
          f"({args.first}..{args.last_inclusive}), model={args.model}")

    results = []
    for inst_id, pp in instances:
        print(f"\n  inst {inst_id}:")
        nl = extract_problem_nl(pp)
        prompt = build_llm_p_prompt(nl, domain_text)
        t0 = time.time()
        try:
            gen = call_llm(prompt, model=args.model)
        except Exception as e:
            print(f"    LLM call error: {e}")
            results.append({"inst": inst_id, "problem": Path(pp).name,
                             "llm_success": False, "error": str(e)})
            continue
        gen_time = time.time() - t0
        problem_pddl = extract_pddl(gen)
        if not problem_pddl:
            print(f"    LLM emitted no PDDL block (gen={gen_time:.1f}s)")
            results.append({"inst": inst_id, "problem": Path(pp).name,
                             "llm_success": False, "error": "no pddl",
                             "llm_time": gen_time})
            continue
        solve_res = try_solve(domain_path, problem_pddl)
        success = solve_res.get("success", False)
        results.append({
            "inst": inst_id, "problem": Path(pp).name,
            "llm_success": success,
            "llm_time": gen_time,
            "plan_length": solve_res.get("plan_length"),
            "solve_time": solve_res.get("time"),
            "error": solve_res.get("error"),
        })
        flag = "OK" if success else f"FAIL: {solve_res.get('error','?')}"
        print(f"    LLM gen {gen_time:.1f}s; {flag}; "
              f"plan_len={solve_res.get('plan_length')}")

    n_solved = sum(1 for r in results if r.get("llm_success"))
    print("\n" + "=" * 60)
    print(f"Matched LLM+P Depots: {n_solved}/{len(results)} = "
          f"{100*n_solved/max(len(results),1):.1f}%")
    print(f"(AGCP on same set: 30/30 = 100%)")
    print("=" * 60)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "model": args.model,
            "instance_set": f"depots [{args.first}..{args.last_inclusive}]",
            "n_solved": n_solved,
            "n_total": len(results),
            "results": results,
        }, f, indent=2, default=str)
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
