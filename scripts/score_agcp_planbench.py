"""Score AGCP plans on PlanBench instances by simulating them with pyperplan.

PlanBench's response_evaluation.py expects natural-language plans; AGCP
emits PDDL, so each `llm_raw_response` is parsed as a LISP plan, applied
through pyperplan's grounded operators from the initial state and checked
against the goal. Output: <task>_score.json beside the responses file, or --out.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import yaml


def parse_lisp_plan(text: str) -> list[tuple]:
    """Parse `(action arg1 arg2)\\n...` into [(action, args), ...]."""
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if not line: continue
        m = re.match(r"^\(([\w-]+)\s*(.*?)\)\s*$", line)
        if not m: continue
        head = m.group(1).lower()
        args = [a for a in m.group(2).split() if a]
        out.append((head, tuple(args)))
    return out


def apply_plan(domain_pddl: str, instance_pddl: str,
                 plan: list[tuple]) -> tuple[bool, str, int]:
    """Apply plan through pyperplan operators; return
    (success, error_or_empty, n_applied)."""
    from scripts.induce_pddl_generic import load_task

    try:
        _, task = load_task(domain_pddl, instance_pddl)
    except Exception as e:
        return False, f"load_task: {e}", 0
    state = task.initial_state
    op_by_name = {op.name: op for op in task.operators}
    n_applied = 0
    for action, args in plan:
        op_name = f"({action} {' '.join(args)})"
        op = op_by_name.get(op_name)
        if op is None:
            return False, f"unknown op {op_name}", n_applied
        if not op.applicable(state):
            return False, f"not applicable: {op_name}", n_applied
        state = op.apply(state)
        n_applied += 1
    success = task.goals <= state
    return success, "" if success else "goal not satisfied", n_applied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan_bench_root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--engine_label", required=True)
    ap.add_argument("--task", default="t1")
    ap.add_argument("--out", required=False)
    args = ap.parse_args()

    cfg_path = os.path.join(args.plan_bench_root, "configs",
                              f"{args.config}.yaml")
    cfg = yaml.safe_load(open(cfg_path))
    domain_name = cfg["domain_name"]
    domain_file = os.path.join(args.plan_bench_root, "instances",
                                  cfg["domain_file"])

    task_to_filename = {
        "t1": "task_1_plan_generation",
        "t4": "task_4_plan_reuse",
        "t5": "task_5_plan_generalization",
        "t6": "task_6_replanning",
        "t7": "task_7_plan_execution",
        "t8_1": "task_8_1_goal_shuffling",
        "t8_2": "task_8_2_full_to_partial",
        "t8_3": "task_8_3_partial_to_full",
    }
    task_name = task_to_filename[args.task]
    resp_json = os.path.join(
        args.plan_bench_root, "responses", domain_name,
        args.engine_label, f"{task_name}.json")
    with open(resp_json) as f:
        d = json.load(f)

    results = []
    for inst in d["instances"]:
        if not inst.get("llm_raw_response"):
            continue
        instance_pddl = os.path.join(
            args.plan_bench_root, "instances",
            cfg["instance_dir"],
            cfg["instances_template"].format(inst["instance_id"]))
        plan = parse_lisp_plan(inst["llm_raw_response"])
        success, err, n_applied = apply_plan(
            domain_file, instance_pddl, plan)
        results.append({
            "instance_id": inst["instance_id"],
            "n_actions_in_plan": len(plan),
            "n_applied": n_applied,
            "success": success,
            "error": err,
        })
        flag = "OK" if success else f"FAIL ({err[:50]})"
        print(f"  instance {inst['instance_id']:>3d}: "
              f"plan_len={len(plan):>2d} applied={n_applied:>2d} "
              f"-> {flag}")

    n_solved = sum(1 for r in results if r["success"])
    n_total = len(results)
    rate = 100 * n_solved / max(n_total, 1)
    print("\n" + "=" * 60)
    print(f"AGCP on PlanBench / {domain_name} / {args.task} / "
          f"{args.engine_label}:")
    print(f"  {n_solved}/{n_total} = {rate:.1f}% success")
    print("=" * 60)

    out_path = args.out or os.path.join(
        args.plan_bench_root, "responses", domain_name,
        args.engine_label, f"{task_name}_score.json")
    with open(out_path, "w") as f:
        json.dump({
            "n_solved": n_solved, "n_total": n_total,
            "success_rate": rate / 100.0,
            "per_instance": results,
        }, f, indent=2)
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
