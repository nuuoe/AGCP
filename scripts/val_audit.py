"""Validate AGCP PlanBench plans with VAL, independently of pyperplan.

Reads task_1_plan_generation.json under plan-bench/responses/<config>/
<engine_label>/ for each config in CONFIGS and writes per-instance VAL
verdicts to --out.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


VAL_BIN = ("external/LLMs-Planning/planner_tools/VAL/"
           "bin/MacOSExecutables/Validate")
PB = "external/LLMs-Planning/plan-bench"

CONFIGS = {
    "blocksworld_3": (
        f"{PB}/instances/blocksworld/generated_domain.pddl",
        f"{PB}/instances/blocksworld/generated_basic_3",
    ),
    "mystery_blocksworld_3": (
        f"{PB}/instances/blocksworld/mystery/generated_domain.pddl",
        f"{PB}/instances/blocksworld/mystery/generated_basic_3",
    ),
    "logistics": (
        f"{PB}/instances/logistics/generated_domain.pddl",
        f"{PB}/instances/logistics/generated_basic",
    ),
    "depots": (
        f"{PB}/instances/depots/generated_domain.pddl",
        f"{PB}/instances/depots/generated_basic",
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine_label", default="final_qwen1.5b")
    ap.add_argument("--repo_root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    val_bin = os.path.join(args.repo_root, VAL_BIN)
    results = {}
    for cfg, (dom_rel, inst_dir_rel) in CONFIGS.items():
        dom = os.path.join(args.repo_root, dom_rel)
        inst_dir = os.path.join(args.repo_root, inst_dir_rel)
        resp_json = os.path.join(
            args.repo_root, PB, "responses", cfg, args.engine_label,
            "task_1_plan_generation.json")
        if not os.path.exists(resp_json):
            print(f"{cfg}: response JSON missing"); continue
        d = json.load(open(resp_json))
        n_val = n_total = 0
        per_instance = []
        for i in d["instances"][:10]:
            plan = i.get("llm_raw_response", "")
            if not plan: continue
            inst_id = i["instance_id"]
            inst_pddl = os.path.join(inst_dir, f"instance-{inst_id}.pddl")
            if not os.path.exists(inst_pddl): continue
            plan_file = "/tmp/agcp_val_plan.txt"
            with open(plan_file, "w") as f: f.write(plan)
            try:
                r = subprocess.run(
                    [val_bin, dom, inst_pddl, plan_file],
                    capture_output=True, text=True, timeout=30,
                )
                ok = "Plan valid" in r.stdout
            except Exception:
                ok = False
            n_total += 1
            if ok: n_val += 1
            per_instance.append({"inst": inst_id, "val_valid": ok})
        results[cfg] = {"n_val_valid": n_val, "n_total": n_total,
                          "per_instance": per_instance}
        print(f"{cfg:30s}  VAL: {n_val}/{n_total} = "
              f"{100*n_val/max(n_total,1):.1f}%")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
