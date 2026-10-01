"""Closed-loop PlanBench demo with no ground-truth fallback in the NL parse.

Variant of demo_closed_loop.py: the parse baseline is an empty goal, a
failed LLM parse short-circuits the instance, and execution success is
checked against the LLM-parsed goal (GT success is logged alongside).
Outputs: runs/n4_no_gt_fallback_<config>_inst<N>.json and _summary.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from scripts.demo_closed_loop import (
    step1_rollouts, step2_induce_schemas,
    step4_build_cfg, step5_decode_plan,
)


def step3_parse_nl_goal_no_fallback(nl_goal, vocab, model_name,
                                       gt_atoms=None):
    """LLM-only NL parse: no GT substitution on failure."""
    from agplan.llm_propose import propose_verify_fallback
    from scripts.benchmark_nl_goal_parse import (
        verifier_factory, build_prompt, GOAL_SCHEMA,
        map_colors_to_letters,
    )
    t0 = time.time()
    nl = map_colors_to_letters(nl_goal, vocab["objects"])
    prompt = build_prompt(nl, vocab)
    verifier = verifier_factory(set(gt_atoms or []), vocab)
    baseline = {"goal": []}  # empty baseline: no GT substitution
    final, info = propose_verify_fallback(
        prompt=prompt, schema_json=GOAL_SCHEMA,
        verifier=verifier, baseline=baseline,
        threshold=0.5, model_name=model_name,
        max_new_tokens=256, n_samples=4, temperature=0.4,
        max_refine_rounds=2,
    )
    parsed = (final.get("goal", [])
              if isinstance(final, dict) else [])
    return {
        "nl_goal": nl_goal,
        "nl_mapped": nl,
        "parsed_goal": parsed,
        "llm_accepted": info.get("accepted_llm", False),
        "verifier_score": info.get("accuracy", 0.0),
        "refine_rounds": info.get("refine_rounds", 0),
        "seconds": time.time() - t0,
    }, set(parsed)


def step6_execute_against_parsed(plan_raw, task, parsed_goal_atoms):
    """Execute the plan and check success against the LLM-parsed goal; GT success is logged too."""
    from scripts.run_agcp_on_apb_simple import (
        _llm_json_to_lisp, parse_lisp_plan,
    )
    t0 = time.time()
    plan_lisp = _llm_json_to_lisp(plan_raw)
    plan = parse_lisp_plan(plan_lisp)
    state = task.initial_state
    op_by_name = {op.name: op for op in task.operators}
    n_applied = 0
    log = []
    for action, args in plan:
        op_name = f"({action} {' '.join(args)})"
        op = op_by_name.get(op_name)
        if op is None:
            log.append({"action": op_name, "applied": False,
                          "reason": "unknown op"})
            return {"success_against_parsed": False,
                     "success_against_gt": task.goals <= state,
                     "n_applied": n_applied, "plan_log": log,
                     "seconds": time.time() - t0}
        if not op.applicable(state):
            log.append({"action": op_name, "applied": False,
                          "reason": "not applicable"})
            return {"success_against_parsed": False,
                     "success_against_gt": task.goals <= state,
                     "n_applied": n_applied, "plan_log": log,
                     "seconds": time.time() - t0}
        state = op.apply(state)
        n_applied += 1
        log.append({"action": op_name, "applied": True})
    # state and parsed_goal_atoms share the induce_pddl_generic atom format.
    success_parsed = bool(parsed_goal_atoms) and parsed_goal_atoms <= state
    success_gt = task.goals <= state
    return {
        "success_against_parsed": success_parsed,
        "success_against_gt": success_gt,
        "n_applied": n_applied,
        "plan_lisp": plan_lisp,
        "plan_log": log,
        "seconds": time.time() - t0,
    }


def run_one_instance(plan_bench_root, config, instance_id, model_name):
    import yaml
    cfg_yaml = yaml.safe_load(open(os.path.join(
        plan_bench_root, "configs", f"{config}.yaml")))
    domain_path = os.path.join(plan_bench_root, "instances",
                                  cfg_yaml["domain_file"])
    instance_path = os.path.join(
        plan_bench_root, "instances",
        cfg_yaml["instance_dir"],
        cfg_yaml["instances_template"].format(instance_id))

    trace = {"instance_id": instance_id}

    s1, trs_ok, task = step1_rollouts(domain_path, instance_path)
    trace["step1"] = s1

    s2, models = step2_induce_schemas(trs_ok)
    trace["step2"] = s2

    prompts_json = os.path.join(plan_bench_root, "prompts",
                                  cfg_yaml["domain_name"],
                                  "task_1_plan_generation.json")
    nl_goal = ""
    if os.path.exists(prompts_json):
        prompts = json.load(open(prompts_json))
        for ip in prompts["instances"]:
            if ip["instance_id"] == instance_id:
                # Take the last match: the prompt's ICL examples use
                # the same phrasing.
                ms = re.findall(
                    r"My goal is to have that (.+?)(?:\.|$)",
                    ip["query"], re.DOTALL)
                if ms:
                    nl_goal = ms[-1].strip()
                break
    if not nl_goal:
        trace["error"] = "no NL goal in prompt"
        trace["llm_e4_succeeded"] = False
        trace["success_against_parsed"] = False
        trace["success_against_gt"] = False
        return trace

    from scripts.benchmark_nl_goal_parse import (
        build_vocab, _extract_goal_atoms,
    )
    vocab = build_vocab(domain_path, instance_path)
    gt_atoms = _extract_goal_atoms(open(instance_path).read())
    s3, parsed_goal = step3_parse_nl_goal_no_fallback(
        nl_goal, vocab, model_name, gt_atoms=gt_atoms,
    )
    trace["step3_nl_parse"] = s3
    trace["gt_atoms"] = sorted(gt_atoms)
    trace["parsed_goal"] = sorted(parsed_goal)

    llm_e4_ok = (s3.get("llm_accepted", False)
                  and len(parsed_goal) > 0)
    trace["llm_e4_succeeded"] = llm_e4_ok

    if not llm_e4_ok:
        trace["short_circuit"] = "LLM E4 failed, skipping step4-6"
        trace["success_against_parsed"] = False
        trace["success_against_gt"] = False
        return trace

    # The CFG is built on the parsed goal, not the GT goal.
    from scripts.induce_pddl_generic import state_to_predicates
    init_state = state_to_predicates(task.initial_state)
    s4, cfg_obj = step4_build_cfg(models, init_state, parsed_goal, task)
    trace["step4"] = s4
    if not cfg_obj:
        trace["short_circuit"] = "no CFG"
        trace["success_against_parsed"] = False
        trace["success_against_gt"] = False
        return trace

    s5_out = step5_decode_plan(cfg_obj, model_name)
    if isinstance(s5_out, tuple):
        s5, raw = s5_out
    else:
        s5, raw = s5_out, ""
    trace["step5"] = s5

    s6 = step6_execute_against_parsed(raw, task, parsed_goal)
    trace["step6"] = s6
    trace["success_against_parsed"] = s6["success_against_parsed"]
    trace["success_against_gt"] = s6["success_against_gt"]
    return trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan_bench_root",
                    default="external/LLMs-Planning/plan-bench")
    ap.add_argument("--config", default="mystery_blocksworld_3")
    ap.add_argument("--first", type=int, default=2)
    ap.add_argument("--last_inclusive", type=int, default=11)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out_dir", default="runs")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    all_traces = []
    n_llm_e4_ok = 0
    n_succ_parsed = 0
    n_succ_gt = 0
    for inst in range(args.first, args.last_inclusive + 1):
        print(f"\n=== inst {inst} ===")
        try:
            tr = run_one_instance(args.plan_bench_root, args.config,
                                    inst, args.model)
        except Exception as e:
            import traceback
            traceback.print_exc()
            tr = {"instance_id": inst, "error": str(e),
                   "llm_e4_succeeded": False,
                   "success_against_parsed": False,
                   "success_against_gt": False}
        all_traces.append(tr)
        with open(f"{args.out_dir}/n4_no_gt_fallback_"
                  f"{args.config}_inst{inst}.json", "w") as f:
            json.dump(tr, f, indent=2, default=str)
        if tr.get("llm_e4_succeeded"): n_llm_e4_ok += 1
        if tr.get("success_against_parsed"): n_succ_parsed += 1
        if tr.get("success_against_gt"): n_succ_gt += 1
        print(f"  llm_e4={tr.get('llm_e4_succeeded')} "
              f"succ_parsed={tr.get('success_against_parsed')} "
              f"succ_gt={tr.get('success_against_gt')}")

    n = len(all_traces)
    print("\n" + "=" * 60)
    print(f"N4 no-GT-fallback / {args.config}")
    print(f"  LLM E4 succeeded (no GT-replay):     {n_llm_e4_ok}/{n}")
    print(f"  End-to-end success vs PARSED goal:   {n_succ_parsed}/{n}")
    print(f"  End-to-end success vs GT goal:       {n_succ_gt}/{n}")
    print("=" * 60)

    with open(f"{args.out_dir}/n4_no_gt_fallback_{args.config}_summary.json",
              "w") as f:
        json.dump({
            "config": args.config,
            "model": args.model,
            "n_total": n,
            "n_llm_e4_ok": n_llm_e4_ok,
            "n_success_parsed_goal": n_succ_parsed,
            "n_success_gt_goal": n_succ_gt,
            "traces": all_traces,
        }, f, indent=2, default=str)


if __name__ == "__main__":
    main()
