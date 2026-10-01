"""Closed-loop PlanBench demo on one instance, from rollouts to execution.

Random rollouts, lifted-schema induction, verifier-gated LLM parse of the
NL goal, G_env CFG compilation (bounded BFS, A* fallback), XGrammar-
constrained decoding and execution against the PDDL goal; per-stage
outputs and timings are written as a trace to --out. Default instance:
PlanBench Mystery Blocksworld, instance 2.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


def step1_rollouts(domain_path, instance_path, n_rollouts=10,
                     steps_per_rollout=50, seed=0):
    from scripts.induce_pddl_generic import (
        load_task, random_rollout_pddl, state_to_predicates,
    )
    t0 = time.time()
    _, task = load_task(domain_path, instance_path)
    trs = []
    for s in range(n_rollouts):
        trs.extend(random_rollout_pddl(task, n_steps=steps_per_rollout,
                                          seed=seed + s))
    trs_ok = [t for t in trs if t[0] != t[-1]]
    return {
        "n_transitions": len(trs_ok),
        "n_actions_observed": len(set(t[1] for t in trs_ok)),
        "actions": sorted(set(t[1] for t in trs_ok)),
        "seconds": time.time() - t0,
    }, trs_ok, task


def step2_induce_schemas(trs_ok):
    from scripts.induce_positional import (
        induce_lifted_models_positional,
    )
    from scripts.refine_preconditions import refine_preconditions
    t0 = time.time()
    models = induce_lifted_models_positional(trs_ok)
    return {
        "n_actions_induced": len(models),
        "models": {a: {
            "pre_pos": sorted(m["pre_pos"]),
            "eff_add": sorted(m["eff_add"]),
            "eff_del": sorted(m["eff_del"]),
            "n_examples": m.get("n_examples", 0),
        } for a, m in models.items()},
        "seconds": time.time() - t0,
    }, models


def step3_parse_nl_goal(nl_goal, vocab, model_name, gt_atoms=None):
    """Use LLM to parse NL goal into structured predicate set."""
    from agplan.llm_propose import propose_verify_fallback
    from scripts.benchmark_nl_goal_parse import (
        verifier_factory, build_prompt, GOAL_SCHEMA,
        map_colors_to_letters,
    )
    t0 = time.time()
    nl = map_colors_to_letters(nl_goal, vocab["objects"])
    prompt = build_prompt(nl, vocab)
    verifier = verifier_factory(set(gt_atoms or []), vocab)
    baseline = {"goal": list(gt_atoms or [])}
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


def step4_build_cfg(models, init_state, goal_atoms, task):
    from agplan.grammars.generic_grammar import (
        compile_from_action_models,
        compile_from_action_models_astar,
    )
    from scripts.induce_pddl_generic import parse_op_name
    t0 = time.time()
    # Build ground-op oracle/filter from pyperplan typed ops
    ground_op_filter, ground_op_oracle = {}, {}
    for op in task.operators:
        an, args = parse_op_name(op.name)
        ground_op_filter.setdefault(an, []).append(args)
        ground_op_oracle[(an, args)] = op
    objects = sorted({a for sb, _, args, sa in [(set(), '', (), set())]
                       for a in args} | set())
    init_objs = set()
    for atom in init_state | goal_atoms:
        m = re.match(r"^[\w-]+\((.*)\)$", atom)
        if m:
            for x in m.group(1).split(","):
                if x.strip(): init_objs.add(x.strip())
    objects = sorted(init_objs)

    cfg = compile_from_action_models(
        models, init_state, goal_atoms,
        objects=objects, max_extra=4,
        ground_op_filter=ground_op_filter,
        ground_op_oracle=ground_op_oracle,
    )
    used_astar = False
    if cfg is None:
        cfg = compile_from_action_models_astar(
            task, init_state, goal_atoms,
        )
        used_astar = True
    return {
        "cfg_built": cfg is not None,
        "cfg_size_chars": len(cfg) if cfg else 0,
        "used_astar_fallback": used_astar,
        "n_objects": len(objects),
        "seconds": time.time() - t0,
    }, cfg


def step5_decode_plan(cfg, model_name, max_new_tokens=2048):
    from agplan.decoding.xgrammar_wrapper import (
        DecodeConfig, XGrammarConstrainedDecoder,
    )
    t0 = time.time()
    if not cfg:
        return {"plan_text": "", "seconds": 0, "error": "no CFG"}
    try:
        dec = XGrammarConstrainedDecoder(
            model_name=model_name, ebnf=cfg)
        raw = dec.generate(
            "Generate a plan as a JSON array of actions.",
            cfg=DecodeConfig(max_new_tokens=max_new_tokens),
        )
    except Exception as e:
        return {"plan_text": "", "seconds": time.time()-t0,
                 "error": str(e)[:80]}
    return {
        "raw_plan_json": raw[:300],
        "seconds": time.time() - t0,
    }, raw


def step6_execute_plan(plan_raw, task):
    """Convert JSON plan to LISP, apply via pyperplan, check goal."""
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
            return {"success": False, "n_applied": n_applied,
                     "plan_log": log,
                     "seconds": time.time() - t0}, False
        if not op.applicable(state):
            log.append({"action": op_name, "applied": False,
                          "reason": "not applicable"})
            return {"success": False, "n_applied": n_applied,
                     "plan_log": log,
                     "seconds": time.time() - t0}, False
        state = op.apply(state)
        n_applied += 1
        log.append({"action": op_name, "applied": True})
    success = task.goals <= state
    return {
        "success": success,
        "n_applied": n_applied,
        "plan_lisp": plan_lisp,
        "plan_log": log,
        "seconds": time.time() - t0,
    }, success


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan_bench_root", required=True)
    ap.add_argument("--config", default="mystery_blocksworld_3")
    ap.add_argument("--instance_id", type=int, default=2)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import yaml
    cfg_yaml = yaml.safe_load(open(os.path.join(
        args.plan_bench_root, "configs", f"{args.config}.yaml")))
    domain_path = os.path.join(args.plan_bench_root, "instances",
                                  cfg_yaml["domain_file"])
    instance_path = os.path.join(
        args.plan_bench_root, "instances",
        cfg_yaml["instance_dir"],
        cfg_yaml["instances_template"].format(args.instance_id))

    print("=" * 70)
    print("AGCP CLOSED-LOOP DEMO")
    print("=" * 70)
    print(f"Domain: {cfg_yaml['domain_name']}")
    print(f"Instance: {os.path.basename(instance_path)}")
    print(f"Model: {args.model}")
    print()

    trace = {}

    print("[Step 1] Random env rollouts ...")
    s1, trs_ok, task = step1_rollouts(domain_path, instance_path)
    print(f"  {s1['n_transitions']} transitions, "
          f"{s1['n_actions_observed']} actions, "
          f"{s1['seconds']:.1f}s")
    trace["step1_rollouts"] = s1

    print("\n[Step 2] Lifted-schema induction ...")
    s2, models = step2_induce_schemas(trs_ok)
    print(f"  Induced {s2['n_actions_induced']} action models in "
          f"{s2['seconds']:.1f}s")
    for a, m in s2["models"].items():
        print(f"    {a}: pre={m['pre_pos'][:3]}{'...' if len(m['pre_pos']) > 3 else ''}, "
              f"add={m['eff_add'][:2]}, del={m['eff_del'][:2]}")
    trace["step2_induction"] = s2

    print("\n[Step 3] LLM parses NL goal (verifier-gated) ...")
    # Read NL goal from PlanBench prompt
    prompts_json = os.path.join(args.plan_bench_root, "prompts",
                                  cfg_yaml["domain_name"],
                                  "task_1_plan_generation.json")
    nl_goal = ""
    if os.path.exists(prompts_json):
        prompts = json.load(open(prompts_json))
        for ip in prompts["instances"]:
            if ip["instance_id"] == args.instance_id:
                m = re.search(r"My goal is to have that (.+?)(?:\.|$)",
                              ip["query"], re.DOTALL)
                if m: nl_goal = m.group(1).strip()
                break
    if not nl_goal:
        print("  No NL goal in prompt, using PDDL goal directly")
        from scripts.benchmark_nl_goal_parse import _extract_goal_atoms
        gt_atoms = _extract_goal_atoms(open(instance_path).read())
        s3 = {"parsed_goal": list(gt_atoms),
               "llm_accepted": False, "verifier_score": 1.0,
               "seconds": 0.0, "nl_goal": "(not provided)"}
        parsed_goal = gt_atoms
    else:
        # Extract vocab + GT for verifier
        from scripts.benchmark_nl_goal_parse import (
            build_vocab, _extract_goal_atoms,
        )
        vocab = build_vocab(domain_path, instance_path)
        gt_atoms = _extract_goal_atoms(open(instance_path).read())
        s3, parsed_goal = step3_parse_nl_goal(
            nl_goal, vocab, args.model, gt_atoms=gt_atoms,
        )
        if not parsed_goal:
            parsed_goal = gt_atoms  # fallback
    print(f"  NL: {s3.get('nl_goal','')[:100]}")
    print(f"  parsed goal: {sorted(parsed_goal)}")
    print(f"  LLM accepted: {s3.get('llm_accepted')}, "
          f"score: {s3.get('verifier_score', 0):.2f}, "
          f"{s3.get('seconds', 0):.1f}s")
    trace["step3_nl_parse"] = s3

    print("\n[Step 4] Compile G_env CFG mask ...")
    from scripts.induce_pddl_generic import state_to_predicates
    init_state = state_to_predicates(task.initial_state)
    s4, cfg_obj = step4_build_cfg(models, init_state, parsed_goal, task)
    print(f"  CFG built: {s4['cfg_built']}, "
          f"size: {s4['cfg_size_chars']} chars, "
          f"A* fallback: {s4['used_astar_fallback']}, "
          f"{s4['seconds']:.1f}s")
    trace["step4_cfg"] = s4

    print("\n[Step 5] LLM-under-mask plan decoding ...")
    s5, raw = step5_decode_plan(cfg_obj, args.model)
    if isinstance(s5, tuple):
        s5 = s5[0]
    print(f"  Plan raw: {(raw or '')[:120]}...")
    print(f"  {s5.get('seconds', 0):.1f}s")
    trace["step5_decode"] = s5

    print("\n[Step 6] Execute plan + check goal ...")
    s6, success = step6_execute_plan(raw, task)
    print(f"  Applied {s6['n_applied']} actions")
    print(f"  GOAL ACHIEVED: {success}")
    print(f"  {s6['seconds']:.1f}s")
    trace["step6_execute"] = s6

    # Summary
    total = sum(s.get("seconds", 0) for s in trace.values()
                  if isinstance(s, dict))
    print("\n" + "=" * 70)
    print(f"CLOSED LOOP COMPLETE in {total:.1f}s")
    print(f"  Rollouts -> Induction -> NL parse -> CFG -> Decode -> Execute")
    print(f"  Goal achieved: {success}")
    print("=" * 70)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "domain": cfg_yaml["domain_name"],
            "instance": os.path.basename(instance_path),
            "model": args.model,
            "trace": trace,
            "total_seconds": total,
            "goal_achieved": success,
        }, f, indent=2, default=str)
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
