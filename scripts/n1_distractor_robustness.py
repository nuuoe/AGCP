"""Distractor robustness test for NL-goal parsing on Blocksworld.

Each PlanBench NL goal is wrapped with irrelevant or misleading sentences
(see DISTRACTORS) and re-parsed with the verifier-gated LLM parse.
Writes per-variant accept counts and per-instance rows to --out.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


DISTRACTORS = {
    "orig":         (lambda nl: nl),
    "prefix_easy":  (lambda nl: f"Note: the kitchen is clean. {nl}"),
    "prefix_hard":  (lambda nl: f"Some people prefer that the blue block is on top of the orange block. However, {nl}"),
    "suffix":       (lambda nl: f"{nl} Also, the weather is nice today."),
    "sandwich":     (lambda nl: f"Some people prefer that the blue block is on top of the orange block. However, {nl} Also, the weather is nice today."),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan_bench_root", required=True)
    ap.add_argument("--n_instances", type=int, default=10)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from scripts.benchmark_nl_goal_parse import (
        extract_nl_goal, build_vocab, _extract_goal_atoms,
        build_prompt, verifier_factory, map_colors_to_letters,
        GOAL_SCHEMA,
    )
    from agplan.llm_propose import propose_verify_fallback

    pb_root = args.plan_bench_root
    cfg = "blocksworld_3"
    dom_path = os.path.join(pb_root, "instances",
                            "blocksworld/generated_domain.pddl")
    prompts = json.load(open(os.path.join(
        pb_root, "prompts", "blocksworld_3",
        "task_1_plan_generation.json")))

    rows = []
    for inst_data in prompts["instances"][:args.n_instances]:
        inst_id = inst_data["instance_id"]
        nl_orig = extract_nl_goal(inst_data["query"])
        if not nl_orig: continue
        instance_pddl = os.path.join(
            pb_root, "instances",
            "blocksworld/generated_basic_3",
            f"instance-{inst_id}.pddl")
        if not os.path.exists(instance_pddl): continue
        vocab = build_vocab(dom_path, instance_pddl)
        gt_atoms = _extract_goal_atoms(open(instance_pddl).read())

        row = {"inst": inst_id, "nl_orig": nl_orig,
                "gt_atoms": sorted(gt_atoms),
                "results": {}}

        for tag, fn in DISTRACTORS.items():
            nl_d = fn(nl_orig)
            nl_d_aliased = map_colors_to_letters(nl_d, vocab["objects"])
            prompt = build_prompt(nl_d_aliased, vocab)
            verifier = verifier_factory(gt_atoms, vocab)
            baseline = {"goal": list(gt_atoms)}
            final, info = propose_verify_fallback(
                prompt=prompt, schema_json=GOAL_SCHEMA,
                verifier=verifier, baseline=baseline,
                threshold=0.9, model_name=args.model,
                max_new_tokens=384, n_samples=4,
                temperature=0.4,
            )
            row["results"][tag] = {
                "distractor_nl": nl_d,
                "llm_accepted": info["accepted_llm"],
                "accuracy": info["accuracy"],
                "proposal": info["proposal"].get("goal")
                            if isinstance(info["proposal"], dict)
                            else None,
            }
            print(f"  inst {inst_id} {tag:13s}: "
                   f"{'OK' if info['accepted_llm'] else 'fail'} "
                   f"F1={info['accuracy']:.2f}")
        rows.append(row)

    per_tag = {tag: 0 for tag in DISTRACTORS}
    for r in rows:
        for tag in DISTRACTORS:
            if r["results"][tag]["llm_accepted"]: per_tag[tag] += 1

    n = len(rows)
    print(f"\n=== Distractor robustness ({n} BW instances) ===")
    for tag in DISTRACTORS:
        print(f"  {tag:13s}  {per_tag[tag]}/{n} = "
                f"{100*per_tag[tag]/n:.1f}%")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "model": args.model, "n_instances": n,
            "per_tag": per_tag,
            "rows": rows,
        }, f, indent=2)
    print(f"\nwrote: {args.out}")


if __name__ == "__main__":
    main()
