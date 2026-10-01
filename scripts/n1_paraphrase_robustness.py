"""Paraphrase robustness test for NL-goal parsing on Blocksworld.

Each PlanBench NL goal is rewritten by three meaning-preserving
paraphrasers (PARAPHRASE_FNS) and re-parsed; reports per-paraphrase
accept counts and per-instance consistency. Writes rows to --out.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


# Three paraphrase rewriters that preserve meaning but vary surface form.
def paraphrase_v1(nl: str) -> str:
    """Rewrite "X is on top of Y" as "Y has X on top of it"."""
    return re.sub(
        r"the (\w+) block is on top of the (\w+) block",
        r"the \2 block has the \1 block on top of it",
        nl,
    )


def paraphrase_v2(nl: str) -> str:
    """Replace "is on top of" with "is stacked above"."""
    return nl.replace(" is on top of ", " is stacked above ")


def paraphrase_v3(nl: str) -> str:
    """Replace "the X block" with "block X" and put a comma before each "and"."""
    s = re.sub(r"the (\w+) block", r"block \1", nl)
    s = re.sub(r" and ", ", and ", s)
    return s


PARAPHRASE_FNS = [
    ("orig", lambda x: x),
    ("v1_swap", paraphrase_v1),
    ("v2_lex", paraphrase_v2),
    ("v3_dropthe", paraphrase_v3),
]


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

        for tag, fn in PARAPHRASE_FNS:
            nl_p = fn(nl_orig)
            nl_p_aliased = map_colors_to_letters(nl_p, vocab["objects"])
            prompt = build_prompt(nl_p_aliased, vocab)
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
                "paraphrase": nl_p,
                "llm_accepted": info["accepted_llm"],
                "accuracy": info["accuracy"],
                "proposal": info["proposal"].get("goal")
                            if isinstance(info["proposal"], dict)
                            else None,
            }
            print(f"  inst {inst_id} {tag:12s}: "
                   f"{'OK' if info['accepted_llm'] else 'fail'} "
                   f"F1={info['accuracy']:.2f}")
        rows.append(row)

    per_tag = {tag: 0 for tag, _ in PARAPHRASE_FNS}
    consistent = 0  # all 4 versions agree (all-OK or all-fail)
    all_ok = 0      # all 4 versions OK
    for r in rows:
        results = [r["results"][tag]["llm_accepted"]
                    for tag, _ in PARAPHRASE_FNS]
        for tag, _ in PARAPHRASE_FNS:
            if r["results"][tag]["llm_accepted"]: per_tag[tag] += 1
        if all(results): all_ok += 1
        if all(results) or not any(results): consistent += 1

    n = len(rows)
    print(f"\n=== Paraphrase robustness ({n} BW instances) ===")
    for tag, _ in PARAPHRASE_FNS:
        print(f"  {tag:12s}  {per_tag[tag]}/{n} = "
                f"{100*per_tag[tag]/n:.1f}%")
    print(f"  ALL-4-OK     {all_ok}/{n} = {100*all_ok/n:.1f}%")
    print(f"  consistent   {consistent}/{n} = {100*consistent/n:.1f}%")
    print(f"     (agree across paraphrases: all-pass or all-fail)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "model": args.model, "n_instances": n,
            "per_tag": per_tag, "all_ok": all_ok,
            "consistent": consistent,
            "rows": rows,
        }, f, indent=2)
    print(f"\nwrote: {args.out}")


if __name__ == "__main__":
    main()
