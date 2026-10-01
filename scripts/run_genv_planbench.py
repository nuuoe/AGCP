"""PlanBench Blocksworld evaluation under the G_env mask.

For each PlanBench instance, parse the PDDL into (labels, initial support,
goal pairs), compile the G_env grammar (partial-goal semantics), sample K
plans under the XGrammar mask, and execute them on BlocksworldEnv. Writes
one JSONL record per instance to --out; --mask syntax runs the coarse-grammar arm.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from agplan.decoding.xgrammar_wrapper import DecodeConfig, XGrammarConstrainedDecoder
from agplan.grammars.blocksworld_grammar import compile_blocksworld_grammar
from agplan.planning.blocksworld import BlocksworldEnv
from agplan.planning.planbench_blocksworld import (
    load_planbench_instances, MYSTERY_ACTION_NAMES,
)
from scripts.run_adversarial_robustness import coarse_mask_ebnf


# Reverse map for executing Mystery-style action strings.
MYSTERY_TO_BLOCKSWORLD = {v: k for k, v in MYSTERY_ACTION_NAMES.items()}


def _execute(plan_actions: list[str], env: BlocksworldEnv) -> dict:
    steps = 0
    for raw in plan_actions:
        m = re.match(r"(\w+)\(([^)]*)\)", raw)
        if not m:
            return {"success": False, "error": f"unparseable: {raw}",
                    "steps_executed": steps}
        op = m.group(1)
        # Translate Mystery action names back to Blocksworld for execution.
        op = MYSTERY_TO_BLOCKSWORLD.get(op, op)
        args = [a.strip() for a in m.group(2).split(",") if a.strip()]
        if op in ("pickup", "putdown"):
            arg1 = args[0] if args else None
            obs, reward, term, trunc, info = env.step(op, arg1, None)
        elif op in ("unstack", "stack"):
            arg1 = args[0] if len(args) >= 1 else None
            arg2 = args[1] if len(args) >= 2 else None
            obs, reward, term, trunc, info = env.step(op, arg1, arg2)
        else:
            return {"success": False, "error": f"unknown action: {op}",
                    "steps_executed": steps}
        steps += 1
        if "error" in info:
            return {"success": False, "error": info["error"],
                    "steps_executed": steps}
        if term:
            return {"success": True, "steps_executed": steps}
        if trunc:
            return {"success": False, "error": "truncated",
                    "steps_executed": steps}
    return {"success": False, "error": "plan exhausted without goal",
            "steps_executed": steps}


def _parse_plan_text(raw_json: str) -> tuple[bool, list[str], str]:
    try:
        d = json.loads(raw_json)
        return True, list(d.get("plan", [])), ""
    except Exception as e:
        return False, [], str(e)


def _render_state(env: BlocksworldEnv) -> str:
    return env.render_text()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances_dir", required=True,
                    help="Directory containing PlanBench instance-*.pddl")
    ap.add_argument("--domain", default="blocksworld",
                    choices=["blocksworld", "mystery"],
                    help="PDDL domain. mystery uses obfuscated names.")
    ap.add_argument("--n_instances", type=int, default=32)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--max_extra", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--mask", default="genv", choices=["genv", "syntax"],
                    help="genv = reachability-intersected G_env (paper); "
                         "syntax = coarse syntax-only action grammar "
                         "(isolation arm: same prompt, same decode, no "
                         "state-dependent applicability)")
    ap.add_argument("--device", default=None,
                    help="torch device override (e.g. mps, cpu)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    instances = load_planbench_instances(Path(args.instances_dir),
                                          limit=args.n_instances,
                                          domain=args.domain)
    print(f"loaded {len(instances)} PlanBench instances "
          f"(domain={args.domain})", flush=True)
    action_name_map = MYSTERY_ACTION_NAMES if args.domain == "mystery" else None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    placeholder = 'root ::= "{\\"plan\\":[\\"pickup(a)\\"]}"\n'
    decoder = XGrammarConstrainedDecoder(model_name=args.model, ebnf=placeholder,
                                         device=args.device)

    template = (
        "You are a Blocksworld planning agent. Initial state:\n"
        "```\n{state}\n```\n"
        "Goal: {goal_desc}\n\n"
        "Output a JSON plan as {{\"plan\": [\"action(arg)\", ...]}}. "
        "Valid actions: pickup(b), putdown(b), unstack(b,c), stack(b,c). "
        "The grammar provided will admit only plans that satisfy the goal."
    )

    f_out = open(out_path, "w")
    for inst in instances:
        env = BlocksworldEnv()
        env.reset_from(
            inst["labels"], inst["initial_support"],
            inst["goal_pairs"], inst.get("held"), partial_goal=True,
        )
        if args.mask == "syntax":
            ebnf = coarse_mask_ebnf()
        else:
            ebnf = compile_blocksworld_grammar(
                inst["initial_support"], inst["goal_pairs"], inst["labels"],
                max_extra=args.max_extra, strict_goal=False,
                action_name_map=action_name_map,
            )
        if ebnf is None:
            rec = {"instance_id": inst["instance_id"],
                   "grammar_compiled": False,
                   "reason": "no goal-reaching path within budget",
                   "initial": inst["initial_support"],
                   "goal": list(inst["goal_pairs"])}
            f_out.write(json.dumps(rec) + "\n"); f_out.flush()
            continue

        try:
            decoder.recompile_ebnf(ebnf)
        except Exception as e:
            rec = {"instance_id": inst["instance_id"],
                   "grammar_compiled": False,
                   "grammar_size": len(ebnf),
                   "reason": f"xgrammar compile failed: {type(e).__name__}: {e}"}
            f_out.write(json.dumps(rec) + "\n"); f_out.flush()
            continue

        state_text = _render_state(env)
        goal_desc = ", ".join(f"{b} on {sup}" for b, sup in inst["goal_pairs"])
        prompt = template.format(state=state_text, goal_desc=goal_desc)

        cfg = DecodeConfig(do_sample=True, temperature=args.temperature,
                           top_p=args.top_p,
                           max_new_tokens=args.max_new_tokens)
        t0 = time.time()
        try:
            outs = decoder.generate_n(user_prompt=prompt, n=args.k, cfg=cfg)
        except Exception as e:
            rec = {"instance_id": inst["instance_id"],
                   "grammar_compiled": True, "grammar_size": len(ebnf),
                   "reason": f"decode failed: {type(e).__name__}: {e}"}
            f_out.write(json.dumps(rec) + "\n"); f_out.flush()
            continue
        wall = time.time() - t0

        any_success = False
        first_success_idx = -1
        cand_results = []
        for k_idx, raw in enumerate(outs):
            ok, actions, err = _parse_plan_text(raw)
            if not ok:
                cand_results.append({"k": k_idx, "parse_ok": False,
                                     "reason": err})
                continue
            run_env = BlocksworldEnv()
            run_env.reset_from(
                inst["labels"], inst["initial_support"],
                inst["goal_pairs"], inst.get("held"), partial_goal=True,
            )
            ex = _execute(actions, run_env)
            cand_results.append({"k": k_idx, "parse_ok": True,
                                 **ex, "n_actions": len(actions),
                                 "actions": actions if ex.get("success") else None})
            if ex.get("success") and not any_success:
                any_success = True
                first_success_idx = k_idx

        rec = {
            "instance_id": inst["instance_id"],
            "model_id": args.model,
            "mask": args.mask,
            "grammar_compiled": True, "grammar_size": len(ebnf),
            "k": args.k,
            "any_success": any_success,
            "first_success_idx": first_success_idx,
            "wall_seconds": wall,
            "initial": inst["initial_support"],
            "goal": list(inst["goal_pairs"]),
            "candidates": cand_results,
        }
        f_out.write(json.dumps(rec) + "\n"); f_out.flush()
        succ_str = "T" if any_success else "F"
        print(f"{inst['instance_id']}  any_success={succ_str}  "
              f"k_first={first_success_idx}  wall={wall:.1f}s  "
              f"ebnf={len(ebnf)}", flush=True)

    f_out.close()
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
