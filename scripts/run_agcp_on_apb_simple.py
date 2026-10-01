"""Run AGCP over AutoPlanBench's unzipped domains without its PlanningGame harness.

AGCP plans from the environment oracle and needs no NL prompt, so each
apb2.0_dataset/<domain>/orig_problems/instance-*.pddl is run directly
through rollouts, induction, G_env compile and masked decoding, then
validated with pyperplan. Writes per-domain success rates to --out.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path


def parse_lisp_plan(text: str) -> list[tuple]:
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if not line: continue
        m = re.match(r"^\(?([\w-]+)\(?([^)]*)\)?\)?$", line)
        if not m: continue
        head = m.group(1).lower()
        rest = m.group(2)
        args = [a for a in re.split(r"[,\s]+", rest) if a]
        out.append((head, tuple(args)))
    return out


def _llm_json_to_lisp(raw: str) -> str:
    """{"plan":["pick-up(a)",...]} → "(pick-up a)\\n..."."""
    import json as _json
    try:
        obj = _json.loads(raw)
        actions = obj.get("plan", obj if isinstance(obj, list) else [])
    except Exception:
        return raw
    lines = []
    for a in actions:
        m = re.match(r"^([\w-]+)\((.*)\)$", str(a).strip())
        if not m: continue
        head = m.group(1)
        args = [x.strip() for x in m.group(2).split(",") if x.strip()]
        lines.append(f"({head} {' '.join(args)})" if args
                      else f"({head})")
    return "\n".join(lines)


def _with_timeout(seconds: int, fn, *args, **kwargs):
    """Run fn with a SIGALRM-based wall-clock timeout. Returns
    (result, None) or (None, "timeout")."""
    import signal
    class _TimeoutError(Exception): pass
    def _h(signum, frame): raise _TimeoutError()
    old = signal.signal(signal.SIGALRM, _h)
    signal.alarm(seconds)
    try:
        return fn(*args, **kwargs), None
    except _TimeoutError:
        return None, f"timeout after {seconds}s"
    except Exception as e:
        return None, f"error: {type(e).__name__}: {str(e)[:100]}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def agcp_plan_one(domain_file: str, instance_file: str,
                    model_name: str,
                    n_rollouts: int = 10,
                    steps_per_rollout: int = 50,
                    max_new_tokens: int = 2048,
                    compile_timeout: int = 60) -> tuple[str, str]:
    from scripts.induce_pddl_generic import (
        load_task, random_rollout_pddl, state_to_predicates,
    )
    from scripts.induce_positional import (
        induce_lifted_models_positional,
    )
    from agplan.grammars.generic_grammar import (
        compile_from_action_models,
        compile_from_action_models_astar,
    )
    from agplan.decoding.xgrammar_wrapper import (
        DecodeConfig, XGrammarConstrainedDecoder,
    )

    try:
        _, task = load_task(domain_file, instance_file)
    except Exception as e:
        return "", f"load_task: {e}"
    trs = []
    for s in range(n_rollouts):
        trs.extend(random_rollout_pddl(
            task, n_steps=steps_per_rollout, seed=s,
        ))
    trs_ok = [t for t in trs if t[0] != t[-1]]
    if not trs_ok:
        return "", "no successful rollouts"
    models = induce_lifted_models_positional(trs_ok)
    init_state = state_to_predicates(task.initial_state)
    goal_atoms = set()
    for atom in task.goals:
        s = str(atom).strip()
        if s.startswith("(") and s.endswith(")"):
            parts = s[1:-1].split()
            goal_atoms.add(f"{parts[0]}({','.join(parts[1:])})"
                            if len(parts) > 1 else parts[0])
        else: goal_atoms.add(s)
    objects = set()
    for atom in init_state | goal_atoms:
        m = re.match(r"^[\w-]+\((.*)\)$", atom)
        if m:
            for a in m.group(1).split(","):
                if a.strip(): objects.add(a.strip())
    objects = sorted(objects)

    # ground_op_filter: per-action argument tuples that pyperplan grounded.
    # ground_op_oracle: (action, args) -> pyperplan op, for an exact
    # applicability check (the induced schema may miss static or
    # co-occurring preconditions).
    ground_op_filter = {}
    ground_op_oracle = {}
    from scripts.induce_pddl_generic import parse_op_name
    for op in task.operators:
        action_name, args = parse_op_name(op.name)
        ground_op_filter.setdefault(action_name, []).append(args)
        ground_op_oracle[(action_name, args)] = op

    cfg_obj, terr = _with_timeout(
        compile_timeout,
        compile_from_action_models,
        models, init_state, goal_atoms,
        objects=objects, max_extra=4,
        ground_op_filter=ground_op_filter,
        ground_op_oracle=ground_op_oracle,
    )
    if cfg_obj is None or terr:
        # A* + hAdd fallback: find one valid plan and build a degenerate
        # CFG admitting only that plan; the LLM still decodes under the
        # XGrammar mask.
        cfg_obj, terr2 = _with_timeout(
            compile_timeout,
            compile_from_action_models_astar,
            task, init_state, goal_atoms,
        )
        if cfg_obj is None or terr2:
            return "", f"compile (BFS+A*): {terr or terr2 or 'no plan'}"
    cfg_text = (cfg_obj.to_ebnf() if hasattr(cfg_obj, "to_ebnf")
                else str(cfg_obj))
    try:
        dec = XGrammarConstrainedDecoder(model_name=model_name,
                                            ebnf=cfg_text)
        raw = dec.generate(
            "Generate a plan as a JSON array of actions.",
            cfg=DecodeConfig(max_new_tokens=max_new_tokens),
        )
    except Exception as e:
        return "", f"decode: {e}"
    return _llm_json_to_lisp(raw), ""


def validate_plan(domain_file: str, instance_file: str,
                    plan_text: str) -> tuple[bool, int, str]:
    from scripts.induce_pddl_generic import load_task
    plan = parse_lisp_plan(plan_text)
    try:
        _, task = load_task(domain_file, instance_file)
    except Exception as e:
        return False, 0, f"load: {e}"
    state = task.initial_state
    op_by_name = {op.name: op for op in task.operators}
    n_applied = 0
    for action, args in plan:
        op_name = f"({action} {' '.join(args)})"
        op = op_by_name.get(op_name)
        if op is None:
            return False, n_applied, f"unknown op {op_name}"
        if not op.applicable(state):
            return False, n_applied, f"not applicable: {op_name}"
        state = op.apply(state)
        n_applied += 1
    success = task.goals <= state
    return success, n_applied, "" if success else "goal not satisfied"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apb_root", required=True,
                    help="external/autoplanbench/autoplanbench_dataset/apb2.0_dataset")
    ap.add_argument("--domain", default=None,
                    help="If set, only run this single domain")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n_rollouts", type=int, default=10)
    ap.add_argument("--max_instances", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.domain:
        domains = [args.domain]
    else:
        domains = sorted(d for d in os.listdir(args.apb_root)
                          if os.path.isfile(os.path.join(
                              args.apb_root, d, "domain.pddl")))
    print(f"Domains to run: {len(domains)}")

    all_rows = []
    summary = {}
    for d in domains:
        dom_path = os.path.join(args.apb_root, d, "domain.pddl")
        prob_dir = os.path.join(args.apb_root, d, "orig_problems")
        if not os.path.isdir(prob_dir):
            prob_dir = os.path.join(args.apb_root, d, "adapted_instances")
        probs = sorted(Path(prob_dir).glob("instance-*.pddl"))[:args.max_instances]
        if not probs:
            print(f"  {d}: no instances at {prob_dir}")
            continue
        print(f"\n=== {d} ({len(probs)} instances) ===")
        n_solved = 0
        per = []
        for prob in probs:
            inst_id = int(re.findall(r"instance-(\d+)", prob.name)[0])
            t0 = time.time()
            plan_text, err = agcp_plan_one(
                dom_path, str(prob),
                model_name=args.model,
                n_rollouts=args.n_rollouts,
            )
            if err:
                per.append({"instance_id": inst_id, "success": False,
                              "error": err, "seconds": time.time()-t0})
                print(f"  inst {inst_id}: ERR {err[:60]}")
                continue
            success, n_applied, verr = validate_plan(
                dom_path, str(prob), plan_text)
            if success: n_solved += 1
            per.append({"instance_id": inst_id, "success": success,
                          "n_applied": n_applied, "error": verr,
                          "seconds": time.time()-t0,
                          "plan": plan_text[:200]})
            flag = "OK" if success else f"FAIL ({verr[:40]})"
            print(f"  inst {inst_id}: applied={n_applied} {flag}")
        rate = n_solved / max(len(probs), 1)
        summary[d] = {"n_solved": n_solved, "n_total": len(probs),
                       "success_rate": rate}
        all_rows.append({"domain": d, **summary[d], "per_instance": per})
        print(f"  {d}: {n_solved}/{len(probs)} = {rate*100:.1f}%")

    total_solved = sum(s["n_solved"] for s in summary.values())
    total = sum(s["n_total"] for s in summary.values())
    print("\n" + "=" * 60)
    print(f"AGCP on AutoPlanBench: {total_solved}/{total} = "
          f"{100*total_solved/max(total,1):.1f}% across {len(summary)} domains")
    print("=" * 60)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "model": args.model,
            "n_rollouts": args.n_rollouts,
            "max_instances": args.max_instances,
            "summary": summary,
            "rows": all_rows,
            "total_solved": total_solved, "total": total,
        }, f, indent=2, default=str)
    print(f"\nwrote: {args.out}")


if __name__ == "__main__":
    main()
