"""Run AGCP through PlanBench's prompts/responses JSON files.

For each instance in prompts/<domain>/<task>.json, the instance PDDL is
resolved via the PlanBench YAML config, AGCP runs (rollouts, induction,
G_env compile, XGrammar-constrained decoding), and the plan is written to
`llm_raw_response` in responses/<domain>/<engine_label>/<task>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


def load_planbench_config(plan_bench_root: str, config_name: str) -> dict:
    import yaml
    p = os.path.join(plan_bench_root, "configs", f"{config_name}.yaml")
    return yaml.safe_load(open(p))


def planbench_instance_pddl(plan_bench_root: str, cfg: dict,
                              instance_id: int) -> str:
    instances_root = os.path.join(plan_bench_root, "instances")
    template = cfg["instances_template"]
    return os.path.join(instances_root,
                          cfg["instance_dir"],
                          template.format(instance_id))


def planbench_domain_pddl(plan_bench_root: str, cfg: dict) -> str:
    return os.path.join(plan_bench_root, "instances", cfg["domain_file"])


def agcp_plan_one(domain_file: str, instance_file: str,
                    model_name: str,
                    n_rollouts: int = 10,
                    steps_per_rollout: int = 50,
                    max_new_tokens: int = 384,
                    max_extra: int = 4) -> tuple[str, str]:
    """Run AGCP on one instance, return (plan_text, error_or_empty)."""
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
        goal_atoms.add(_norm_atom(str(atom)))
    objects = sorted({a for atom in (init_state | goal_atoms)
                       for a in _extract_objects(atom)})

    # Build ground_op_filter from pyperplan's typed grounded ops.
    # Eliminates typed-grounding mismatch + high-arity Cartesian
    # explosion. See generic_grammar.compile_from_action_models.
    from scripts.induce_pddl_generic import parse_op_name
    ground_op_filter = {}
    for op in task.operators:
        a_name, a_args = parse_op_name(op.name)
        ground_op_filter.setdefault(a_name, []).append(a_args)

    try:
        cfg_obj = compile_from_action_models(
            models, init_state, goal_atoms,
            objects=objects, max_extra=max_extra,
            ground_op_filter=ground_op_filter,
        )
    except Exception as e:
        cfg_obj = None
    if cfg_obj is None:
        # A* + hAdd fallback for large state-spaces
        try:
            cfg_obj = compile_from_action_models_astar(
                task, init_state, goal_atoms,
            )
        except Exception as e:
            return "", f"compile (BFS+A*): {e}"
        if cfg_obj is None:
            return "", "no plan found by BFS or A*"

    try:
        from agplan.decoding.xgrammar_wrapper import (
            DecodeConfig, XGrammarConstrainedDecoder,
        )
    except Exception as e:
        return "", f"xgrammar import: {e}"

    cfg_text = (cfg_obj.to_ebnf() if hasattr(cfg_obj, "to_ebnf")
                else str(cfg_obj))
    try:
        dec = XGrammarConstrainedDecoder(
            model_name=model_name, ebnf=cfg_text,
        )
    except Exception as e:
        return "", f"decoder init: {e}"
    try:
        plan_text = dec.generate(
            "Generate a plan as a JSON array of actions.",
            cfg=DecodeConfig(max_new_tokens=max_new_tokens),
        )
    except Exception as e:
        return "", f"decode: {e}"
    return plan_text, ""


def _norm_atom(s: str) -> str:
    s = s.strip()
    if s.startswith("(") and s.endswith(")"):
        parts = s[1:-1].split()
        return f"{parts[0]}({','.join(parts[1:])})" if len(parts) > 1 else parts[0]
    return s


def _extract_objects(atom: str) -> list[str]:
    # Predicate names can include hyphens (`pick-up`, `on-table`).
    m = re.match(r"^[\w-]+\((.*)\)$", atom)
    if not m: return []
    return [a.strip() for a in m.group(1).split(",") if a.strip()]


def _llm_json_to_planbench_plan(raw: str) -> str:
    """Convert the JSON plan {"plan": ["pick-up(red)", ...]} to PlanBench's
    LISP form, one "(pick-up red)" line per action."""
    import json as _json
    try:
        obj = _json.loads(raw)
        actions = obj.get("plan", obj if isinstance(obj, list) else [])
    except Exception:
        return raw
    lines = []
    for a in actions:
        m = re.match(r"^([\w-]+)\((.*)\)$", str(a).strip())
        if not m:
            continue
        head = m.group(1)
        args = [x.strip() for x in m.group(2).split(",") if x.strip()]
        if args:
            lines.append(f"({head} {' '.join(args)})")
        else:
            lines.append(f"({head})")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan_bench_root", required=True,
                    help="external/LLMs-Planning/plan-bench")
    ap.add_argument("--config", required=True,
                    help="Domain config name, e.g. mystery_blocksworld_3")
    ap.add_argument("--task", required=True,
                    help="t1, t4, t5, t6, t7, t8_1, t8_2, t8_3")
    ap.add_argument("--engine_label", required=True,
                    help="Subfolder name under responses/<domain>/")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n_rollouts", type=int, default=10)
    ap.add_argument("--max_instances", type=int, default=50)
    ap.add_argument("--specific_instances", nargs="+", type=int,
                    default=None)
    args = ap.parse_args()

    cfg = load_planbench_config(args.plan_bench_root, args.config)
    domain_name = cfg["domain_name"]
    domain_file = planbench_domain_pddl(args.plan_bench_root, cfg)
    print(f"Domain: {domain_name}  domain_file={domain_file}")

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
    prompts_json = os.path.join(args.plan_bench_root, "prompts",
                                  domain_name, f"{task_name}.json")
    if not os.path.exists(prompts_json):
        print(f"ERROR: no prompts JSON at {prompts_json}")
        print("Run prompt_generation.py for this (domain, task) first.")
        return

    with open(prompts_json) as f:
        structured = json.load(f)
    structured["engine"] = args.engine_label

    n_done = 0
    n_attempted = 0
    n_errs = 0
    for instance in structured["instances"]:
        if args.specific_instances and \
            instance["instance_id"] not in args.specific_instances:
            continue
        if n_done >= args.max_instances:
            break
        n_attempted += 1
        instance_pddl = planbench_instance_pddl(
            args.plan_bench_root, cfg, instance["instance_id"])
        if not os.path.exists(instance_pddl):
            print(f"  miss: {instance_pddl}")
            n_errs += 1
            continue
        print(f"  inst {instance['instance_id']}: "
              f"{os.path.basename(instance_pddl)}...", end=" ")
        # Hard per-instance timeout (300 s) against BFS blow-ups or model hangs.
        import signal
        def _timeout_handler(signum, frame):
            raise TimeoutError("per-instance timeout")
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(300)
        try:
            plan_text, err = agcp_plan_one(
                domain_file, instance_pddl,
                model_name=args.model,
                n_rollouts=args.n_rollouts,
            )
        except TimeoutError:
            plan_text, err = "", "timeout (300s)"
        except Exception as e:
            plan_text, err = "", f"crash: {type(e).__name__}: {e}"
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        if err:
            print(f"ERR: {err[:60]}")
            n_errs += 1
            instance["llm_raw_response"] = ""
            instance["agcp_error"] = err
        else:
            planbench_plan = _llm_json_to_planbench_plan(plan_text)
            instance["llm_raw_response"] = planbench_plan
            instance["agcp_raw_plan"] = plan_text
            n_lines = len(planbench_plan.splitlines())
            print(f"plan ({n_lines} actions)")
            n_done += 1

    out_dir = os.path.join(args.plan_bench_root, "responses",
                             domain_name, args.engine_label)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_json = os.path.join(out_dir, f"{task_name}.json")
    with open(out_json, "w") as f:
        json.dump(structured, f, indent=2)

    print(f"\n{n_done}/{n_attempted} instances planned ({n_errs} errors)")
    print(f"wrote: {out_json}")
    print(f"\nTo score: cd {args.plan_bench_root} && \\")
    print(f"  python response_evaluation.py --task {args.task} \\")
    print(f"    --config {args.config} --engine {args.engine_label}")


if __name__ == "__main__":
    main()
