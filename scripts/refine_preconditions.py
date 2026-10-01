"""Precondition refinement with the environment oracle.

Intersection-based induction keeps atoms that always co-occur with the
true precondition but are not required. Each candidate atom is removed
from an observed pre-state and applicability is re-tested via pyperplan's
grounded operators; droppable atoms are removed. Writes models and a log to --out.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from scripts.induce_pddl_generic import (
    load_task, random_rollout_pddl, induce_lifted_models,
    ground_truth_models, f1, _normalize_atom,
)
from scripts.grounding import ground_lifted


def _normalize_op_pre(op_pre_set) -> set:
    """Normalise pyperplan `(pred a b)` precondition strings to `pred(a,b)`."""
    return {_normalize_atom(p) for p in op_pre_set}


def find_grounded_op(task, action_name: str, args: tuple):
    """Return the grounded pyperplan operator with this name+args, or None."""
    target_prefix = f"({action_name}"
    target = f"({action_name} {' '.join(args)})"
    for op in task.operators:
        if op.name == target:
            return op
    return None


def refine_preconditions(task, action: str, ind_pre: set,
                          rollout_args_list: list,
                          ) -> tuple[set, list]:
    """Drop each atom of ind_pre whose removal leaves the action applicable
    in some observed (s_before, args) context.

    Returns (refined_pre, dropped_atoms_log).
    """
    refined = set(ind_pre)
    dropped = []
    for p_tmpl in sorted(ind_pre):
        can_drop = False
        first_evidence = None
        for sb, args in rollout_args_list:
            grounded = ground_lifted({p_tmpl}, args)
            if not grounded:
                continue
            grounded_atom = next(iter(grounded))
            if grounded_atom not in sb:
                continue  # no evidence from this pre-state
            sb_minus = sb - {grounded_atom}
            op = find_grounded_op(task, action, args)
            if op is None:
                continue
            # op.preconditions are `(pred arg ..)` strings; sb uses `pred(arg,..)`.
            op_pre = _normalize_op_pre(op.preconditions)
            if op_pre <= sb_minus:
                can_drop = True
                first_evidence = (sb_minus, args, grounded_atom)
                break
        if can_drop:
            refined.discard(p_tmpl)
            dropped.append({
                "atom": p_tmpl,
                "evidence_args": list(first_evidence[1]),
                "grounded_atom_dropped": first_evidence[2],
            })
    return refined, dropped


def run_with_refinement(dom_path: str, prob_paths: list[str],
                          n_rollouts: int = 5,
                          steps_per_rollout: int = 200,
                          ) -> tuple[dict, dict]:
    """Induce + refine preconditions. Returns (refined_models,
    pre_refinement_log)."""
    trs = []
    args_by_action: dict[str, list] = {}
    tasks_by_problem: dict[str, object] = {}
    for i, p in enumerate(prob_paths):
        try:
            _, task = load_task(dom_path, p)
        except Exception:
            continue
        tasks_by_problem[p] = task
        for s in range(n_rollouts):
            for tr in random_rollout_pddl(task, n_steps=steps_per_rollout,
                                            seed=i * 100 + s):
                trs.append(tr)
                sb, action, args, sa = tr
                args_by_action.setdefault(action, []).append((sb, args, p))
    trs_ok = [t for t in trs if t[0] != t[-1]]
    models = induce_lifted_models(trs_ok)
    refined = {}
    refinement_log = {}
    for action, model in models.items():
        ind_pre = set(model["pre_pos"])
        rollout_args = []
        for sb, args, prob in args_by_action.get(action, []):
            task = tasks_by_problem[prob]
            rollout_args.append((sb, args))
            if len(rollout_args) > 100: break
        # Grounded operators are per problem: refine against the first
        # three problems in turn.
        new_pre = set(ind_pre)
        dropped_all = []
        for prob in list(tasks_by_problem.keys())[:3]:
            task = tasks_by_problem[prob]
            rargs_for_task = [(sb, args) for sb, args, p
                               in args_by_action.get(action, [])
                               if p == prob][:30]
            if not rargs_for_task: continue
            attempted, dropped = refine_preconditions(
                task, action, new_pre, rargs_for_task,
            )
            new_pre = attempted
            dropped_all.extend(dropped)
        refined[action] = {
            "pre_pos": new_pre,
            "eff_add": model["eff_add"],
            "eff_del": model["eff_del"],
        }
        refinement_log[action] = {
            "induced_pre": sorted(ind_pre),
            "refined_pre": sorted(new_pre),
            "dropped": dropped_all,
        }
    return refined, refinement_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--n_problems", type=int, default=5)
    ap.add_argument("--n_rollouts", type=int, default=10)
    ap.add_argument("--steps_per_rollout", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    probs = sorted(glob.glob(args.problems))[:args.n_problems]
    refined, log = run_with_refinement(args.domain, probs,
                                          n_rollouts=args.n_rollouts,
                                          steps_per_rollout=args.steps_per_rollout)
    dom, _ = load_task(args.domain, probs[0])
    gt = ground_truth_models(dom)
    print(f"Domain: {args.domain}")
    print("Per-action precondition F1 before and after refinement")
    print("-" * 70)
    for action in sorted(gt):
        if action not in refined: continue
        ind_pre = set(log[action]["induced_pre"])
        ref_pre = set(log[action]["refined_pre"])
        gt_pre = set(gt[action]["pre_pos"])
        ind_f1 = f1(ind_pre, gt_pre)[2]
        ref_f1 = f1(ref_pre, gt_pre)[2]
        dropped = [d["atom"] for d in log[action]["dropped"]]
        print(f"  {action:25s}  pre F1 {ind_f1:.2f} -> {ref_f1:.2f}  dropped: {dropped}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "refined": {a: {k: sorted(v) if isinstance(v, set) else v
                             for k, v in m.items()}
                         for a, m in refined.items()},
            "refinement_log": log,
        }, f, indent=2, default=str)
    print(f"\nwrote: {args.out}")


if __name__ == "__main__":
    main()
