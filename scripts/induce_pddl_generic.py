"""Induce lifted action models for any PDDL domain from random rollouts.

pyperplan grounds the domain and serves as a black-box executor; random
rollouts give (state, action, args, next_state) transitions, from which
intersection statistics recover per-action pre/add/del templates that
are then compared with the ground-truth schemas. Output: --out JSON.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from pyperplan.pddl.parser import Parser
from pyperplan.grounding import ground


def load_task(domain_path: str, problem_path: str,
               keep_irrelevant: bool = True):
    """Return (parsed_domain, grounded_task).

    With keep_irrelevant (the default) pyperplan keeps operators and
    effects that are not goal-relevant; otherwise the grounder drops add
    effects such as those of `(paint_up _ _ _ black)` when the goal never
    mentions black, which would corrupt the induced add lists.
    """
    parser = Parser(domain_path, problem_path)
    dom = parser.parse_domain()
    prob = parser.parse_problem(dom)
    if keep_irrelevant:
        task = ground(prob, remove_irrelevant_operators=False)
    else:
        task = ground(prob)
    return dom, task


def _normalize_atom(s: str) -> str:
    """Convert pyperplan's '(predicate arg1 arg2)' to predicate(arg1,arg2)."""
    s = s.strip()
    assert s.startswith("(") and s.endswith(")"), s
    inner = s[1:-1]
    parts = inner.split()
    pred = parts[0]
    args = parts[1:]
    if not args:
        return pred
    return f"{pred}({','.join(args)})"


def state_to_predicates(state: frozenset) -> frozenset[str]:
    """Convert a pyperplan state (frozenset of '(pred arg...)' strings) to pred(arg,...) atoms."""
    return frozenset(_normalize_atom(a) for a in state)


def parse_op_name(name: str) -> tuple[str, tuple[str, ...]]:
    """Parse '(action_name arg1 arg2 ...)' -> (action_name, (arg1, ...))."""
    name = name.strip()
    if name.startswith("("):
        name = name[1:].rstrip(")")
    parts = name.split()
    return parts[0], tuple(parts[1:])


def random_rollout_pddl(task, n_steps: int = 100, seed: int = 0):
    """Apply random applicable operators from the initial state; return (state_before, op_name, op_args, state_after) tuples."""
    rng = random.Random(seed)
    state = task.initial_state
    transitions = []
    for _ in range(n_steps):
        applicable = [op for op in task.operators if op.applicable(state)]
        if not applicable:
            break
        op = rng.choice(applicable)
        sb = state_to_predicates(state)
        ns = op.apply(state)
        sa = state_to_predicates(ns)
        action, args = parse_op_name(op.name)
        transitions.append((sb, action, args, sa))
        state = ns
    return transitions


def collect_transitions(
    domain_path: str, problem_paths: list[str],
    steps_per_problem: int = 80, seed_base: int = 0,
):
    all_transitions = []
    for i, pp in enumerate(problem_paths):
        try:
            _dom, task = load_task(domain_path, pp)
        except Exception as e:
            print(f"  skip {pp}: {type(e).__name__}: {e}")
            continue
        trs = random_rollout_pddl(
            task, n_steps=steps_per_problem, seed=seed_base + i,
        )
        all_transitions.extend(trs)
    return all_transitions


def _templ(pred: str, args: tuple[str, ...]) -> str:
    """Replace each action arg in pred with its {a_i} template in one regex pass.

    A single pass means an already-templated token such as '{a1}' is never
    re-matched. When the same object fills several positions, the first
    position's template is used.
    """
    if not args:
        return pred
    arg_to_template = {}
    for i, a in enumerate(args):
        if a not in arg_to_template:
            arg_to_template[a] = f"{{a{i+1}}}"
    pattern = "|".join(re.escape(a) for a in arg_to_template)
    return re.sub(rf"\b({pattern})\b",
                  lambda m: arg_to_template[m.group(1)], pred)


def induce_lifted_models(transitions) -> dict:
    """Induce {action: {pre_pos, eff_add, eff_del, n_examples}} by intersection statistics.

    Predicates are templated with the action's args; only those that
    mention an arg or are nullary are kept. No-op transitions (sb == sa,
    which pyperplan produces when an add effect already holds) are
    dropped, since they would empty the add-effect intersection.
    """
    by_action: dict[str, list] = defaultdict(list)
    for sb, action, args, sa in transitions:
        if sb == sa:
            continue
        by_action[action].append((sb, args, sa))

    def _keep(p: str) -> bool:
        return ("{a" in p) or all(c not in p for c in "()")

    models = {}
    for action, items in by_action.items():
        if not items:
            continue
        pre_sets = []
        sb_t_list = []
        sa_t_list = []
        for sb, args, sa in items:
            pre_t = {_templ(p, args) for p in sb}
            pre_t = {p for p in pre_t if _keep(p)}
            sb_t = {_templ(p, args) for p in sb}
            sb_t = {p for p in sb_t if _keep(p)}
            sa_t = {_templ(p, args) for p in sa}
            sa_t = {p for p in sa_t if _keep(p)}
            pre_sets.append(pre_t)
            sb_t_list.append(sb_t)
            sa_t_list.append(sa_t)
        pre_inter = (frozenset.intersection(*[frozenset(s)
                                              for s in pre_sets])
                     if pre_sets else frozenset())
        n = len(items)
        # Causal effect semantics: p is in eff_add iff no transition has p
        # absent from both state_before and state_after. This captures
        # effects such as miconic's served(p), which only show in the
        # diff when p was not already present.
        candidate_adds = set()
        candidate_dels = set()
        for sb_t, sa_t in zip(sb_t_list, sa_t_list):
            candidate_adds.update(sa_t - sb_t)
            candidate_dels.update(sb_t - sa_t)
        eff_add = set()
        for p in candidate_adds:
            if all((p in sb_t) or (p in sa_t)
                   for sb_t, sa_t in zip(sb_t_list, sa_t_list)):
                eff_add.add(p)
        eff_del = set()
        for p in candidate_dels:
            if all((p not in sb_t) or (p not in sa_t)
                   for sb_t, sa_t in zip(sb_t_list, sa_t_list)):
                eff_del.add(p)
        models[action] = {
            "pre_pos": set(pre_inter),
            "eff_add": eff_add,
            "eff_del": eff_del,
            "n_examples": n,
        }
    return models


def ground_truth_models(dom, filter_static: bool = True) -> dict:
    """Extract lifted preconditions and effects from the parsed PDDL domain.

    Action parameters are mapped to {a1}, {a2}, ... in signature order.
    With filter_static (the default), static type predicates that never
    appear in any add/del effect (e.g. (AIRPLANE ?a)) are dropped from
    preconditions, since they are not part of the observable state.
    """
    dynamic_names: set[str] = set()
    for action in dom.actions.values():
        for e in action.effect.addlist:
            dynamic_names.add(e.name)
        for e in action.effect.dellist:
            dynamic_names.add(e.name)

    out = {}
    for name, action in dom.actions.items():
        params = [p[0] for p in action.signature]
        rename = {p: f"{{a{i+1}}}" for i, p in enumerate(params)}

        def _render(pred) -> str:
            pname = pred.name
            args = [rename.get(var, var) for var, _typ in pred.signature]
            if not args:
                return pname
            return f"{pname}({','.join(args)})"

        def _is_dynamic(pred) -> bool:
            return (not filter_static) or pred.name in dynamic_names

        pre_pos = {_render(c) for c in action.precondition
                   if _is_dynamic(c)}
        eff_add = {_render(e) for e in action.effect.addlist}
        eff_del = {_render(e) for e in action.effect.dellist}
        out[name] = {
            "pre_pos": pre_pos,
            "eff_add": eff_add,
            "eff_del": eff_del,
        }
    return out


def f1(pred_set: set, gt_set: set):
    """Return (precision, recall, F1); two empty sets count as a perfect match."""
    if not pred_set and not gt_set:
        return 1.0, 1.0, 1.0
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f = 2 * p * r / max(p + r, 1e-9) if (p + r) > 0 else 0.0
    return p, r, f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True,
                    help="Path to PDDL domain file.")
    ap.add_argument("--problems_glob", required=True,
                    help="Glob for problem files (e.g. "
                         "'instances/*.pddl').")
    ap.add_argument("--n_problems", type=int, default=10)
    ap.add_argument("--steps_per_problem", type=int, default=80)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import glob
    problem_paths = sorted(glob.glob(args.problems_glob))[:args.n_problems]
    print(f"using {len(problem_paths)} problems", flush=True)

    print("collecting transitions...", flush=True)
    transitions = collect_transitions(
        args.domain, problem_paths,
        steps_per_problem=args.steps_per_problem,
    )
    print(f"  total transitions: {len(transitions)}")

    print("inducing lifted models...", flush=True)
    induced = induce_lifted_models(transitions)
    for action, m in induced.items():
        print(f"  {action} ({m['n_examples']} ex):")
        print(f"    pre_pos: {sorted(m['pre_pos'])}")
        print(f"    eff_add: {sorted(m['eff_add'])}")
        print(f"    eff_del: {sorted(m['eff_del'])}")

    print("\nground-truth from PDDL:", flush=True)
    dom, _task = load_task(args.domain, problem_paths[0])
    gt = ground_truth_models(dom)
    for action, m in gt.items():
        print(f"  {action}:")
        print(f"    pre_pos: {sorted(m['pre_pos'])}")
        print(f"    eff_add: {sorted(m['eff_add'])}")
        print(f"    eff_del: {sorted(m['eff_del'])}")

    print("\n=== COMPARISON ===")
    overall = []
    for action in gt:
        if action not in induced:
            print(f"  {action}: MISSING")
            continue
        ind = induced[action]
        gtm = gt[action]
        pp = f1(ind["pre_pos"], gtm["pre_pos"])
        aa = f1(ind["eff_add"], gtm["eff_add"])
        dd = f1(ind["eff_del"], gtm["eff_del"])
        overall.append((action, pp, aa, dd))
        print(f"  {action}:  pre P={pp[0]:.2f} R={pp[1]:.2f} F1={pp[2]:.2f} | "
              f"add P={aa[0]:.2f} R={aa[1]:.2f} F1={aa[2]:.2f} | "
              f"del P={dd[0]:.2f} R={dd[1]:.2f} F1={dd[2]:.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "n_transitions": len(transitions),
            "induced": {a: {k: sorted(v) if isinstance(v, set) else v
                            for k, v in m.items()}
                        for a, m in induced.items()},
            "ground_truth": {a: {k: sorted(v) if isinstance(v, set)
                                  else v for k, v in m.items()}
                              for a, m in gt.items()},
            "comparison": [{
                "action": a,
                "pre_f1": pp[2], "add_f1": aa[2], "del_f1": dd[2],
            } for (a, pp, aa, dd) in overall],
        }, f, indent=2)
    print(f"\nwrote: {args.out}")


if __name__ == "__main__":
    main()
