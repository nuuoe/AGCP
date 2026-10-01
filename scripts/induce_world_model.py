"""Induce Blocksworld action models from random env rollouts.

Collects (state, action, next_state) transitions from BlocksworldEnv
(optionally renamed to the Mystery vocabulary), induces per-action
pre/add/del templates by intersection statistics and scores them
against hand-written ground truth. Output: --out JSON.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from agplan.planning.blocksworld import BlocksworldEnv


_DOMAIN_MAPS = {
    "blocksworld": None,
    "mystery": {
        "handempty": "harmony",
        "clear": "province",
        "ontable": "planet",
        "on": "craves",
        "holding": "pain",
    },
}


def state_predicates(env: BlocksworldEnv,
                     domain: str = "blocksworld") -> frozenset[str]:
    """Render the Blocksworld state as a predicate set, optionally renamed to the Mystery vocabulary."""
    pmap = _DOMAIN_MAPS.get(domain) or {}
    def _r(name: str) -> str:
        return pmap.get(name, name)
    out: set[str] = set()
    labels = list(env.labels)
    if env.held is None:
        out.add(_r("handempty"))
    else:
        out.add(f"{_r('holding')}({env.held})")
    for b in labels:
        sup = env.support.get(b)
        if sup is None:
            continue
        if sup == "TABLE":
            out.add(f"{_r('ontable')}({b})")
        else:
            out.add(f"{_r('on')}({b},{sup})")
    on_top = {sup for b, sup in env.support.items() if sup != "TABLE"}
    for b in labels:
        if b == env.held:
            continue
        if b not in on_top and b in env.support:
            out.add(f"{_r('clear')}({b})")
    return frozenset(out)


_ACTION_RENAME = {
    "blocksworld": None,
    "mystery": {
        "pickup": "attack", "putdown": "succumb",
        "stack": "overcome", "unstack": "feast",
    },
}


@dataclass
class Transition:
    state_before: frozenset[str]
    action: tuple[str, ...]   # ("pickup", "a") etc.
    state_after: Optional[frozenset[str]]  # None on failure
    error: Optional[str]


def random_rollout(
    seed: int, labels: list[str], steps: int = 50,
    domain: str = "blocksworld",
) -> list[Transition]:
    """Apply random actions from a random initial state, logging successful and failed attempts."""
    rng = random.Random(seed)
    env = BlocksworldEnv(n_blocks=len(labels))
    env.reset(seed=seed)
    # Lowercase labels, as in PlanBench.
    env._labels = list(labels)
    sup: dict[str, str] = {}
    order = list(labels); rng.shuffle(order)
    for b in order:
        choices = ["TABLE"] + [
            x for x in sup if all(s != x for s in sup.values())
        ]
        sup[b] = rng.choice(choices)
    env.reset_from(labels, sup, (), held=None, partial_goal=True)

    transitions: list[Transition] = []
    actions_specs = [
        ("pickup", 1), ("putdown", 1),
        ("unstack", 2), ("stack", 2),
    ]
    for _ in range(steps):
        op, arity = rng.choice(actions_specs)
        # putdown and stack act on the held block regardless of arg1, so
        # a1 is set to the held block to keep the predicate labels right.
        if op == "putdown":
            if env.held is None:
                continue
            a1 = env.held
            a2 = None
        elif op == "stack":
            if env.held is None:
                continue
            a1 = env.held
            a2 = rng.choice([b for b in labels if b != a1])
        else:
            a1 = rng.choice(labels)
            a2 = rng.choice(labels) if arity == 2 else None
        if arity == 2 and a1 == a2:
            continue
        s_before = state_predicates(env, domain=domain)
        try:
            _obs, _r, _t, _tr, info = env.step(op, a1, a2)
            arename = _ACTION_RENAME.get(domain) or {}
            op_out = arename.get(op, op)
            if info.get("error"):
                transitions.append(Transition(
                    state_before=s_before,
                    action=(op_out, a1) if a2 is None else (op_out, a1, a2),
                    state_after=None,
                    error=info["error"],
                ))
            else:
                transitions.append(Transition(
                    state_before=s_before,
                    action=(op_out, a1) if a2 is None else (op_out, a1, a2),
                    state_after=state_predicates(env, domain=domain),
                    error=None,
                ))
        except Exception as e:
            continue
    return transitions


def _gt_for_domain(domain: str) -> dict:
    """Return the hand-written precondition/effect tables, renamed for the domain."""
    pmap = _DOMAIN_MAPS.get(domain) or {}
    arename = _ACTION_RENAME.get(domain) or {}
    def _rp(pred: str) -> str:
        # Rename the predicate name only; the arg template is kept.
        for src, dst in pmap.items():
            if pred.startswith(src + "(") or pred == src:
                return dst + pred[len(src):]
        return pred
    base = {
        "pickup":  {"pre_pos": {"handempty", "clear({a1})", "ontable({a1})"},
                    "pre_neg": set(),
                    "eff_add": {"holding({a1})"},
                    "eff_del": {"handempty", "clear({a1})",
                                "ontable({a1})"}},
        "unstack": {"pre_pos": {"handempty", "clear({a1})", "on({a1},{a2})"},
                    "pre_neg": set(),
                    "eff_add": {"holding({a1})", "clear({a2})"},
                    "eff_del": {"handempty", "clear({a1})",
                                "on({a1},{a2})"}},
        "putdown": {"pre_pos": {"holding({a1})"},
                    "pre_neg": set(),
                    "eff_add": {"handempty", "clear({a1})",
                                "ontable({a1})"},
                    "eff_del": {"holding({a1})"}},
        "stack":   {"pre_pos": {"holding({a1})", "clear({a2})"},
                    "pre_neg": set(),
                    "eff_add": {"handempty", "clear({a1})",
                                "on({a1},{a2})"},
                    "eff_del": {"holding({a1})", "clear({a2})"}},
    }
    out = {}
    for op, m in base.items():
        op_r = arename.get(op, op)
        out[op_r] = {k: {_rp(p) for p in v} if isinstance(v, set) else v
                     for k, v in m.items()}
    return out


GROUND_TRUTH = _gt_for_domain("blocksworld")


def induce_action_models(transitions: list[Transition]) -> dict:
    """Induce {action: {pre_pos, pre_neg, eff_add, eff_del, n_examples}} from successful transitions.

    Preconditions are the intersection of templated state_before sets;
    effects are the add/del diffs present in every instance. Predicate
    arguments are templated as {a1}, {a2}.
    """
    succ_by_action: dict[str, list[Transition]] = defaultdict(list)
    for tr in transitions:
        if tr.state_after is not None:
            succ_by_action[tr.action[0]].append(tr)

    models: dict[str, dict] = {}
    for op, ts in succ_by_action.items():
        if not ts:
            continue
        # Single-pass regex templating so already-templated tokens are
        # not re-matched; a repeated object takes its first position's
        # template.
        def _templ(pred: str, args: tuple[str, ...]) -> str:
            if not args:
                return pred
            arg_to_tmpl = {}
            for i, a in enumerate(args):
                if a not in arg_to_tmpl:
                    arg_to_tmpl[a] = f"{{a{i+1}}}"
            pattern = "|".join(re.escape(a) for a in arg_to_tmpl)
            return re.sub(rf"\b({pattern})\b",
                          lambda m: arg_to_tmpl[m.group(1)], pred)

        pre_template_sets = []
        eff_add_template = []
        eff_del_template = []
        for tr in ts:
            args = tr.action[1:]
            # Keep only predicates that mention an action argument or
            # are nullary (e.g. handempty).
            def _keep(p: str) -> bool:
                return ("{a" in p) or all(c not in p for c in "()")
            pre_set = {_templ(p, args) for p in tr.state_before}
            pre_set = {p for p in pre_set if _keep(p)}
            after = tr.state_after or frozenset()
            add = {_templ(p, args) for p in (after - tr.state_before)}
            add = {p for p in add if _keep(p)}
            dele = {_templ(p, args) for p in (tr.state_before - after)}
            dele = {p for p in dele if _keep(p)}
            pre_template_sets.append(pre_set)
            eff_add_template.append(add)
            eff_del_template.append(dele)

        pre_inter = (frozenset.intersection(*[frozenset(s) for s in pre_template_sets])
                      if pre_template_sets else frozenset())
        # Effects must appear in every instance of the action.
        add_count: defaultdict[str, int] = defaultdict(int)
        del_count: defaultdict[str, int] = defaultdict(int)
        for add in eff_add_template:
            for p in add: add_count[p] += 1
        for dele in eff_del_template:
            for p in dele: del_count[p] += 1
        n = len(ts)
        eff_add = {p for p, c in add_count.items() if c == n}
        eff_del = {p for p, c in del_count.items() if c == n}

        models[op] = {
            "pre_pos": set(pre_inter),
            "pre_neg": set(),
            "eff_add": eff_add,
            "eff_del": eff_del,
            "n_examples": n,
        }
    return models


def compare_models(induced: dict, domain: str = "blocksworld") -> dict:
    """Return per-action precision/recall/F1 of pre/add/del sets against ground truth."""
    out = {}
    gt_table = _gt_for_domain(domain)
    for op, gt in gt_table.items():
        if op not in induced:
            out[op] = {"missing": True}
            continue
        ind = induced[op]
        def _pr(pred_set, gt_set):
            tp = len(pred_set & gt_set)
            fp = len(pred_set - gt_set)
            fn = len(gt_set - pred_set)
            p = tp / max(tp + fp, 1)
            r = tp / max(tp + fn, 1)
            f1 = 2 * p * r / max(p + r, 1e-9) if (p + r) > 0 else 0.0
            return {"precision": p, "recall": r, "f1": f1}
        out[op] = {
            "pre_pos": _pr(ind["pre_pos"], gt["pre_pos"]),
            "eff_add": _pr(ind["eff_add"], gt["eff_add"]),
            "eff_del": _pr(ind["eff_del"], gt["eff_del"]),
            "n_examples": ind.get("n_examples", 0),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_rollouts", type=int, default=20)
    ap.add_argument("--steps_per_rollout", type=int, default=50)
    ap.add_argument("--n_blocks", type=int, default=4)
    ap.add_argument("--seed_base", type=int, default=0)
    ap.add_argument("--domain", default="blocksworld",
                    choices=["blocksworld", "mystery"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    labels = [chr(ord("a") + i) for i in range(args.n_blocks)]
    transitions: list[Transition] = []
    for k in range(args.n_rollouts):
        transitions.extend(random_rollout(
            seed=args.seed_base + k, labels=labels,
            steps=args.steps_per_rollout, domain=args.domain,
        ))
    print(f"collected {len(transitions)} transitions (of which "
          f"{sum(1 for t in transitions if t.state_after is not None)} "
          f"successful and "
          f"{sum(1 for t in transitions if t.error)} failed)")

    induced = induce_action_models(transitions)
    print("\n=== INDUCED MODELS ===")
    for op, m in induced.items():
        print(f"\n{op}:  ({m['n_examples']} examples)")
        print(f"  pre_pos: {sorted(m['pre_pos'])}")
        print(f"  eff_add: {sorted(m['eff_add'])}")
        print(f"  eff_del: {sorted(m['eff_del'])}")

    comparison = compare_models(induced, domain=args.domain)
    print("\n=== VS GROUND TRUTH ===")
    for op, cmp in comparison.items():
        if cmp.get("missing"):
            print(f"  {op}: MISSING from induced")
            continue
        print(f"\n  {op}:")
        for field_name, m in cmp.items():
            if field_name == "n_examples":
                continue
            print(f"    {field_name}: P={m['precision']:.2f} "
                  f"R={m['recall']:.2f} F1={m['f1']:.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "n_rollouts": args.n_rollouts,
            "n_transitions": len(transitions),
            "induced": {op: {k: sorted(list(v)) if isinstance(v, set)
                              else v for k, v in m.items()}
                        for op, m in induced.items()},
            "comparison": comparison,
        }, f, indent=2)
    print(f"\nwrote: {args.out}")


if __name__ == "__main__":
    main()
