"""Discover a predicate vocabulary from raw structured state observations.

Candidate Boolean predicates are enumerated over a small DSL of field/key
equalities (nullary, unary, binary, exists/forall) and filtered to those
whose truth value changes across rollouts; the kept vocabulary then feeds
intersection-stats induction. Demo on Blocksworld; output: --out JSON.
"""
from __future__ import annotations

import re
from typing import Callable


class Predicate:
    def __init__(self, name: str, arity: int,
                 fn: Callable[..., bool]):
        self.name = name
        self.arity = arity
        self.fn = fn  # signature: fn(raw_state, *args) -> bool

    def __repr__(self) -> str:
        return f"Predicate({self.name}/{self.arity})"


def enumerate_atomic_predicates(
    raw_state: dict, objects: list[str], specials: list = None,
) -> list[Predicate]:
    """Enumerate templated Boolean predicates over a dict state.

    Primitive fields yield nullary `field == c` and unary `field == X`
    predicates; dict fields yield unary `field[X] == c`, binary
    `field[X] == Y` and exists/forall-over-keys predicates. `specials`
    are the constants compared against (default 'TABLE' and None).
    One Predicate per template, not per ground instance.
    """
    if specials is None:
        specials = ["TABLE", None]
    preds: list[Predicate] = []

    for field, val in raw_state.items():
        if isinstance(val, dict):
            # FIELD_AT_KEY_IS for each special value
            for c in specials:
                def make_fn(field=field, c=c):
                    def fn(s, a1):
                        return s.get(field, {}).get(a1) == c
                    return fn
                cstr = "NONE" if c is None else str(c)
                preds.append(Predicate(
                    name=f"{field}_at_X_is_{cstr}", arity=1,
                    fn=make_fn(field, c),
                ))
            # FIELD_AT_KEY_IS_KEY: state[field][a1] == a2 (binary)
            def make_bin(field=field):
                def fn(s, a1, a2):
                    return s.get(field, {}).get(a1) == a2
                return fn
            preds.append(Predicate(
                name=f"{field}_at_X_is_Y", arity=2,
                fn=make_bin(field),
            ))
            # FORALL_KEY_FIELD_NEQ (clear pattern)
            def make_forall(field=field):
                def fn(s, a1):
                    return all(v != a1 for v in s.get(field, {}).values())
                return fn
            preds.append(Predicate(
                name=f"forall_K_{field}_at_K_neq_X", arity=1,
                fn=make_forall(field),
            ))
            # EXISTS_KEY_FIELD_IS
            def make_exists(field=field):
                def fn(s, a1):
                    return any(v == a1 for v in s.get(field, {}).values())
                return fn
            preds.append(Predicate(
                name=f"exists_K_{field}_at_K_is_X", arity=1,
                fn=make_exists(field),
            ))
        else:
            # Primitive field. Generate scalar-equality nullaries.
            for c in specials:
                def make_fn(field=field, c=c):
                    def fn(s):
                        return s.get(field) == c
                    return fn
                cstr = "NONE" if c is None else str(c)
                preds.append(Predicate(
                    name=f"{field}_is_{cstr}", arity=0,
                    fn=make_fn(field, c),
                ))
            # Primitive field can also equal an object: state[field] == a1
            def make_obj_eq(field=field):
                def fn(s, a1):
                    return s.get(field) == a1
                return fn
            preds.append(Predicate(
                name=f"{field}_is_X", arity=1,
                fn=make_obj_eq(field),
            ))

    return preds


def evaluate_predicates_at_state(
    raw_state: dict, predicates: list[Predicate],
    objects: list[str],
) -> set[str]:
    """Return the set of GROUND predicate strings true at raw_state.

    Each Predicate template is instantiated over `objects` (or
    nullary). True predicates are formatted as `<name>(arg1,arg2,...)`.
    """
    out = set()
    for p in predicates:
        if p.arity == 0:
            try:
                if p.fn(raw_state):
                    out.add(p.name)
            except Exception:
                pass
        elif p.arity == 1:
            for a1 in objects:
                try:
                    if p.fn(raw_state, a1):
                        out.add(f"{p.name}({a1})")
                except Exception:
                    pass
        elif p.arity == 2:
            for a1 in objects:
                for a2 in objects:
                    if a1 == a2:
                        continue
                    try:
                        if p.fn(raw_state, a1, a2):
                            out.add(f"{p.name}({a1},{a2})")
                    except Exception:
                        pass
    return out


def filter_action_relevant(
    predicates: list[Predicate], raw_state_pairs: list[tuple],
    objects: list[str],
) -> list[Predicate]:
    """Keep predicates whose ground truth value changes in at least one (s_before, s_after) pair."""
    def truths(rs):
        return evaluate_predicates_at_state(rs, predicates, objects)
    kept_names: set[str] = set()
    for sb, sa in raw_state_pairs:
        tb = truths(sb)
        ta = truths(sa)
        diff = tb.symmetric_difference(ta)
        for d in diff:
            # Strip arguments to get predicate template name
            m = re.match(r"^([^(]+)", d)
            if m:
                kept_names.add(m.group(1))
    return [p for p in predicates if p.name in kept_names]


def blocksworld_raw_state(env) -> dict:
    """Expose BlocksworldEnv's implementation fields (support, held) as a dict; no predicate vocabulary is assumed."""
    return {"support": dict(env.support), "held": env.held}


def main() -> None:
    import argparse, json
    from agplan.planning.blocksworld import BlocksworldEnv
    import random

    ap = argparse.ArgumentParser()
    ap.add_argument("--n_rollouts", type=int, default=10)
    ap.add_argument("--steps_per_rollout", type=int, default=50)
    ap.add_argument("--n_blocks", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    labels = [chr(ord("a") + i) for i in range(args.n_blocks)]

    # Random rollouts: collect (sb, action, sa) raw-state triples.
    transitions = []
    for k in range(args.n_rollouts):
        rng = random.Random(k)
        env = BlocksworldEnv()
        sup = {}
        order = list(labels); rng.shuffle(order)
        for b in order:
            choices = ["TABLE"] + [x for x in sup
                                    if all(s != x for s in sup.values())]
            sup[b] = rng.choice(choices)
        env.reset_from(labels, sup, (), held=None, partial_goal=True)
        actions = [("pickup", 1), ("putdown", 1),
                   ("unstack", 2), ("stack", 2)]
        for _ in range(args.steps_per_rollout):
            op, ar = rng.choice(actions)
            if op == "putdown":
                if env.held is None: continue
                a1, a2 = env.held, None
            elif op == "stack":
                if env.held is None: continue
                a1 = env.held
                others = [b for b in labels if b != a1]
                if not others: continue
                a2 = rng.choice(others)
            else:
                a1 = rng.choice(labels)
                a2 = rng.choice(labels) if ar == 2 else None
            if ar == 2 and a1 == a2: continue
            sb = blocksworld_raw_state(env)
            _o, _r, _t, _tr, info = env.step(op, a1, a2)
            sa = blocksworld_raw_state(env)
            if info.get("error"):
                continue
            transitions.append((sb, (op, a1) if a2 is None
                                 else (op, a1, a2), sa))

    print(f"collected {len(transitions)} successful transitions")

    # Candidate predicates are enumerated from the first raw state.
    sample_state = transitions[0][0]
    cand = enumerate_atomic_predicates(sample_state, labels)
    print(f"enumerated {len(cand)} candidate predicates from DSL:")
    for p in cand:
        print(f"  {p.name}/{p.arity}")

    pairs = [(sb, sa) for sb, _act, sa in transitions]
    kept = filter_action_relevant(cand, pairs, labels)
    print(f"\nkept {len(kept)} action-relevant predicates:")
    for p in kept:
        print(f"  {p.name}/{p.arity}")

    # Render states with the kept predicates and run intersection-stats
    # induction.
    from scripts.induce_world_model import induce_action_models, Transition
    state_aware_transitions = []
    for sb, action, sa in transitions:
        sb_preds = evaluate_predicates_at_state(sb, kept, labels)
        sa_preds = evaluate_predicates_at_state(sa, kept, labels)
        state_aware_transitions.append(Transition(
            state_before=frozenset(sb_preds),
            action=action,
            state_after=frozenset(sa_preds),
            error=None,
        ))

    induced = induce_action_models(state_aware_transitions)
    print(f"\n=== INDUCED MODELS (vocab from discovery, not handwritten) ===")
    for op, m in induced.items():
        print(f"  {op}:  ({m['n_examples']} ex)")
        print(f"    pre_pos: {sorted(m['pre_pos'])}")
        print(f"    eff_add: {sorted(m['eff_add'])}")
        print(f"    eff_del: {sorted(m['eff_del'])}")

    # Induced names differ from the canonical ones ("support_at_X_is_TABLE"
    # vs "ontable(X)"), so compare the sizes of the pre/add/del sets with
    # the hand-written models instead.
    print("\n=== STRUCTURAL CHECK ===")
    GT_SIZES = {
        "pickup":  {"pre": 3, "add": 1, "del": 3},
        "putdown": {"pre": 1, "add": 3, "del": 1},
        "stack":   {"pre": 2, "add": 3, "del": 2},
        "unstack": {"pre": 3, "add": 2, "del": 3},
    }
    n_match = 0
    n_total = 0
    for op, m in induced.items():
        gt = GT_SIZES.get(op)
        if not gt: continue
        for k, exp in gt.items():
            n_total += 1
            ind_key = {"pre": "pre_pos", "add": "eff_add",
                       "del": "eff_del"}[k]
            ind_size = len(m[ind_key])
            match = ind_size == exp
            if match: n_match += 1
            print(f"  {op}.{k}: induced={ind_size} expected={exp}  "
                  f"{'ok' if match else 'mismatch'}")
    print(f"\n  STRUCTURAL MATCH: {n_match}/{n_total}")

    import json as _json
    from pathlib import Path
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        _json.dump({
            "n_transitions": len(transitions),
            "n_candidates": len(cand),
            "n_kept": len(kept),
            "kept_predicates": [
                {"name": p.name, "arity": p.arity} for p in kept
            ],
            "induced": {op: {k: sorted(list(v))
                              if isinstance(v, set) else v
                              for k, v in m.items()}
                        for op, m in induced.items()},
            "structural_match": f"{n_match}/{n_total}",
        }, f, indent=2)
    print(f"\nwrote: {out}")


if __name__ == "__main__":
    main()
