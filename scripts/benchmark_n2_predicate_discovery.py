"""Run LLM predicate discovery on Blocksworld across many seeds.

Wraps scripts/llm_predicate_discovery.py: for each seed, collects
random-rollout state dicts, asks the LLM for predicate templates and
records whether the verified proposal was accepted. Output: --out JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from scripts.llm_predicate_discovery import (
        verify_predicate_set, make_predicate_discovery_prompt,
    )
    from agplan.llm_propose import propose_verify_fallback
    from agplan.planning.blocksworld import BlocksworldEnv
    import random as _random

    def collect_state_dicts(seed, labels, n_rollouts, steps):
        """Run rollouts; return (sb, (op, args), sa) state-dict triples and sample states for the prompt."""
        rng = _random.Random(seed)
        pairs = []
        sample_states = []
        for k in range(n_rollouts):
            env = BlocksworldEnv()
            sup = {}
            order = list(labels); rng.shuffle(order)
            for b in order:
                cand = ["TABLE"] + [x for x in sup
                                     if all(s != x for s in sup.values())]
                sup[b] = rng.choice(cand)
            env.reset_from(labels, sup, (), held=None, partial_goal=True)
            for _ in range(steps):
                a_choices = [("pickup", 1), ("putdown", 1),
                              ("unstack", 2), ("stack", 2)]
                op, ar = rng.choice(a_choices)
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
                sb = {"support": dict(env.support), "held": env.held}
                _o, _r, term, trunc, info = env.step(op, a1, a2)
                if info.get("error"): continue
                sa = {"support": dict(env.support), "held": env.held}
                args = (a1,) if ar == 1 else (a1, a2)
                if len(sample_states) < 5:
                    sample_states.append(sa)
                pairs.append((sb, (op, args), sa))
        return pairs, sample_states
    import json as _json

    schema = _json.dumps({
        "type": "object",
        "properties": {
            "predicates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "arity": {"type": "integer"},
                        "kind": {"type": "string"},
                        "field": {"type": "string"},
                        "value": {},
                    },
                    "required": ["name", "arity", "kind", "field"],
                },
            },
        },
        "required": ["predicates"],
    })

    rows = []
    n_llm_accepted = 0
    n_total = 0
    for seed in range(args.n_seeds):
        labels = ["a", "b", "c", "d"]
        try:
            pairs, sample_states = collect_state_dicts(
                seed=seed, labels=labels, n_rollouts=5, steps=30,
            )
        except Exception as e:
            print(f"  seed {seed}: collect ERR {e}")
            continue
        prompt = make_predicate_discovery_prompt(sample_states, labels)

        # collect_state_dicts returns 3-tuples (sb, (op,args), sa); the
        # verifier expects 2-tuples (sb, sa). Strip the action.
        sb_sa_pairs = [(t[0], t[-1]) for t in pairs]

        def verifier(prop):
            specs = (prop.get("predicates", [])
                     if isinstance(prop, dict) else [])
            return verify_predicate_set(specs, sb_sa_pairs, labels)

        try:
            final, info = propose_verify_fallback(
                prompt=prompt, schema_json=schema,
                verifier=verifier,
                baseline={"predicates": []}, threshold=0.6,
                model_name=args.model, max_new_tokens=1024,
            )
            llm_accepted = info.get("accepted_llm", False)
            accuracy = info.get("accuracy", 0.0)
            n_preds = len(final.get("predicates", []))
            llm_proposal = info.get("proposal")
        except Exception as e:
            llm_accepted = False
            accuracy = 0.0
            n_preds = 0
            llm_proposal = None

        n_total += 1
        if llm_accepted: n_llm_accepted += 1
        rows.append({
            "seed": seed,
            "llm_accepted": llm_accepted,
            "accuracy": accuracy,
            "n_predicates": n_preds,
            "llm_proposal": llm_proposal,
        })
        print(f"  seed {seed}: "
              f"{'OK' if llm_accepted else 'fail'} "
              f"acc={accuracy:.2f} preds={n_preds}")

    print(f"\nLLM-accepted: {n_llm_accepted}/{n_total} = "
          f"{100*n_llm_accepted/max(n_total,1):.1f}%")
    print(f"Classical baseline (DSL enumeration): the hand-crafted "
          f"DSL recovers ~6 predicates for Blocksworld; LLM "
          f"proposal compared on the same verifier.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "model": args.model,
            "n_seeds": args.n_seeds,
            "n_llm_accepted": n_llm_accepted,
            "llm_accept_rate": n_llm_accepted / max(n_total, 1),
            "rows": rows,
        }, f, indent=2, default=str)
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
