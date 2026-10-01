"""Predicate discovery with a strict verifier and a non-Blocksworld exemplar.

Differs from llm_predicate_discovery.py in two ways: the verifier
deduplicates proposals by name and requires at least min_distinct names,
of which a fraction >= rel_threshold must vary across transitions; the
in-context example is an abstract warehouse domain. Writes --out.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from agplan.llm_propose import propose_verify_fallback
from scripts.llm_predicate_discovery import evaluate_predicate


def verify_predicate_set_strict(specs: list, transition_pairs: list,
                                  objects: list,
                                  min_distinct: int = 3,
                                  rel_threshold: float = 0.6) -> float:
    """Score a predicate set strictly.

    Specs are deduplicated by name; fewer than min_distinct names scores 0.
    Otherwise the score is the fraction of names whose truth value varies
    across some transition, zeroed if below rel_threshold.
    """
    if not specs:
        return 0.0
    by_name: dict[str, dict] = {}
    for s in specs:
        nm = s.get("name") if isinstance(s, dict) else None
        if nm and nm not in by_name:
            by_name[nm] = s
    if len(by_name) < min_distinct:
        return 0.0
    n_relevant = 0
    for nm, spec in by_name.items():
        try:
            varied = False
            for sb, sa in transition_pairs[:30]:
                tb = evaluate_predicate(spec, sb, objects)
                ta = evaluate_predicate(spec, sa, objects)
                if tb != ta:
                    varied = True
                    break
            if varied:
                n_relevant += 1
        except Exception:
            pass
    score = n_relevant / len(by_name)
    return score if score >= rel_threshold else 0.0


def make_strict_prompt(sample_states: list, objects: list) -> str:
    """Build the discovery prompt with an abstract warehouse exemplar, so the
    LLM cannot copy canonical Blocksworld predicate names."""
    examples = "\n".join(
        f"  state_{i+1}: {json.dumps(s, default=str)}"
        for i, s in enumerate(sample_states[:5])
    )
    return f"""You are a planning-domain modeller. Given sample state
dicts from an env, propose Boolean predicates that capture its
dynamic structure. You MUST use only these 5 predicate KINDS:

  KIND                       MEANING                         ARITY
  field_eq_value             state[field] == value           0
  field_at_X_eq_value        state[field][a1] == value       1
  field_at_X_eq_Y            state[field][a1] == a2          2
  forall_K_field_at_K_neq_X  for all k, state[field][k] != a1  1
  exists_K_field_at_K_eq_X   exists k, state[field][k] == a1   1

## Worked example (abstract WAREHOUSE domain — NOT your target)
Suppose env state has fields 'cart' (object name or null = empty)
and 'pos' (dict obj->location). Predicates THAT WOULD WORK:
  cart_empty:    arity 0, kind field_eq_value,    field 'cart', value null
  at(a1,a2):     arity 2, kind field_at_X_eq_Y,   field 'pos'
  at_dest(a1):   arity 1, kind field_at_X_eq_value, field 'pos', value 'DEST'
  any_in_cart(a1): arity 1, kind exists_K_field_at_K_eq_X, field 'cart', value a1
  no_blockers(a1): arity 1, kind forall_K_field_at_K_neq_X, field 'pos', value a1

(The names and fields above are for the WAREHOUSE example only.
You must invent your own names/fields based on your env's actual
state structure.)

## Format invariants (must hold)
- arity 0 ↔ kind `field_eq_value`
- arity 1 ↔ kind ∈ {{field_at_X_eq_value, forall_K_field_at_K_neq_X, exists_K_field_at_K_eq_X}}
- arity 2 ↔ kind `field_at_X_eq_Y`
- DO NOT enumerate per-object (e.g., NOT 'holding_a', 'holding_b').
  Use templates; verifier fills arguments at grounding time.

## Your task
Object identifiers visible in the env: {objects[:8]}{'...' if len(objects) > 8 else ''}

Sample states observed (look at fields and value patterns):
{examples}

Propose Boolean predicates following the schema. Aim for predicates
that change across states (dynamic) and capture the env's STRUCTURE
(positional / state-dependent / quantified). The exact names and
fields are for YOU to discover from the sample states — do NOT copy
the warehouse example.

Output JSON in EXACTLY this format (replace with YOUR predicates):
{{"predicates": [
  {{"name": "your_name_1", "arity": 0|1|2, "kind": "<one of 5>",
    "field": "<field_name>", "value": <if applicable>}},
  ...
]}}

Output only the JSON object — no commentary."""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="blocksworld",
                    choices=["blocksworld"])
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--n_rollouts", type=int, default=5)
    ap.add_argument("--n_steps", type=int, default=30)
    ap.add_argument("--n_blocks", type=int, default=4)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--min_distinct", type=int, default=3)
    ap.add_argument("--rel_threshold", type=float, default=0.6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    schema = json.dumps({
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
    n_accept = 0
    for seed in range(args.n_seeds):
        labels = [chr(ord('a') + i) for i in range(args.n_blocks)]
        from agplan.planning.blocksworld import BlocksworldEnv
        rng = random.Random(seed)
        pairs = []
        sample_states = []
        for k in range(args.n_rollouts):
            env = BlocksworldEnv()
            sup = {}
            order = list(labels); rng.shuffle(order)
            for b in order:
                cand = ["TABLE"] + [x for x in sup
                                     if all(s != x for s in sup.values())]
                sup[b] = rng.choice(cand)
            env.reset_from(labels, sup, (), held=None, partial_goal=True)
            for _ in range(args.n_steps):
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
                if len(sample_states) < 5:
                    sample_states.append(sa)
                pairs.append((sb, sa))

        prompt = make_strict_prompt(sample_states, labels)

        def verifier(prop, pairs=pairs, labels=labels):
            specs = (prop.get("predicates", [])
                     if isinstance(prop, dict) else [])
            return verify_predicate_set_strict(
                specs, pairs, labels,
                min_distinct=args.min_distinct,
                rel_threshold=args.rel_threshold)

        try:
            final, info = propose_verify_fallback(
                prompt=prompt, schema_json=schema,
                verifier=verifier,
                baseline={"predicates": []},
                threshold=args.rel_threshold,
                model_name=args.model, max_new_tokens=1024,
                n_samples=4, temperature=0.5,
                max_refine_rounds=1,
            )
            accepted = info.get("accepted_llm", False)
            accuracy = info.get("accuracy", 0.0)
            n_unique = len({
                s["name"] for s in (final.get("predicates", []) or [])
                if isinstance(s, dict) and "name" in s
            })
            llm_proposal = info.get("proposal")
        except Exception as e:
            accepted, accuracy, n_unique, llm_proposal = False, 0.0, 0, None

        if accepted:
            n_accept += 1
        rows.append({
            "seed": seed,
            "llm_accepted": accepted,
            "accuracy": accuracy,
            "n_unique_predicates": n_unique,
            "llm_proposal": llm_proposal,
        })
        print(f"  seed {seed}: "
              f"{'OK' if accepted else 'fail'} "
              f"acc={accuracy:.2f} unique={n_unique}")

    print(f"\nStrict predicate discovery / Blocksworld (min_distinct={args.min_distinct}, "
          f"threshold={args.rel_threshold}, ICL=abstract-warehouse)")
    print(f"LLM accepted: {n_accept}/{args.n_seeds} = "
          f"{100*n_accept/max(args.n_seeds,1):.1f}%")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "model": args.model,
            "n_seeds": args.n_seeds,
            "n_llm_accepted": n_accept,
            "llm_accept_rate": n_accept / max(args.n_seeds, 1),
            "verifier": {
                "min_distinct": args.min_distinct,
                "rel_threshold": args.rel_threshold,
                "icl_exemplar": "abstract-warehouse (not Blocksworld)",
            },
            "rows": rows,
        }, f, indent=2, default=str)
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
