"""LLM-proposed predicate discovery, verified by action relevance.

The LLM proposes named predicate templates in the restricted DSL of
discover_predicates.py (field/key equalities and quantified forms); a
proposal scores by the fraction of predicates whose value changes across
rollout transitions, with the enumerated DSL as baseline. Output: --out JSON.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from agplan.llm_propose import propose_verify_fallback


def compile_predicate_from_spec(spec: dict):
    """Return (name, arity, fn) with fn(state, *args) -> bool.

    The DSL is deliberately restricted (no arbitrary Python) so LLM
    output cannot execute code.
    """
    name = spec.get("name", "anon")
    arity = int(spec.get("arity", 0))
    kind = spec.get("kind", "")
    field = spec.get("field", "")
    value = spec.get("value", None)

    if kind == "field_eq_value":
        def fn(state, *a):
            return state.get(field) == value
        return name, 0, fn
    if kind == "field_at_X_eq_value":
        def fn(state, a1, *_):
            return state.get(field, {}).get(a1) == value
        return name, 1, fn
    if kind == "field_at_X_eq_Y":
        def fn(state, a1, a2, *_):
            return state.get(field, {}).get(a1) == a2
        return name, 2, fn
    if kind == "forall_K_field_at_K_neq_X":
        def fn(state, a1, *_):
            return all(v != a1 for v in state.get(field, {}).values())
        return name, 1, fn
    if kind == "exists_K_field_at_K_eq_X":
        def fn(state, a1, *_):
            return any(v == a1 for v in state.get(field, {}).values())
        return name, 1, fn
    raise ValueError(f"unknown predicate kind: {kind}")


def evaluate_predicate(spec, state, objects):
    """Return set of ground predicate strings true at `state`."""
    name, arity, fn = compile_predicate_from_spec(spec)
    out = set()
    if arity == 0:
        try:
            if fn(state): out.add(name)
        except Exception:
            pass
    elif arity == 1:
        for a in objects:
            try:
                if fn(state, a): out.add(f"{name}({a})")
            except Exception:
                pass
    elif arity == 2:
        for a in objects:
            for b in objects:
                if a == b: continue
                try:
                    if fn(state, a, b):
                        out.add(f"{name}({a},{b})")
                except Exception:
                    pass
    return out


def verify_predicate_set(specs: list, transition_pairs: list,
                          objects: list) -> float:
    """Return the fraction of proposed predicates whose value changes in at least one transition."""
    if not specs: return 0.0
    n_relevant = 0
    for spec in specs:
        try:
            varied = False
            for sb, sa in transition_pairs[:30]:
                tb = evaluate_predicate(spec, sb, objects)
                ta = evaluate_predicate(spec, sa, objects)
                if tb != ta:
                    varied = True; break
            if varied:
                n_relevant += 1
        except Exception:
            pass
    return n_relevant / max(len(specs), 1)


def make_predicate_discovery_prompt(sample_states: list,
                                     objects: list) -> str:
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

## Worked example (Blocksworld)
States expose two fields: 'support' (dict block -> block-below)
and 'held' (block-being-held or null).

Good predicates for Blocksworld:
  handempty: arity 0, kind field_eq_value, field 'held', value null
       (means: nothing is held)
  on(a1, a2): arity 2, kind field_at_X_eq_Y, field 'support', value null
       (means: support[a1] == a2)
  ontable(a1): arity 1, kind field_at_X_eq_value, field 'support', value 'TABLE'
       (means: support[a1] == 'TABLE')
  clear(a1): arity 1, kind forall_K_field_at_K_neq_X, field 'support', value null
       (means: no other block has a1 below it)
## ⚠️ Format invariants (must hold)
- arity 0 ↔ kind `field_eq_value`
- arity 1 ↔ kind ∈ {{field_at_X_eq_value, forall_K_field_at_K_neq_X, exists_K_field_at_K_eq_X}}
- arity 2 ↔ kind `field_at_X_eq_Y`
- DO NOT mix arities and kinds. DO NOT enumerate per-object
  (e.g., do NOT output 'holding_a', 'holding_b'). Templates are
  filled by the verifier at grounding time.

Output JSON in EXACTLY this format (these 4 are sufficient for
Blocksworld; you may add fewer or different ones for other envs):
{{"predicates": [
  {{"name": "handempty", "arity": 0, "kind": "field_eq_value", "field": "held", "value": null}},
  {{"name": "on", "arity": 2, "kind": "field_at_X_eq_Y", "field": "support"}},
  {{"name": "ontable", "arity": 1, "kind": "field_at_X_eq_value", "field": "support", "value": "TABLE"}},
  {{"name": "clear", "arity": 1, "kind": "forall_K_field_at_K_neq_X", "field": "support"}}
]}}

## Your task
Object identifiers: {objects[:8]}{'...' if len(objects) > 8 else ''}

Sample states observed:
{examples}

Propose Boolean predicates following the schema above. Aim for
predicates that change across states (dynamic), not constants.
Output only the JSON object — no commentary."""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="blocksworld")
    ap.add_argument("--n_rollouts", type=int, default=10)
    ap.add_argument("--n_steps", type=int, default=50)
    ap.add_argument("--n_blocks", type=int, default=4)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--mock_llm", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Raw-state (sb, sa) pairs from random Blocksworld rollouts.
    from agplan.planning.blocksworld import BlocksworldEnv
    labels = [chr(ord('a') + i) for i in range(args.n_blocks)]
    pairs = []
    sample_states = []
    rng = random.Random(0)
    for k in range(args.n_rollouts):
        env = BlocksworldEnv()
        sup = {}
        order = list(labels); rng.shuffle(order)
        for b in order:
            choices = ["TABLE"] + [x for x in sup
                                    if all(s != x for s in sup.values())]
            sup[b] = rng.choice(choices)
        env.reset_from(labels, sup, (), held=None, partial_goal=True)
        for _ in range(args.n_steps):
            actions = [("pickup", 1), ("putdown", 1),
                       ("unstack", 2), ("stack", 2)]
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
            sb = {"support": dict(env.support), "held": env.held}
            _o, _r, _t, _tr, info = env.step(op, a1, a2)
            sa = {"support": dict(env.support), "held": env.held}
            if info.get("error"): continue
            pairs.append((sb, sa))
            if len(sample_states) < 5:
                sample_states.append(sb)
    print(f"collected {len(pairs)} transition pairs", flush=True)

    # Baseline: the enumerated DSL from discover_predicates.py.
    from scripts.discover_predicates import (
        enumerate_atomic_predicates, filter_action_relevant,
    )
    baseline_preds = enumerate_atomic_predicates(
        sample_states[0], labels,
    )
    baseline_kept = filter_action_relevant(baseline_preds, pairs, labels)
    print(f"baseline (hardcoded DSL): {len(baseline_preds)} candidates, "
          f"{len(baseline_kept)} kept", flush=True)

    prompt = make_predicate_discovery_prompt(sample_states, labels)
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

    def verifier(prop):
        specs = prop.get("predicates", []) if isinstance(prop, dict) else []
        return verify_predicate_set(specs, pairs, labels)

    # Mock proposal covering the canonical Blocksworld predicates; the
    # "holding" entry has an invalid kind on purpose.
    mock = {"predicates": [
        {"name": "handempty", "arity": 0, "kind": "field_eq_value",
         "field": "held", "value": None},
        {"name": "holding", "arity": 1, "kind": "field_eq_value_???",
         "field": "held", "value": None},
        {"name": "ontable", "arity": 1, "kind": "field_at_X_eq_value",
         "field": "support", "value": "TABLE"},
        {"name": "on", "arity": 2, "kind": "field_at_X_eq_Y",
         "field": "support"},
        {"name": "clear", "arity": 1, "kind": "forall_K_field_at_K_neq_X",
         "field": "support"},
    ]}

    final, info = propose_verify_fallback(
        prompt=prompt,
        schema_json=schema,
        verifier=verifier,
        baseline={"predicates": []},   # used when no proposal verifies
        threshold=0.6,
        model_name=args.model,
        use_mock=args.mock_llm,
        mock_value=mock,
    )

    print(f"\n=== RESULT ===")
    print(f"accepted_llm: {info['accepted_llm']}")
    print(f"accuracy:     {info['accuracy']:.2f}")
    print(f"final predicate count: {len(final.get('predicates', []))}")
    for p in final.get('predicates', [])[:10]:
        print(f"  {p}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "baseline": {
                "n_candidates": len(baseline_preds),
                "n_kept": len(baseline_kept),
                "kept_names": [p.name for p in baseline_kept],
            },
            "llm": info,
            "final": final,
        }, f, indent=2, default=str)
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
