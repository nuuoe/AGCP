"""SCI-ReDuce end-to-end solve rate on held-out Mystery Blocksworld.

Uses run_leave_out_n.py's split protocol: per seed, the per-instance
state-class EBNF is compiled from the train-induced grammar, K plans are
decoded under the XGrammar mask and executed, and end-to-end success is
recorded beside gold-plan coverage. Input: a G_env run JSONL; output: --out.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

from agplan.grammar_learning.reduce_s import (
    StateAwareTrajectory,
    compile_state_aware_ebnf,
    induce_state_aware,
    wrap_state_aware_ebnf_in_plan_array,
)
from agplan.planning.blocksworld import BlocksworldEnv
from agplan.decoding.xgrammar_wrapper import (
    DecodeConfig, XGrammarConstrainedDecoder,
)
from scripts.run_leave_out_n import (
    _reconstruct_trajectory, _plan_admits, MYSTERY_TO_BLOCKSWORLD,
)


def load_trajectories(mystery_jsonl: str):
    rows = []
    with open(mystery_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    inst_by_id: dict = {}
    trajectories: list[tuple[str, StateAwareTrajectory]] = []
    for rec in rows:
        iid = rec.get("instance_id")
        if rec.get("initial") is None:
            continue
        init_sup = dict(rec["initial"])
        goal_pairs = tuple(tuple(p) for p in rec.get("goal", []))
        labels = sorted(set(list(init_sup.keys())
                            + [v for v in init_sup.values() if v != "TABLE"]
                            + [b for b, _ in goal_pairs]
                            + [s for _, s in goal_pairs if s != "TABLE"]))
        inst = {"labels": labels, "initial_support": init_sup,
                "goal_pairs": goal_pairs, "held": None,
                "domain": "mystery", "instance_id": iid}
        inst_by_id[iid] = inst
        for cand in rec.get("candidates", []):
            if not cand.get("success"):
                continue
            actions = cand.get("actions") or rec.get("success_actions")
            if not actions:
                continue
            traj = _reconstruct_trajectory(inst, actions)
            if traj is not None:
                trajectories.append((iid, traj))
                break
    return inst_by_id, trajectories


_SC_REF = re.compile(r"\bsc\d+\b")


def sanitize_ebnf(core: str):
    """Drop alternatives that reference state classes compile_state_aware_ebnf
    never defined (continuations pruned by the depth/goal cutoff), since
    XGrammar requires every referenced rule to exist; drop classes left with
    no alternatives; return None if the start class dies. This only removes
    strings from L(G), so soundness is preserved."""
    lines = [ln for ln in core.splitlines() if "::=" in ln]
    rules: dict[str, list[str]] = {}
    order: list[str] = []
    for ln in lines:
        lhs, rhs = ln.split("::=", 1)
        lhs = lhs.strip()
        rules[lhs] = [a.strip() for a in rhs.split("|")]
        order.append(lhs)
    changed = True
    while changed:
        changed = False
        defined = set(rules)
        for name in list(rules):
            kept = []
            for alt in rules[name]:
                refs = _SC_REF.findall(alt)
                if all(r in defined for r in refs):
                    kept.append(alt)
            if len(kept) != len(rules[name]):
                changed = True
            if kept:
                rules[name] = kept
            else:
                del rules[name]
                changed = True
    if "action_seq" not in rules:
        return None
    start_refs = [r for alt in rules["action_seq"]
                  for r in _SC_REF.findall(alt)]
    if not any(r in rules for r in start_refs):
        return None
    return "\n".join(f"{n} ::= {' | '.join(rules[n])}"
                     for n in order if n in rules)


_QUOTED = re.compile(r'"(?:\\.|[^"\\])*"')


def unroll_with_budget(clean: str, budget: int):
    """Budget-index the right-linear SCI-ReDuce grammar, mirroring
    G_env's (state, budget) construction: rule copies sc{i}__b{k}
    where k = remaining action budget. Alternatives that transition
    consume len(actions-in-body); transitions that would exceed the
    budget are dropped; terminal alternatives survive at any budget.
    The unrolled language is finite, so decoding must terminate."""
    lines = [ln for ln in clean.splitlines() if "::=" in ln]
    rules: dict[str, list[str]] = {}
    for ln in lines:
        lhs, rhs = ln.split("::=", 1)
        rules[lhs.strip()] = [a.strip() for a in rhs.split("|")]
    starts = [r for alt in rules.get("action_seq", [])
              for r in _SC_REF.findall(alt)]
    if not starts:
        return None

    def n_actions(alt: str) -> int:
        return sum(1 for tok in _QUOTED.findall(alt) if tok != '","'
                   and tok.strip('"') != ",")

    out: dict[str, list[str]] = {}
    seen: set = set()
    stack = [(s, budget) for s in starts]
    while stack:
        name, b = stack.pop()
        key = f"{name}__b{b}"
        if key in seen:
            continue
        seen.add(key)
        alts_out = []
        for alt in rules.get(name, []):
            refs = _SC_REF.findall(alt)
            cost = n_actions(alt)
            if not refs:
                alts_out.append(alt)  # terminal (goal) alternative
                continue
            if cost > b:
                continue
            new_alt = alt
            for r in refs:
                new_alt = re.sub(rf"\b{r}\b", f"{r}__b{b - cost}", new_alt)
                stack.append((r, b - cost))
            alts_out.append(new_alt)
        if alts_out:
            out[key] = alts_out
    # prune alternatives referencing keys that ended up empty
    changed = True
    while changed:
        changed = False
        for k in list(out):
            kept = [a for a in out[k]
                    if all(rf in out for rf in
                           re.findall(r"\bsc\d+__b\d+\b", a))]
            if len(kept) != len(out[k]):
                changed = True
            if kept:
                out[k] = kept
            else:
                del out[k]
                changed = True
    start_keys = [f"{s}__b{budget}" for s in starts if f"{s}__b{budget}" in out]
    if not start_keys:
        return None
    header = f"action_seq ::= {' | '.join(start_keys)}"
    body = "\n".join(f"{k} ::= {' | '.join(v)}" for k, v in out.items())
    return header + "\n" + body


def make_goal_check(goal_pairs):
    def chk(s):
        return all(
            (f"on({b},{sup})" in s if sup != "TABLE"
             else f"ontable({b})" in s)
            for b, sup in goal_pairs
        ) and ("handempty" in s)
    return chk


def execute_plan(inst: dict, actions: list[str]) -> bool:
    env = BlocksworldEnv()
    env.reset_from(inst["labels"], inst["initial_support"],
                   inst["goal_pairs"], inst.get("held"), partial_goal=True)
    for raw in actions:
        m = re.match(r"(\w+)\(([^)]*)\)", raw)
        if not m:
            return False
        op = MYSTERY_TO_BLOCKSWORLD.get(m.group(1), m.group(1))
        args = [a.strip() for a in m.group(2).split(",") if a.strip()]
        if op in ("pickup", "putdown"):
            _o, _r, term, trunc, info = env.step(
                op, args[0] if args else None, None)
        elif op in ("unstack", "stack"):
            _o, _r, term, trunc, info = env.step(
                op, args[0] if len(args) >= 1 else None,
                args[1] if len(args) >= 2 else None)
        else:
            return False
        if "error" in info:
            return False
        if term:
            return env.goal_satisfied()
        if trunc:
            return False
    return env.goal_satisfied()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mystery_jsonl", required=True)
    ap.add_argument("--n_holdout", type=int, default=10)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--variant", default="budget",
                    choices=["raw", "budget"],
                    help="raw = SCI-ReDuce grammar as induced "
                         "(admits unbounded macro loops); budget = "
                         "budget-unrolled finite grammar analogous to "
                         "G_env's (state,budget) indexing, budget = "
                         "gold plan length + 4")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    inst_by_id, trajectories = load_trajectories(args.mystery_jsonl)
    print(f"reconstructed {len(trajectories)} trajectories", flush=True)

    placeholder = 'root ::= "{\\"plan\\":[\\"pickup(a)\\"]}"\n'
    decoder = XGrammarConstrainedDecoder(model_name=args.model,
                                         ebnf=placeholder,
                                         device=args.device)
    template = (
        "You are a planning agent in an obfuscated blocks domain. "
        "Goal: {goal_desc}\n"
        "Output a JSON plan as {{\"plan\": [\"action(arg)\", ...]}}. "
        "The grammar admits only applicable action sequences."
    )

    per_seed = []
    all_rows = []
    for seed in args.seeds:
        rng = random.Random(seed)
        trajs = list(trajectories)
        rng.shuffle(trajs)
        test_set = trajs[: args.n_holdout]
        train_set = trajs[args.n_holdout:]
        grammar = induce_state_aware([t for _, t in train_set])
        n_cov = 0
        n_e2e = 0
        n_compiled = 0
        for iid, test_traj in test_set:
            inst = inst_by_id[iid]
            s0 = test_traj.states[0]
            ebnf_core = compile_state_aware_ebnf(
                grammar, s0, make_goal_check(inst["goal_pairs"]),
                max_depth=20)
            row = {"seed": seed, "instance_id": iid,
                   "n_gold_actions": len(test_traj.actions)}
            if ebnf_core is None:
                row.update({"compiled": False, "coverage": False,
                            "e2e_success": False})
                all_rows.append(row)
                print(f"  seed{seed} {iid}: no goal-reaching compile",
                      flush=True)
                continue
            n_compiled += 1
            cov = _plan_admits(ebnf_core, test_traj.actions)
            n_cov += int(cov)
            ebnf_clean = sanitize_ebnf(ebnf_core)
            if ebnf_clean is not None and args.variant == "budget":
                ebnf_clean = unroll_with_budget(
                    ebnf_clean, budget=len(test_traj.actions) + 4)
            if ebnf_clean is None:
                row.update({"compiled": True, "coverage": bool(cov),
                            "e2e_success": False,
                            "sanitize": "no goal-reaching language "
                                        f"({args.variant})"})
                all_rows.append(row)
                print(f"  seed{seed} {iid}: cov={cov} e2e=False "
                      f"(sanitize/{args.variant}: no goal-reaching "
                      f"language)", flush=True)
                continue
            full_ebnf = wrap_state_aware_ebnf_in_plan_array(ebnf_clean)
            try:
                decoder.recompile_ebnf(full_ebnf)
            except Exception as e:  # noqa: BLE001
                row.update({"compiled": True, "coverage": bool(cov),
                            "e2e_success": False,
                            "xgrammar_error": str(e)[:120]})
                all_rows.append(row)
                continue
            goal_desc = ", ".join(f"{b} on {sup}"
                                  for b, sup in inst["goal_pairs"])
            prompt = template.format(goal_desc=goal_desc)
            cfg = DecodeConfig(do_sample=True,
                               temperature=args.temperature,
                               top_p=args.top_p,
                               max_new_tokens=args.max_new_tokens)
            t0 = time.time()
            outs = decoder.generate_n(user_prompt=prompt, n=args.k, cfg=cfg)
            success = False
            n_valid = 0
            for raw in outs:
                try:
                    plan = list(json.loads(raw).get("plan", []))
                except Exception:  # noqa: BLE001
                    continue
                if execute_plan(inst, plan):
                    n_valid += 1
                    success = True
            n_e2e += int(success)
            row.update({"compiled": True, "coverage": bool(cov),
                        "e2e_success": success,
                        "n_goal_reaching_samples": n_valid,
                        "k": args.k,
                        "wall_seconds": round(time.time() - t0, 1)})
            all_rows.append(row)
            print(f"  seed{seed} {iid}: cov={cov} e2e={success} "
                  f"({n_valid}/{args.k} samples) "
                  f"{row['wall_seconds']}s", flush=True)
        n = len(test_set)
        per_seed.append({"seed": seed, "n": n, "compiled": n_compiled,
                         "coverage": n_cov, "e2e_success": n_e2e})
        print(f"seed {seed}: compiled {n_compiled}/{n}  "
              f"coverage {n_cov}/{n}  e2e {n_e2e}/{n}", flush=True)

    out = {"mystery_jsonl": args.mystery_jsonl,
           "model": args.model, "k": args.k,
           "variant": args.variant,
           "n_holdout": args.n_holdout,
           "per_seed": per_seed, "rows": all_rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    cov_m = sum(s["coverage"] for s in per_seed) / len(per_seed)
    e2e_m = sum(s["e2e_success"] for s in per_seed) / len(per_seed)
    print(f"\nMEAN over {len(per_seed)} seeds (n_holdout={args.n_holdout}): "
          f"coverage {cov_m:.1f}/10  e2e {e2e_m:.1f}/10")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
