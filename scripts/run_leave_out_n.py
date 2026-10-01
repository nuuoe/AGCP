"""Leave-out-N coverage test for the ReDuce-S grammar on Mystery Blocksworld.

Reconstructs state-annotated trajectories from a G_env run JSONL
(e.g. runs/scireduce_input/mystery.jsonl) by re-executing each successful
candidate, splits them into train and held-out sets, induces a state-class
indexed grammar from the train set, and checks whether the induced EBNF
admits each held-out gold plan (grammar membership only, no LLM). Writes JSONL to --out.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Optional

from agplan.grammar_learning.reduce_s import (
    StateAwareTrajectory,
    blocksworld_predicates,
    compile_state_aware_ebnf,
    induce_state_aware,
)
from agplan.planning.blocksworld import BlocksworldEnv
from agplan.planning.planbench_blocksworld import (
    MYSTERY_ACTION_NAMES,
    load_planbench_instances,
)

MYSTERY_TO_BLOCKSWORLD = {v: k for k, v in MYSTERY_ACTION_NAMES.items()}


# Trajectory reconstruction.


def _reconstruct_trajectory(
    instance: dict,
    actions: list[str],
) -> Optional[StateAwareTrajectory]:
    """Re-execute a recorded action sequence on the env, logging the state
    at each step; return None if execution fails or the goal is not reached."""
    env = BlocksworldEnv()
    env.reset_from(
        instance["labels"],
        instance["initial_support"],
        instance["goal_pairs"],
        instance.get("held"),
        partial_goal=True,
    )
    labels = list(env.labels)
    states: list[frozenset[str]] = [
        blocksworld_predicates(env.support, env.held, labels)
    ]
    applied: list[str] = []
    for raw in actions:
        m = re.match(r"(\w+)\(([^)]*)\)", raw)
        if not m:
            return None
        op = MYSTERY_TO_BLOCKSWORLD.get(m.group(1), m.group(1))
        args = [a.strip() for a in m.group(2).split(",") if a.strip()]
        if op in ("pickup", "putdown"):
            obs, _r, term, trunc, info = env.step(
                op, args[0] if args else None, None
            )
        elif op in ("unstack", "stack"):
            a1 = args[0] if len(args) >= 1 else None
            a2 = args[1] if len(args) >= 2 else None
            obs, _r, term, trunc, info = env.step(op, a1, a2)
        else:
            return None
        if "error" in info:
            return None
        applied.append(raw)
        states.append(
            blocksworld_predicates(env.support, env.held, labels)
        )
        if term:
            break
        if trunc:
            return None
    if not env.goal_satisfied():
        return None
    return StateAwareTrajectory(
        actions=tuple(applied),
        states=tuple(states),
    )


# EBNF parser for the coverage check (no xgrammar dependency).


def _plan_admits(ebnf_action_seq: str, action_seq: tuple[str, ...]) -> bool:
    """Check whether the ReDuce-S EBNF admits action_seq.

    The grammar is right-linear (sc_i -> body sc_j), so matching is a
    DFA-style walk over class-indexed productions in O(|plan| * |grammar|).
    Returns True iff the comma-separated JSON-quoted rendering of
    action_seq is derivable from a start class.
    """
    # rules: src_name -> list of (body_tokens, dst_name | None)
    rules: dict[str, list[tuple[list[str], Optional[str]]]] = {}
    for line in ebnf_action_seq.splitlines():
        line = line.strip()
        if not line or "::=" not in line:
            continue
        lhs, rhs = line.split("::=", 1)
        lhs = lhs.strip()
        if lhs == "action_seq":
            # action_seq ::= sc0  (or alternatives)
            rules.setdefault(lhs, [])
            for alt in rhs.split("|"):
                tok = alt.strip()
                rules[lhs].append(([], tok))
            continue
        if not lhs.startswith("sc"):
            continue
        rules.setdefault(lhs, [])
        for alt in rhs.split("|"):
            alt = alt.strip()
            if alt == '""':
                rules[lhs].append(([], None))
                continue
            # Tokenize: literals are "..." possibly escaped; classes
            # are sc<digits>.
            tokens: list[str] = []
            i = 0
            n = len(alt)
            dst: Optional[str] = None
            while i < n:
                if alt[i] == '"':
                    j = i + 1
                    while j < n and alt[j] != '"':
                        if alt[j] == "\\":
                            j += 2
                        else:
                            j += 1
                    raw = alt[i + 1:j]
                    raw = raw.replace('\\"', '"')
                    tokens.append(raw)
                    i = j + 1
                elif alt[i:i + 2] == "sc" and i + 2 < n and alt[i + 2].isdigit():
                    j = i + 2
                    while j < n and alt[j].isdigit():
                        j += 1
                    dst = alt[i:j]
                    i = j
                elif alt[i] == "w" and alt[i:i + 2] == "ws":
                    i += 2
                else:
                    i += 1
            rules[lhs].append((tokens, dst))

    # Render the action sequence as a flat token list with commas
    # interleaved -- same shape as bodies in the parsed grammar
    # (which include their own ',' separator tokens).
    target_tokens: list[str] = []
    for idx, a in enumerate(action_seq):
        if idx > 0:
            target_tokens.append(",")
        target_tokens.append(f'"{a}"')

    if "action_seq" not in rules:
        return False
    starts = [dst for _b, dst in rules["action_seq"] if dst]
    if not starts:
        return False

    def match(state: Optional[str], pos: int) -> bool:
        if state is None:
            return pos == len(target_tokens)
        if state not in rules:
            return False
        for body, dst in rules[state]:
            # Bodies are flat token lists with ',' separators; a trailing
            # ',' is stripped so the transition decides whether a
            # separator follows.
            b = list(body)
            trailing_comma = False
            if b and b[-1] == ",":
                trailing_comma = True
                b = b[:-1]
            blen = len(b)
            if pos + blen > len(target_tokens):
                continue
            if target_tokens[pos:pos + blen] != b:
                continue
            new_pos = pos + blen
            if dst is None:
                # terminal alternative: accept only at end of input
                if new_pos == len(target_tokens):
                    return True
                continue
            # Transition to dst; a trailing comma in the body consumes one
            # separator from the target.
            if trailing_comma:
                if new_pos < len(target_tokens) and \
                   target_tokens[new_pos] == ",":
                    if match(dst, new_pos + 1):
                        return True
                continue
            # No trailing comma: match dst at the same position.
            if match(dst, new_pos):
                return True
        return False

    for s in starts:
        if match(s, 0):
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mystery_jsonl", required=True,
                    help="Path to mystery.jsonl from a G_env run.")
    ap.add_argument("--instances_dir", default=None,
                    help="PlanBench instance dir to resolve initial / "
                         "goals. Optional: if the jsonl records embed "
                         "'initial' and 'goal', they are used directly.")
    ap.add_argument("--n_holdout", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--domain", default="mystery",
                    choices=["blocksworld", "mystery"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Index PlanBench instances by id for state recovery.
    inst_by_id = {}
    if args.instances_dir:
        instances = load_planbench_instances(
            Path(args.instances_dir), domain=args.domain
        )
        inst_by_id = {i["instance_id"]: i for i in instances}

    # Load mystery run jsonl, collect successful plans per instance.
    rows = []
    with open(args.mystery_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    trajectories: list[tuple[str, StateAwareTrajectory]] = []
    for rec in rows:
        iid = rec.get("instance_id")
        # Use the PlanBench instance if loaded, else the embedded (initial, goal).
        inst = inst_by_id.get(iid)
        if inst is None and rec.get("initial") is not None:
            init_sup = dict(rec["initial"])
            goal_pairs = tuple(tuple(p) for p in rec.get("goal", []))
            labels = sorted(set(list(init_sup.keys())
                                + [v for v in init_sup.values()
                                   if v != "TABLE"]
                                + [b for b, _ in goal_pairs]
                                + [s for _, s in goal_pairs
                                   if s != "TABLE"]))
            inst = {
                "labels": labels,
                "initial_support": init_sup,
                "goal_pairs": goal_pairs,
                "held": None,
                "domain": args.domain,
                "instance_id": iid,
            }
            inst_by_id[iid] = inst
        if inst is None:
            continue
        candidates = rec.get("candidates", [])
        for cand in candidates:
            if not cand.get("success"):
                continue
            actions = cand.get("actions") or rec.get("success_actions")
            if not actions:
                continue
            traj = _reconstruct_trajectory(inst, actions)
            if traj is not None:
                trajectories.append((iid, traj))
                break

    print(f"reconstructed {len(trajectories)} trajectories with state traces",
          flush=True)
    if len(trajectories) < args.n_holdout + 5:
        print(f"too few trajectories ({len(trajectories)}) for "
              f"holdout={args.n_holdout}; aborting", flush=True)
        return

    rng = random.Random(args.seed)
    rng.shuffle(trajectories)
    test_set = trajectories[:args.n_holdout]
    train_set = trajectories[args.n_holdout:]

    train_trajs = [t for _, t in train_set]
    grammar = induce_state_aware(train_trajs)
    print(f"trained ReDuce-S on {len(train_trajs)} trajectories; "
          f"{len(grammar.profiles)} macros profiled", flush=True)

    # For each test trajectory: compile per-instance EBNF, check if
    # the induced grammar admits the gold test plan.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    covered = 0
    compiled_ok = 0
    f_out = open(out_path, "w")
    for iid, test_traj in test_set:
        inst = inst_by_id[iid]
        s0 = test_traj.states[0]

        def make_goal_check(gp):
            def chk(s):
                return all(
                    (f"on({b},{sup})" in s if sup != "TABLE"
                     else f"ontable({b})" in s)
                    for b, sup in gp
                ) and ("handempty" in s)
            return chk

        action_seq_ebnf = compile_state_aware_ebnf(
            grammar, s0, make_goal_check(inst["goal_pairs"]),
            max_depth=20,
        )
        if action_seq_ebnf is None:
            rec = {"instance_id": iid, "compiled": False,
                   "coverage": False,
                   "n_actions": len(test_traj.actions)}
            f_out.write(json.dumps(rec) + "\n")
            continue
        compiled_ok += 1
        ok = _plan_admits(action_seq_ebnf, test_traj.actions)
        if ok:
            covered += 1
        rec = {"instance_id": iid, "compiled": True,
               "coverage": bool(ok),
               "n_actions": len(test_traj.actions),
               "ebnf_size": len(action_seq_ebnf)}
        f_out.write(json.dumps(rec) + "\n")
    f_out.close()

    n = len(test_set)
    print(f"compiled per-instance EBNF for {compiled_ok}/{n}", flush=True)
    print(f"coverage on held-out plans: {covered}/{n} = "
          f"{100*covered/n:.1f}%", flush=True)
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
