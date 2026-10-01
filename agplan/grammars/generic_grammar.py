"""Compile induced action models into a CFG of goal-reaching plans.

Given action models (precondition and effect tables over a templated
predicate vocabulary), an initial state, a goal predicate set and an
object universe, the compiler grounds the actions, runs a state-space
BFS and emits an EBNF that admits exactly the goal-reaching plans
within the budget. Action-model format, as produced by
induce_world_model.py:

    {"pickup": {"pre_pos": {"handempty", "clear({a1})", "ontable({a1})"},
                "eff_add": {"holding({a1})"},
                "eff_del": {"handempty", "clear({a1})", "ontable({a1})"},
                "arity": 1}, ...}

Predicates are templated with {a1}, {a2}, ... for action arguments.
"""
from __future__ import annotations

import re
from collections import deque
from typing import Optional


# A state is a frozenset of predicate strings.
State = frozenset


def _ground_pred(template: str, args: tuple[str, ...]) -> str:
    """Ground a templated predicate by substituting {a1}, {a2}, ..."""
    out = template
    for i, a in enumerate(args, start=1):
        out = out.replace(f"{{a{i}}}", a)
    return out


def _ground_action_set(action: str, model: dict, args: tuple[str, ...]
                       ) -> tuple[set[str], set[str], set[str]]:
    """Ground (pre_pos, eff_add, eff_del) for one specific arg tuple."""
    pre = {_ground_pred(p, args) for p in model["pre_pos"]}
    add = {_ground_pred(p, args) for p in model["eff_add"]}
    rem = {_ground_pred(p, args) for p in model["eff_del"]}
    return pre, add, rem


def _enumerate_args(action: str, model: dict, objects: list[str]
                    ) -> list[tuple[str, ...]]:
    """All argument tuples of distinct objects for an action; the arity is
    the model's, or the largest {aN} index in its templates."""
    max_idx = 0
    for tmpl_set in (model.get("pre_pos", set()),
                     model.get("eff_add", set()),
                     model.get("eff_del", set())):
        for tmpl in tmpl_set:
            for m in re.finditer(r"\{a(\d+)\}", tmpl):
                max_idx = max(max_idx, int(m.group(1)))
    arity = model.get("arity", max_idx)
    if arity == 0:
        return [()]
    if arity == 1:
        return [(o,) for o in objects]
    # General arity n: all length-n tuples of distinct objects.
    import itertools
    return [tup for tup in itertools.product(objects, repeat=arity)
            if len(set(tup)) == arity]


def _applicable(state: State, pre: set[str]) -> bool:
    """Positive preconditions hold; negative preconditions are not modelled."""
    return pre <= state


def _apply(state: State, add: set[str], rem: set[str]) -> State:
    """Standard PDDL effect application: state - eff_del + eff_add."""
    return frozenset((set(state) - rem) | add)


def _action_to_json(action: str, args: tuple[str, ...]) -> str:
    """The action token the model emits: "action(arg1,arg2,...)"."""
    if not args:
        return action
    return f"{action}({','.join(args)})"


def compile_from_action_models_astar(
    pyperplan_task,
    initial_state: State,
    goal_predicates: set[str],
    max_extra: int = 4,
) -> Optional[str]:
    """Fallback for domains whose state space is too large for
    compile_from_action_models: find one plan with pyperplan's A* + hAdd
    and emit a CFG that admits exactly that sequence. Decoding still runs
    through the mask, which now admits a single plan.
    """
    from pyperplan.search import astar_search
    from pyperplan.heuristics.relaxation import hAddHeuristic

    try:
        sol = astar_search(pyperplan_task,
                              heuristic=hAddHeuristic(pyperplan_task))
    except Exception:
        return None
    if not sol:
        return None

    productions: list[str] = []
    productions.append('root ::= "{\\"plan\\":[" plan_seq "]}"')

    action_strs = []
    import re as _re
    for op in sol:
        # op.name is "(action arg1 arg2 ...)"
        name = op.name.strip()
        m = _re.match(r"^\(([\w-]+)\s*(.*?)\)\s*$", name)
        if not m: continue
        head = m.group(1)
        args = [a for a in m.group(2).split() if a]
        if args:
            tok = f"{head}({','.join(args)})"
        else:
            tok = head
        action_strs.append(tok)

    if not action_strs:
        productions.append('plan_seq ::= ""')
    else:
        # Separators live in named variables because Python 3.11
        # forbids backslashes inside f-string expressions.
        bs = '\\'
        parts = ['"' + bs + '"' + tok + bs + '""' for tok in action_strs]
        sep = ' "," '
        productions.append("plan_seq ::= " + sep.join(parts))
    return "\n".join(productions)


def compile_from_action_models(
    action_models: dict,
    initial_state: State,
    goal_predicates: set[str],
    objects: list[str],
    max_extra: int = 4,
    max_length: Optional[int] = None,
    ground_op_filter: Optional[dict] = None,
    ground_op_oracle: Optional[dict] = None,
    max_reachable: int = 5000,
    stats: Optional[dict] = None,
) -> Optional[str]:
    """Compile an EBNF admitting exactly the goal-reaching plans under the action models.

    action_models maps each action name to {pre_pos, eff_add, eff_del[,
    arity]}. The plan-length budget is the shortest path plus max_extra,
    or max_length when given. Grounding uses, in order of precedence,
    ground_op_oracle ((action, args) -> object with applicable(state) and
    apply(state), pyperplan's exact check), ground_op_filter (action ->
    allowed argument tuples, applicability still from the induced model),
    or the untyped product over objects. Reachability stops after
    max_reachable states, which keeps large domains (depot, satellite)
    tractable; stats, if given, receives the counts and the failure
    reason. Returns None when no goal state is reachable within the
    budget.
    """
    # Reachability BFS.
    reachable = {initial_state}
    transitions: dict[tuple[State, str, tuple[str, ...]], State] = {}
    queue = deque([initial_state])
    while queue:
        if len(reachable) >= max_reachable:
            # Cap hit: the grammar is built from the explored part of the
            # state space.
            if stats is not None:
                stats["cap_hit"] = True
            break
        s = queue.popleft()
        for action, model in action_models.items():
            if ground_op_filter is not None:
                args_iter = ground_op_filter.get(action, [])
            else:
                args_iter = _enumerate_args(action, model, objects)
            for args in args_iter:
                pre, add, rem = _ground_action_set(action, model, args)
                if ground_op_oracle is not None:
                    op = ground_op_oracle.get((action, args))
                    if op is None or not op.applicable(s):
                        continue
                    ns = op.apply(s)
                else:
                    if not _applicable(s, pre):
                        continue
                    ns = _apply(s, add, rem)
                transitions[(s, action, args)] = ns
                if ns not in reachable:
                    reachable.add(ns)
                    queue.append(ns)

    # Goal states.
    goal_states = {s for s in reachable if goal_predicates <= s}
    if stats is not None:
        stats.setdefault("cap_hit", False)
        stats["n_reachable"] = len(reachable)
        stats["n_transitions"] = len(transitions)
        stats["n_goal_states"] = len(goal_states)
    if not goal_states:
        if stats is not None:
            stats["fail_reason"] = "no_goal_state_in_reachable"
        return None

    # Reverse BFS for distance to goal.
    rev: dict[State, list[State]] = {}
    for (s, a, args), ns in transitions.items():
        rev.setdefault(ns, []).append(s)
    dist: dict[State, int] = {g: 0 for g in goal_states}
    q = deque(goal_states)
    while q:
        ns = q.popleft()
        for s in rev.get(ns, []):
            if s not in dist:
                dist[s] = dist[ns] + 1
                q.append(s)
    if initial_state not in dist:
        if stats is not None:
            stats["fail_reason"] = "s0_not_connected_to_goal"
        return None
    shortest = dist[initial_state]
    budget = (max_length
              if max_length is not None else shortest + max_extra)
    if stats is not None:
        stats["shortest"] = shortest
        stats["budget"] = budget

    # Forward BFS over (state, budget).
    state_id: dict[State, int] = {}
    def _name(s: State, b: int) -> str:
        if s not in state_id:
            state_id[s] = len(state_id)
        return f"s{state_id[s]}_b{b}"

    productions: list[str] = []
    productions.append('root ::= "{\\"plan\\":[" action_seq "]}"')
    productions.append(f"action_seq ::= {_name(initial_state, budget)}")

    visited: set = set()
    q2 = deque([(initial_state, budget)])
    while q2:
        s, b = q2.popleft()
        if (s, b) in visited:
            continue
        visited.add((s, b))
        if b == 0:
            continue
        rules: list[str] = []
        # A goal state admits the empty plan; otherwise an instance whose
        # initial state satisfies the goal gets a non-terminating grammar.
        if s in goal_states:
            rules.append('""')
        for action, model in action_models.items():
            if ground_op_filter is not None:
                args_iter = ground_op_filter.get(action, [])
            else:
                args_iter = _enumerate_args(action, model, objects)
            for args in args_iter:
                pre, add, rem = _ground_action_set(action, model, args)
                if ground_op_oracle is not None:
                    op = ground_op_oracle.get((action, args))
                    if op is None or not op.applicable(s):
                        continue
                else:
                    if not _applicable(s, pre):
                        continue
                ns = transitions.get((s, action, args))
                if ns is None:
                    continue
                ns_b = b - 1
                tok = _action_to_json(action, args)
                if ns in goal_states:
                    rules.append(f'"\\"{tok}\\""')
                elif ns_b > 0 and dist.get(ns, 10**9) <= ns_b:
                    rules.append(
                        f'"\\"{tok}\\"," {_name(ns, ns_b)}'
                    )
                    q2.append((ns, ns_b))
        if not rules:
            # No transition reaches the goal within the budget; an empty
            # production keeps the name, already referenced upstream,
            # defined.
            productions.append(f'{_name(s, b)} ::= ""')
            continue
        productions.append(f"{_name(s, b)} ::= {' | '.join(rules)}")

    return "\n".join(productions)


__all__ = ["compile_from_action_models",
            "compile_from_action_models_astar"]
