"""Verifier-induced grammar (G_env) for Blocksworld.

Compiles a per-task EBNF whose every accepted plan reaches the goal from
the given state, by the construction of `env_grammar.compile_env_grammar`:
BFS over (state, budget) with state = (support map, held block) and a
goal test on the configuration. The budget is the BFS-shortest plan
length plus max_extra.
"""
from __future__ import annotations

from collections import deque
from typing import Optional


# State = (frozenset of (block, support) pairs, held)
# where each (block, support) pair encodes that `block` sits on
# `support` ('TABLE' or another block).

State = tuple[frozenset[tuple[str, str]], Optional[str]]


def _is_clear(state: State, b: str) -> bool:
    support_set, held = state
    if held == b:
        return False
    for (x, sup) in support_set:
        if sup == b:
            return False
    return True


def _apply_action(state: State, action: tuple) -> Optional[State]:
    """Return new state after action, or None if action invalid.

    Actions are encoded as tuples:
      ('pickup', b)
      ('unstack', b, c)
      ('putdown', b)
      ('stack', b, c)
    """
    support_set, held = state
    support = dict(support_set)
    op = action[0]
    if op == "pickup":
        b = action[1]
        if held is not None: return None
        if support.get(b) != "TABLE": return None
        if not _is_clear(state, b): return None
        del support[b]
        return (frozenset(support.items()), b)
    elif op == "unstack":
        b, c = action[1], action[2]
        if held is not None: return None
        if support.get(b) != c: return None
        if not _is_clear(state, b): return None
        del support[b]
        return (frozenset(support.items()), b)
    elif op == "putdown":
        b = action[1]
        if held != b: return None
        support[b] = "TABLE"
        return (frozenset(support.items()), None)
    elif op == "stack":
        b, c = action[1], action[2]
        if held != b: return None
        if not _is_clear(state, c): return None
        if c == b: return None
        support[b] = c
        return (frozenset(support.items()), None)
    return None


def _all_actions(labels: list[str]) -> list[tuple]:
    """Enumerate every (op, args...) action for given blocks."""
    out = []
    for b in labels:
        out.append(("pickup", b))
        out.append(("putdown", b))
    for b in labels:
        for c in labels:
            if b == c: continue
            out.append(("unstack", b, c))
            out.append(("stack", b, c))
    return out


def _action_to_json(action: tuple, action_name_map: Optional[dict] = None) -> str:
    """JSON encoding of an action, "op(b)" or "op(b,c)", with op renamed
    through action_name_map if given (Mystery Blocksworld)."""
    op = action[0]
    if action_name_map:
        op = action_name_map.get(op, op)
    if action[0] in ("pickup", "putdown"):
        return f"{op}({action[1]})"
    return f"{op}({action[1]},{action[2]})"


def _goal_states(
    goal_pairs: tuple, reachable: set[State], strict: bool = True
) -> set[State]:
    """Reachable states with nothing held whose support map equals
    goal_pairs (strict) or satisfies every (block, support) constraint in
    it (PlanBench partial goals)."""
    goal_dict = dict(goal_pairs)
    goals = set()
    for s in reachable:
        support_set, held = s
        if held is not None:
            continue
        support = dict(support_set)
        if strict:
            if support == goal_dict:
                goals.add(s)
        else:
            if all(support.get(b) == sup for b, sup in goal_pairs):
                goals.add(s)
    return goals


def _bfs_distance_blocksworld(
    init: State, goals: set[State], transitions: dict
) -> dict[State, int]:
    """Reverse BFS from goals to compute shortest steps from each
    reachable state to any goal."""
    rev: dict[State, list[State]] = {}
    for (s, a), ns in transitions.items():
        rev.setdefault(ns, []).append(s)
    dist: dict[State, int] = {g: 0 for g in goals}
    q = deque(goals)
    while q:
        ns = q.popleft()
        for s in rev.get(ns, []):
            if s not in dist:
                dist[s] = dist[ns] + 1
                q.append(s)
    return dist


def _reachable_states(
    init: State, all_actions: list[tuple]
) -> tuple[set[State], dict]:
    reachable = {init}
    transitions: dict[tuple[State, tuple], State] = {}
    q = deque([init])
    while q:
        s = q.popleft()
        for a in all_actions:
            ns = _apply_action(s, a)
            if ns is None:
                continue
            transitions[(s, a)] = ns
            if ns not in reachable:
                reachable.add(ns)
                q.append(ns)
    return reachable, transitions


def _state_name(state: State, budget: int, state_id: dict) -> str:
    if state not in state_id:
        state_id[state] = len(state_id)
    return f"s{state_id[state]}_b{budget}"


def compile_blocksworld_grammar(
    initial_support: dict[str, str],
    goal_pairs: tuple,
    labels: list[str],
    max_extra: int = 4,
    strict_goal: bool = True,
    action_name_map: Optional[dict] = None,
) -> Optional[str]:
    """Compile the EBNF of goal-reaching Blocksworld plans from initial_support.

    initial_support: block -> 'TABLE' or the block below.
    goal_pairs: (block, support) pairs the goal requires.
    max_extra: budget = shortest path + max_extra.
    strict_goal: the whole support map must match goal_pairs; otherwise
                 any state satisfying the listed constraints is a goal.
    """
    init: State = (frozenset(initial_support.items()), None)
    actions = _all_actions(labels)
    reachable, trans = _reachable_states(init, actions)
    goals = _goal_states(goal_pairs, reachable, strict=strict_goal)
    if not goals:
        return None
    dist = _bfs_distance_blocksworld(init, goals, trans)
    if init not in dist:
        return None
    shortest = dist[init]
    max_len = shortest + max_extra

    state_id: dict = {}
    productions: list[str] = []
    productions.append('root ::= "{\\"plan\\":[" action_seq "]}"')
    productions.append(f"action_seq ::= {_state_name(init, max_len, state_id)}")

    # Forward BFS over (state, budget).
    visited: set = set()
    q = deque([(init, max_len)])
    while q:
        s, b = q.popleft()
        if (s, b) in visited:
            continue
        visited.add((s, b))
        if b == 0:
            continue
        rules: list[str] = []
        for a in actions:
            ns = trans.get((s, a))
            if ns is None:
                continue
            ns_budget = b - 1
            if ns in goals:
                rules.append(f'"\\"{_action_to_json(a, action_name_map)}\\""')
            elif ns_budget > 0 and dist.get(ns, 10**9) <= ns_budget:
                rules.append(
                    f'"\\"{_action_to_json(a, action_name_map)}\\"," '
                    f'{_state_name(ns, ns_budget, state_id)}'
                )
                q.append((ns, ns_budget))
        if not rules:
            continue
        productions.append(f"{_state_name(s, b, state_id)} ::= {' | '.join(rules)}")

    return "\n".join(productions)


def shortest_path_length(
    initial_support: dict[str, str],
    goal_pairs: tuple,
    labels: list[str],
    held: Optional[str] = None,
) -> Optional[int]:
    """BFS-shortest plan length under partial-goal semantics, or None if
    the goal is unreachable."""
    init: State = (frozenset(initial_support.items()), held)
    actions = _all_actions(labels)
    reachable, trans = _reachable_states(init, actions)
    goals = _goal_states(goal_pairs, reachable, strict=False)
    if not goals:
        return None
    dist = _bfs_distance_blocksworld(init, goals, trans)
    return dist.get(init)


__all__ = ["compile_blocksworld_grammar", "shortest_path_length"]
