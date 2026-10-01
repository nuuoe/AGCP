"""Per-task applicability grammar for BabyAI GoToRedBall.

Enumerates the (x, y, dir) states reachable from the initial state in
MiniGrid, adds a production for every transition and a terminating
production for every transition into a goal state (agent adjacent to the
red ball and facing it), and wraps the result in the JSON plan schema.
Any sequence admitted by the mask reaches the goal, the constructive form
of Propositions 1 and 2 (App. B) in the paper.
"""
from __future__ import annotations

from collections import deque
from typing import Optional, Any


# direction encoding matches MiniGrid: 0=east, 1=south, 2=west, 3=north
_DIR_OFFSET = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}


def _is_passable(grid, x: int, y: int) -> bool:
    if x < 0 or y < 0 or x >= grid.width or y >= grid.height:
        return False
    cell = grid.get(x, y)
    if cell is None:
        return True
    if cell.type == "wall":
        return False
    if cell.type == "lava":
        return False
    # ball/box/key are obstacles the agent can face but cannot stand on
    if cell.type in ("ball", "box", "key"):
        return False
    if cell.type == "door" and not getattr(cell, "is_open", False):
        return False
    return True


def _step(grid, x: int, y: int, d: int, action: str) -> Optional[tuple[int, int, int]]:
    """Apply an action to (x, y, d). Return new state or None if invalid/no-op-illegal."""
    if action == "turn_left":
        return (x, y, (d - 1) % 4)
    if action == "turn_right":
        return (x, y, (d + 1) % 4)
    if action == "move_forward":
        dx, dy = _DIR_OFFSET[d]
        nx, ny = x + dx, y + dy
        if _is_passable(grid, nx, ny):
            return (nx, ny, d)
        return None
    return None


def _find_red_ball(grid) -> Optional[tuple[int, int]]:
    for j in range(grid.height):
        for i in range(grid.width):
            cell = grid.get(i, j)
            if cell is not None and cell.type == "ball" and cell.color == "red":
                return (i, j)
    return None


# Mission parsing: determine the target object(s) from the mission
# text. BabyAI mission strings follow patterns like:
#   "go to the red ball"
#   "go to a key"
#   "pick up the blue box on your right"
#   "open the green door"
# The parser extracts object type, optional color, and the terminating
# action.

_TYPES = ("ball", "box", "key", "door")
_COLORS = ("red", "green", "blue", "purple", "yellow", "grey", "gray", "white")


def _parse_mission(mission: str) -> dict:
    """Return a dict {action: 'goto'|'pickup'|'open',
                       type: str|None, color: str|None}."""
    m = mission.lower()
    if m.startswith("go to"):
        action = "goto"
    elif m.startswith("pick up"):
        action = "pickup"
    elif m.startswith("open"):
        action = "open"
    else:
        action = "goto"  # default
    obj_type = None
    for t in _TYPES:
        if t in m:
            obj_type = t
            break
    obj_color = None
    for c in _COLORS:
        if c in m:
            obj_color = c if c != "gray" else "grey"
            break
    return {"action": action, "type": obj_type, "color": obj_color}


def _find_targets(grid, type_: Optional[str], color: Optional[str]) -> list[tuple[int, int]]:
    """Find all (x, y) positions of cells matching type/color."""
    targets = []
    for j in range(grid.height):
        for i in range(grid.width):
            cell = grid.get(i, j)
            if cell is None:
                continue
            if type_ is not None and cell.type != type_:
                continue
            if color is not None and cell.color != color:
                continue
            targets.append((i, j))
    return targets


def _goal_states(grid, target_pos: tuple[int, int]) -> set[tuple[int, int, int]]:
    """States where agent is adjacent to target and facing it."""
    goals: set[tuple[int, int, int]] = set()
    tx, ty = target_pos
    for d, (dx, dy) in _DIR_OFFSET.items():
        agent_x, agent_y = tx - dx, ty - dy
        if 0 <= agent_x < grid.width and 0 <= agent_y < grid.height:
            cell = grid.get(agent_x, agent_y)
            if cell is None:  # passable empty cell
                goals.add((agent_x, agent_y, d))
    return goals


def _reachable_states(
    grid, init: tuple[int, int, int], actions: tuple[str, ...]
) -> tuple[set[tuple[int, int, int]], dict]:
    """BFS from init under given action set. Return (reachable, transitions)."""
    reachable = {init}
    transitions: dict[tuple[tuple[int, int, int], str], tuple[int, int, int]] = {}
    q = deque([init])
    while q:
        s = q.popleft()
        for a in actions:
            ns = _step(grid, *s, a)
            if ns is None:
                continue
            transitions[(s, a)] = ns
            if ns not in reachable:
                reachable.add(ns)
                q.append(ns)
    return reachable, transitions


def _bfs_distance(
    init: tuple[int, int, int],
    goals: set[tuple[int, int, int]],
    transitions: dict,
) -> dict[tuple[int, int, int], int]:
    """Return min steps from each reachable state to any goal state."""
    rev_adj: dict[tuple[int, int, int], list] = {}
    for (s, a), ns in transitions.items():
        rev_adj.setdefault(ns, []).append(s)
    dist = {g: 0 for g in goals}
    q = deque(goals)
    while q:
        ns = q.popleft()
        for s in rev_adj.get(ns, []):
            if s not in dist:
                dist[s] = dist[ns] + 1
                q.append(s)
    return dist


def compile_gotoredball_grammar(env: Any, max_extra: int = 8) -> Optional[str]:
    """Build an EBNF for GoToRedBall from the current env state.

    The grammar admits action sequences from init to a goal-adjacent
    facing state, restricted to plans of length at most
    (shortest-path + max_extra). This bounds plan length so the LLM
    cannot wander arbitrarily; without this bound the grammar admits
    paths of unbounded length and the LLM exhausts its token budget.

    Returns None if no red ball found (env not solvable as written).
    Returns the EBNF text otherwise.
    """
    u = env.unwrapped
    grid = u.grid
    init = (int(u.agent_pos[0]), int(u.agent_pos[1]), int(u.agent_dir))
    target = _find_red_ball(grid)
    if target is None:
        return None
    goals = _goal_states(grid, target)
    actions = ("turn_left", "turn_right", "move_forward")
    reachable, trans = _reachable_states(grid, init, actions)
    if not (goals & reachable):
        return None  # goal unreachable
    # Compute distance from each reachable state to nearest goal.
    dist = _bfs_distance(init, goals & reachable, trans)
    shortest = dist.get(init)
    if shortest is None:
        return None  # no path
    max_len = shortest + max_extra
    # Build (state, remaining_budget) BFS forward to enumerate states
    # that participate in any path of length <= max_len.
    # State here: (pos, dir, steps_taken). Productions only continue
    # while steps_taken + dist(ns) <= max_len.
    # The budget is baked into the production names directly.

    # Length-budgeted state names: s{x}_{y}_{d}_b{budget_remaining}
    def name(s: tuple[int, int, int], budget: int) -> str:
        return f"s{s[0]}_{s[1]}_{s[2]}_b{budget}"

    productions: list[str] = []
    productions.append(
        'root ::= "{\\"subgoals\\":[" subgoal "]}"'
    )
    productions.append(
        'subgoal ::= "{\\"name\\":\\"go\\",\\"actions\\":[" action_seq "]}"'
    )
    productions.append(f"action_seq ::= {name(init, max_len)}")

    # BFS over (state, budget) to enumerate productions reachable
    # within budget that can still reach goal.
    visited: set[tuple[tuple[int, int, int], int]] = set()
    q = deque([(init, max_len)])
    while q:
        s, b = q.popleft()
        if (s, b) in visited:
            continue
        visited.add((s, b))
        if b == 0:
            # No budget left; only terminating productions allowed,
            # but a terminating production needs at least 1 action.
            # So a state with budget=0 has no productions (dead).
            continue
        rules: list[str] = []
        for a in actions:
            ns = trans.get((s, a))
            if ns is None:
                continue
            ns_budget = b - 1
            if ns in goals:
                # Terminating production: emit this action and stop.
                # Goal states do not continue (env auto-terminates).
                rules.append(f'"\\"{a}\\""')
            elif ns_budget > 0 and dist.get(ns, 10**9) <= ns_budget:
                # Continuation: ns is non-goal and there's a path
                # forward from ns to a goal within remaining budget.
                rules.append(f'"\\"{a}\\"," {name(ns, ns_budget)}')
                q.append((ns, ns_budget))
        if not rules:
            continue
        productions.append(f"{name(s, b)} ::= {' | '.join(rules)}")

    return "\n".join(productions)


def _get_targets_from_env(env: Any) -> Optional[list[tuple[int, int]]]:
    """Try to extract target object positions from the env's
    instruction object. BabyAI envs store the verified target list
    as `env.unwrapped.instrs.desc.obj_poss`.

    This works for instructions with spatial constraints
    (``in front of you'', ``behind you'', ``on your left'',
    ``on your right'') which the mission-text parser cannot resolve
    on its own. Returns None if the attribute is unavailable.
    """
    u = env.unwrapped
    if not hasattr(u, "instrs"):
        return None
    instrs = u.instrs
    desc = getattr(instrs, "desc", None)
    if desc is None:
        return None
    poss = getattr(desc, "obj_poss", None)
    if poss is None:
        return None
    return [(int(p[0]), int(p[1])) for p in poss]


def compile_env_grammar(env: Any, mission: str, max_extra: int = 8) -> Optional[str]:
    """General BabyAI G_env compiler. Parses mission text to extract
    target object and terminating action, then builds an EBNF that
    admits only goal-achieving plans.

    Supports:
      - 'go to X'    : goal = adjacent + facing X (no extra action)
      - 'pick up X'  : goal = adjacent + facing X + final pickup action
      - 'open X'     : goal = adjacent + facing door X + final toggle

    For missions with spatial constraints (``in front of you'', etc.)
    we use env.instrs.desc.obj_poss as the authoritative target list,
    falling back to mission-text type/color matching otherwise.
    """
    spec = _parse_mission(mission)
    u = env.unwrapped
    grid = u.grid
    init = (int(u.agent_pos[0]), int(u.agent_pos[1]), int(u.agent_dir))
    # Prefer env-supplied target positions when available -- these are
    # the positions the env's verifier actually checks against, and
    # encode spatial constraints the mission-text parser can't.
    env_targets = _get_targets_from_env(env)
    if env_targets:
        targets = env_targets
    else:
        targets = _find_targets(grid, spec["type"], spec["color"])
    if not targets:
        return None
    # All adjacent-facing positions across all matching targets.
    all_goal_states: set[tuple[int, int, int]] = set()
    for t in targets:
        all_goal_states.update(_goal_states(grid, t))
    if not all_goal_states:
        return None

    nav_actions = ("turn_left", "turn_right", "move_forward")
    reachable, trans = _reachable_states(grid, init, nav_actions)
    nav_goals = all_goal_states & reachable
    if not nav_goals:
        return None
    dist = _bfs_distance(init, nav_goals, trans)
    shortest = dist.get(init)
    if shortest is None:
        return None
    max_len = shortest + max_extra

    def name(s: tuple[int, int, int], budget: int) -> str:
        return f"s{s[0]}_{s[1]}_{s[2]}_b{budget}"

    productions: list[str] = []
    productions.append(
        'root ::= "{\\"subgoals\\":[" subgoal "]}"'
    )
    productions.append(
        'subgoal ::= "{\\"name\\":\\"go\\",\\"actions\\":[" action_seq "]}"'
    )
    productions.append(f"action_seq ::= {name(init, max_len)}")

    # Terminating action depends on mission type:
    # goto: when next state is goal-state, terminating production
    #       emits the navigation action without continuation.
    # pickup: when next state is goal-state, terminating must emit
    #         the navigation action AND a pickup action after.
    # open:   similarly, navigation action then toggle.
    term_action = spec["action"]

    visited: set[tuple[tuple[int, int, int], int]] = set()
    q = deque([(init, max_len)])
    while q:
        s, b = q.popleft()
        if (s, b) in visited:
            continue
        visited.add((s, b))
        if b == 0:
            continue
        rules: list[str] = []
        for a in nav_actions:
            ns = trans.get((s, a))
            if ns is None:
                continue
            ns_budget = b - 1
            if ns in nav_goals:
                # Goal reached after this action. Terminating production
                # depends on mission type.
                if term_action == "goto":
                    rules.append(f'"\\"{a}\\""')
                elif term_action == "pickup":
                    rules.append(f'"\\"{a}\\",\\"pickup\\""')
                elif term_action == "open":
                    rules.append(f'"\\"{a}\\",\\"toggle\\""')
            elif ns_budget > 0 and dist.get(ns, 10**9) <= ns_budget:
                rules.append(f'"\\"{a}\\"," {name(ns, ns_budget)}')
                q.append((ns, ns_budget))
        if not rules:
            continue
        productions.append(f"{name(s, b)} ::= {' | '.join(rules)}")

    return "\n".join(productions)


__all__ = ["compile_gotoredball_grammar", "compile_env_grammar"]
