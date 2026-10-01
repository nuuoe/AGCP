"""Action conversion + grid-state rendering between agplan and MiniGrid."""
from __future__ import annotations

from typing import Any

from minigrid.core.actions import Actions

from agplan.planning.plan_dsl import Action


_DIR_CHAR = {0: ">", 1: "v", 2: "<", 3: "^"}
_DIR_NAME = {0: "east", 1: "south", 2: "west", 3: "north"}

# Two-character type prefixes for grid rendering, so that ball ('Ba')
# and box ('Bx') stay distinguishable.
_TYPE_PREFIX = {
    "wall": "Wa",
    "floor": "Fl",
    "ball": "Ba",
    "box": "Bx",
    "key": "Ke",
    "door": "Do",
    "goal": "Go",
    "lava": "La",
}


_AGPLAN_TO_MINIGRID: dict[Action, Actions] = {
    Action.TURN_LEFT:    Actions.left,
    Action.TURN_RIGHT:   Actions.right,
    Action.MOVE_FORWARD: Actions.forward,
    Action.PICKUP:       Actions.pickup,
    Action.DROP:         Actions.drop,
    Action.TOGGLE:       Actions.toggle,
    Action.DONE:         Actions.done,
}

_MINIGRID_TO_AGPLAN: dict[Actions, Action] = {v: k for k, v in _AGPLAN_TO_MINIGRID.items()}


def to_minigrid(action: Action) -> int:
    """agplan Action → MiniGrid action int."""
    return int(_AGPLAN_TO_MINIGRID[action])


def from_minigrid(action_int: int) -> Action:
    """MiniGrid action int → agplan Action."""
    return _MINIGRID_TO_AGPLAN[Actions(action_int)]


def to_minigrid_seq(actions: tuple[Action, ...]) -> list[int]:
    return [to_minigrid(a) for a in actions]


def render_grid_text(env: Any) -> str:
    """Return a compact ASCII rendering of the full MiniGrid state.

    Each non-empty cell is `<2-char-type><1-char-color>`:
      Ba=ball, Bx=box, Ke=key, Do=door, Wa=wall, Fl=floor,
      Go=goal, La=lava
    plus a single character for the color (r=red, g=green, b=blue,
    p=purple, y=yellow, w=white, s=grey, etc.).
    Empty cells are `.`. The agent is `> v < ^` by direction.
    Rows are space-separated, one per line.
    """
    u = env.unwrapped
    agent_pos = (int(u.agent_pos[0]), int(u.agent_pos[1]))
    rows: list[str] = []
    for j in range(u.grid.height):
        row: list[str] = []
        for i in range(u.grid.width):
            if (i, j) == agent_pos:
                row.append(_DIR_CHAR[u.agent_dir])
                continue
            cell = u.grid.get(i, j)
            if cell is None:
                row.append(".")
            else:
                pre = _TYPE_PREFIX.get(cell.type, cell.type[:2].title())
                # Use 's' for 'grey' rather than 'g' to free up 'g' for green
                color_short = "s" if cell.color == "grey" else cell.color[0]
                row.append(f"{pre}{color_short}")
        rows.append(" ".join(row))
    return "\n".join(rows)


def agent_state_text(env: Any) -> str:
    """One-line description of the agent's current pose."""
    u = env.unwrapped
    pos = (int(u.agent_pos[0]), int(u.agent_pos[1]))
    return f"Agent at (x={pos[0]}, y={pos[1]}) facing {_DIR_NAME[u.agent_dir]}."
