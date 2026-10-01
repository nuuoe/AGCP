"""Macros: named action subroutines that the LLM can emit as one token.

A `Macro` is a name plus a tuple of base `Action`s it expands to. A
`MacroLibrary` is a collection of named macros; it knows how to expand a
mixed plan (base actions + macro names) into a flat base-action sequence
for execution by the verifier.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from agplan.planning.plan_dsl import Action


@dataclass(frozen=True)
class Macro:
    name: str
    expansion: tuple[Action, ...]

    def __post_init__(self) -> None:
        if not re.fullmatch(r"m_[a-z0-9_]+", self.name):
            raise ValueError(
                f"macro name must match m_[a-z0-9_]+, got {self.name!r}"
            )
        if not self.expansion:
            raise ValueError(f"macro {self.name} has empty expansion")


class MacroLibrary:
    """Collection of macros; lookup by name; expand mixed action lists."""

    def __init__(self, macros: Iterable[Macro] = ()) -> None:
        self._by_name: dict[str, Macro] = {}
        for m in macros:
            self.add(m)

    def add(self, macro: Macro) -> None:
        if macro.name in self._by_name:
            raise ValueError(f"duplicate macro name: {macro.name}")
        self._by_name[macro.name] = macro

    def __len__(self) -> int:
        return len(self._by_name)

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def names(self) -> tuple[str, ...]:
        return tuple(self._by_name.keys())

    def get(self, name: str) -> Macro:
        return self._by_name[name]

    def expand_token(self, token: str) -> tuple[Action, ...]:
        """Expand a single action-or-macro token into base actions.

        Raises KeyError if the token is neither a known macro nor a base
        Action value.
        """
        if token in self._by_name:
            return self._by_name[token].expansion
        return (Action(token),)

    def expand(self, tokens: Iterable[str]) -> tuple[Action, ...]:
        out: list[Action] = []
        for t in tokens:
            out.extend(self.expand_token(t))
        return tuple(out)

    def vocabulary(self) -> tuple[str, ...]:
        """Base actions + macro names. Order: base actions first, then
        macros in insertion order."""
        return tuple(a.value for a in Action) + tuple(self._by_name.keys())


def name_from_actions(actions: tuple[Action, ...]) -> str:
    """Generate a stable, descriptive macro name from an action sequence.

    Uses an abbreviation per action (`l` `r` `f` `p` `d` `t` `o`) and
    runs of the same action collapse to `<count><letter>`. So
    (move_forward, move_forward, move_forward) -> "m_3f", and
    (turn_left, move_forward) -> "m_l1f".
    """
    abbr = {
        Action.TURN_LEFT: "l",
        Action.TURN_RIGHT: "r",
        Action.MOVE_FORWARD: "f",
        Action.PICKUP: "p",
        Action.DROP: "d",
        Action.TOGGLE: "t",
        Action.DONE: "o",  # `done` -> 'o' to keep `d` for `drop`
    }
    if not actions:
        raise ValueError("cannot name empty macro")
    parts: list[str] = []
    run_letter = abbr[actions[0]]
    run_count = 1
    for a in actions[1:]:
        letter = abbr[a]
        if letter == run_letter:
            run_count += 1
        else:
            parts.append(f"{run_count if run_count > 1 else ''}{run_letter}")
            run_letter = letter
            run_count = 1
    parts.append(f"{run_count if run_count > 1 else ''}{run_letter}")
    return "m_" + "".join(parts)
