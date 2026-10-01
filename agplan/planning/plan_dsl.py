"""Typed BabyAI-style plan DSL with JSON parser.

The parser supports two action vocabularies: the base `Action` enum,
and base+macros via an optional `MacroLibrary` argument. Macros expand
to base-action sequences at parse time, so downstream consumers
(verifier, scoring) see only base actions in `Plan.flat_actions()`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from agplan.planning.macro import MacroLibrary


class Action(str, Enum):
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    MOVE_FORWARD = "move_forward"
    PICKUP = "pickup"
    DROP = "drop"
    TOGGLE = "toggle"
    DONE = "done"


@dataclass(frozen=True)
class Subgoal:
    name: str
    actions: tuple[Action, ...]


@dataclass(frozen=True)
class Plan:
    subgoals: tuple[Subgoal, ...]

    def flat_actions(self) -> tuple[Action, ...]:
        out: list[Action] = []
        for sg in self.subgoals:
            out.extend(sg.actions)
        return tuple(out)


class PlanParseError(ValueError):
    pass


def parse_plan(raw: str, macros: Optional["MacroLibrary"] = None) -> Plan:
    """Parse a JSON-encoded plan into a `Plan`.

    If `macros` is supplied, action tokens that are macro names expand
    to their base-action sequences before being placed in the parsed
    `Subgoal`. Otherwise, every action token must be a base `Action`.
    """
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise PlanParseError(f"invalid JSON: {e}") from e

    if not isinstance(obj, dict) or "subgoals" not in obj:
        raise PlanParseError("missing 'subgoals' key")

    subgoals_raw = obj["subgoals"]
    if not isinstance(subgoals_raw, list):
        raise PlanParseError("'subgoals' must be a list")

    subgoals: list[Subgoal] = []
    for i, sg in enumerate(subgoals_raw):
        if not isinstance(sg, dict):
            raise PlanParseError(f"subgoal[{i}] must be an object")
        name = sg.get("name")
        actions_raw = sg.get("actions")
        if not isinstance(name, str):
            raise PlanParseError(f"subgoal[{i}].name must be a string")
        if not isinstance(actions_raw, list):
            raise PlanParseError(f"subgoal[{i}].actions must be a list")
        try:
            if macros is not None:
                actions: tuple[Action, ...] = tuple()
                for tok in actions_raw:
                    if not isinstance(tok, str):
                        raise ValueError(f"non-string action token: {tok!r}")
                    actions = actions + macros.expand_token(tok)
            else:
                actions = tuple(Action(a) for a in actions_raw)
        except (KeyError, ValueError) as e:
            raise PlanParseError(
                f"subgoal[{i}].actions has invalid action: {e}"
            ) from e
        subgoals.append(Subgoal(name=name, actions=actions))

    return Plan(subgoals=tuple(subgoals))


def schema_path() -> Path:
    return Path(__file__).resolve().parent.parent / "grammars" / "babyai_plan_schema.json"


def load_schema_text() -> str:
    return schema_path().read_text()
