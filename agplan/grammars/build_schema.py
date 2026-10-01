"""Build a JSON-Schema string for the plan DSL with an extended action
vocabulary that includes macros."""
from __future__ import annotations

import json
from typing import Iterable

from agplan.planning.plan_dsl import Action


def build_plan_schema(action_names: Iterable[str]) -> str:
    """Return a JSON-Schema string compatible with our Plan DSL whose
    action enum is the supplied iterable. Use the result with XGrammar's
    `compile_json_schema`.
    """
    schema = {
        "type": "object",
        "properties": {
            "subgoals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "actions": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": list(action_names),
                            },
                        },
                    },
                    "required": ["name", "actions"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["subgoals"],
        "additionalProperties": False,
    }
    return json.dumps(schema)


def base_action_names() -> tuple[str, ...]:
    return tuple(a.value for a in Action)
