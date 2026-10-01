import pytest

from agplan.planning.plan_dsl import (
    Action,
    Plan,
    PlanParseError,
    Subgoal,
    parse_plan,
)


def test_parse_simple_plan():
    raw = '{"subgoals": [{"name": "go_to_key", "actions": ["turn_left", "move_forward", "pickup"]}]}'
    plan = parse_plan(raw)
    assert len(plan.subgoals) == 1
    assert plan.subgoals[0].name == "go_to_key"
    assert plan.subgoals[0].actions == (
        Action.TURN_LEFT,
        Action.MOVE_FORWARD,
        Action.PICKUP,
    )


def test_parse_empty_subgoals():
    plan = parse_plan('{"subgoals": []}')
    assert plan.subgoals == ()


def test_flat_actions_concatenates_in_order():
    raw = (
        '{"subgoals": ['
        '{"name": "a", "actions": ["turn_left"]},'
        '{"name": "b", "actions": ["move_forward", "done"]}'
        "]}"
    )
    plan = parse_plan(raw)
    assert plan.flat_actions() == (
        Action.TURN_LEFT,
        Action.MOVE_FORWARD,
        Action.DONE,
    )


def test_parse_invalid_json_raises():
    with pytest.raises(PlanParseError):
        parse_plan("{not json}")


def test_parse_missing_subgoals_key_raises():
    with pytest.raises(PlanParseError, match="subgoals"):
        parse_plan('{"plan": []}')


def test_parse_invalid_action_raises():
    with pytest.raises(PlanParseError, match="invalid action"):
        parse_plan('{"subgoals": [{"name": "x", "actions": ["fly"]}]}')


def test_parse_non_string_name_raises():
    with pytest.raises(PlanParseError, match="name"):
        parse_plan('{"subgoals": [{"name": 1, "actions": []}]}')
