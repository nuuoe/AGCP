from minigrid.core.actions import Actions

from agplan.planning.babyai_adapter import (
    from_minigrid,
    to_minigrid,
    to_minigrid_seq,
)
from agplan.planning.plan_dsl import Action


def test_action_mapping_round_trip_for_all_actions():
    for a in Action:
        assert from_minigrid(to_minigrid(a)) == a


def test_minigrid_int_round_trip():
    for mg in Actions:
        assert to_minigrid(from_minigrid(int(mg))) == int(mg)


def test_to_minigrid_specific_values():
    assert to_minigrid(Action.TURN_LEFT) == int(Actions.left)
    assert to_minigrid(Action.MOVE_FORWARD) == int(Actions.forward)
    assert to_minigrid(Action.DONE) == int(Actions.done)


def test_to_minigrid_seq():
    seq = (Action.TURN_LEFT, Action.MOVE_FORWARD, Action.DONE)
    assert to_minigrid_seq(seq) == [int(Actions.left), int(Actions.forward), int(Actions.done)]
