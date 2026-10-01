import pytest

from agplan.grammar_learning.subtree_extraction import extract_action_ngrams
from agplan.planning.plan_dsl import Action, Plan, Subgoal


def _plan(actions: list[Action]) -> Plan:
    return Plan(subgoals=(Subgoal(name="x", actions=tuple(actions)),))


def test_empty_plan_yields_no_ngrams():
    assert extract_action_ngrams(_plan([])) == []


def test_plan_shorter_than_min_n_yields_no_ngrams():
    assert extract_action_ngrams(_plan([Action.MOVE_FORWARD]), min_n=2) == []


def test_bigrams_count_matches_window():
    p = _plan([Action.TURN_LEFT, Action.MOVE_FORWARD, Action.PICKUP])
    bigrams = extract_action_ngrams(p, min_n=2, max_n=2)
    assert bigrams == [
        ("turn_left", "move_forward"),
        ("move_forward", "pickup"),
    ]


def test_bigrams_and_trigrams_concatenate():
    p = _plan([Action.TURN_LEFT, Action.MOVE_FORWARD, Action.PICKUP, Action.DONE])
    frags = extract_action_ngrams(p, min_n=2, max_n=3)
    # 3 bigrams + 2 trigrams
    assert len(frags) == 5
    assert all(len(f) in (2, 3) for f in frags)


def test_invalid_window_raises():
    with pytest.raises(ValueError):
        extract_action_ngrams(_plan([Action.MOVE_FORWARD]), min_n=3, max_n=2)
