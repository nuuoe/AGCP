import math

from agplan.grammar_learning.pcfg import PCFGPrior
from agplan.grammar_learning.rule_estimation import (
    ACTION_LHS,
    fit_action_unigram,
    score_plan,
)
from agplan.planning.plan_dsl import Action, Plan, Subgoal


def _plan(actions: list[Action]) -> Plan:
    return Plan(subgoals=(Subgoal(name="x", actions=tuple(actions)),))


def test_fit_unigram_counts_actions():
    plans = [
        _plan([Action.MOVE_FORWARD, Action.MOVE_FORWARD, Action.PICKUP]),
        _plan([Action.MOVE_FORWARD, Action.DONE]),
    ]
    prior = fit_action_unigram(plans, alpha=0.0)
    # MOVE_FORWARD x3, PICKUP x1, DONE x1, total 5
    assert prior.counts[(ACTION_LHS, ("move_forward",))] == 3
    assert prior.counts[(ACTION_LHS, ("pickup",))] == 1
    assert prior.counts[(ACTION_LHS, ("done",))] == 1
    assert prior.totals[ACTION_LHS] == 5


def test_score_plan_higher_when_actions_match_prior():
    train = [_plan([Action.MOVE_FORWARD] * 10), _plan([Action.PICKUP])]
    prior = fit_action_unigram(train, alpha=0.1)
    matching = _plan([Action.MOVE_FORWARD] * 3)
    off_distribution = _plan([Action.PICKUP, Action.PICKUP, Action.PICKUP])
    assert score_plan(matching, prior) > score_plan(off_distribution, prior)


def test_score_plan_finite_for_unseen_action():
    prior = PCFGPrior(alpha=0.5)
    prior.update_rule(ACTION_LHS, ("move_forward",))
    plan = _plan([Action.PICKUP])
    s = score_plan(plan, prior)
    assert math.isfinite(s)
