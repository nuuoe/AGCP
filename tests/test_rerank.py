import json
import math

from agplan.decoding.rerank import rerank_candidates
from agplan.grammar_learning.adaptor_cache import AdaptorCache
from agplan.grammar_learning.macro_induction import update_cache_from_plan
from agplan.grammar_learning.rule_estimation import fit_action_unigram
from agplan.planning.plan_dsl import Action, Plan, Subgoal


def _plan_json(actions: list[Action]) -> str:
    return json.dumps(
        {"subgoals": [{"name": "x", "actions": [a.value for a in actions]}]}
    )


def _plan(actions: list[Action]) -> Plan:
    return Plan(subgoals=(Subgoal(name="x", actions=tuple(actions)),))


def test_rerank_prefers_in_distribution_candidate_pcfg_only():
    train = [_plan([Action.MOVE_FORWARD] * 10), _plan([Action.PICKUP])]
    prior = fit_action_unigram(train, alpha=0.1)
    cands = [
        _plan_json([Action.MOVE_FORWARD, Action.MOVE_FORWARD]),
        _plan_json([Action.PICKUP, Action.PICKUP, Action.PICKUP]),
    ]
    ranked = rerank_candidates(cands, pcfg=prior, lambda_pcfg=1.0)
    assert ranked[0].plan is not None
    assert ranked[0].plan.flat_actions() == (Action.MOVE_FORWARD, Action.MOVE_FORWARD)


def test_unparseable_candidates_rank_last():
    train = [_plan([Action.MOVE_FORWARD])]
    prior = fit_action_unigram(train)
    cands = [
        _plan_json([Action.MOVE_FORWARD]),
        "{not json",
        '{"subgoals": [{"name": "x", "actions": ["fly"]}]}',
    ]
    ranked = rerank_candidates(cands, pcfg=prior)
    assert ranked[0].plan is not None
    assert ranked[1].total_score == -math.inf
    assert ranked[2].total_score == -math.inf
    assert ranked[1].parse_error is not None
    assert ranked[2].parse_error is not None


def test_base_logprobs_break_ties():
    train = [_plan([Action.MOVE_FORWARD])]
    prior = fit_action_unigram(train, alpha=0.0)
    cands = [_plan_json([Action.MOVE_FORWARD]), _plan_json([Action.MOVE_FORWARD])]
    ranked = rerank_candidates(cands, pcfg=prior, base_logprobs=[-2.0, -1.0])
    assert ranked[0].raw == cands[1]


def test_adaptor_only_rerank_prefers_cached_macro():
    cache = AdaptorCache()
    template = _plan([Action.TURN_LEFT, Action.MOVE_FORWARD, Action.PICKUP])
    update_cache_from_plan(cache, template, success=True, min_n=2, max_n=3)
    cands = [
        _plan_json([Action.TURN_LEFT, Action.MOVE_FORWARD, Action.PICKUP]),
        _plan_json([Action.TOGGLE, Action.DROP, Action.TOGGLE]),
    ]
    ranked = rerank_candidates(cands, cache=cache, lambda_adaptor=1.0)
    assert ranked[0].plan is not None
    assert ranked[0].plan.flat_actions() == (
        Action.TURN_LEFT,
        Action.MOVE_FORWARD,
        Action.PICKUP,
    )


def test_no_priors_returns_zeros_and_preserves_input_order_via_base():
    cands = [_plan_json([Action.MOVE_FORWARD]), _plan_json([Action.PICKUP])]
    ranked = rerank_candidates(cands, base_logprobs=[-1.0, -2.0])
    assert ranked[0].pcfg_score == 0.0
    assert ranked[0].cache_score == 0.0
    assert ranked[0].total_score == -1.0
