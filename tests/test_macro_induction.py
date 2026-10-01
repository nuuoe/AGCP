from agplan.grammar_learning.adaptor_cache import AdaptorCache
from agplan.grammar_learning.macro_induction import (
    score_plan_by_cache,
    update_cache_from_plan,
    update_cache_from_plans,
)
from agplan.planning.plan_dsl import Action, Plan, Subgoal


def _plan(actions: list[Action]) -> Plan:
    return Plan(subgoals=(Subgoal(name="x", actions=tuple(actions)),))


def test_failed_plan_does_not_update_cache():
    cache = AdaptorCache()
    plan = _plan([Action.TURN_LEFT, Action.MOVE_FORWARD, Action.PICKUP])
    update_cache_from_plan(cache, plan, success=False)
    assert cache.total == 0
    assert cache.fragments == {}


def test_successful_plan_adds_all_ngrams():
    cache = AdaptorCache()
    plan = _plan([Action.TURN_LEFT, Action.MOVE_FORWARD, Action.PICKUP])
    update_cache_from_plan(cache, plan, success=True, min_n=2, max_n=3)
    # 2 bigrams + 1 trigram = 3 fragments
    assert cache.total == 3
    assert len(cache.fragments) == 3


def test_update_from_plans_pairs_with_successes():
    cache = AdaptorCache()
    p1 = _plan([Action.MOVE_FORWARD, Action.PICKUP])
    p2 = _plan([Action.TOGGLE, Action.DONE])
    update_cache_from_plans(cache, [p1, p2], [True, False], min_n=2, max_n=2)
    assert cache.total == 1
    assert ("move_forward", "pickup") in cache.fragments
    assert ("toggle", "done") not in cache.fragments


def test_score_plan_by_cache_prefers_in_cache_plan():
    cache = AdaptorCache()
    in_dist = _plan([Action.MOVE_FORWARD] * 5)
    update_cache_from_plan(cache, in_dist, success=True, min_n=2, max_n=3)
    out_dist = _plan([Action.TOGGLE] * 5)
    s_in = score_plan_by_cache(in_dist, cache, min_n=2, max_n=3)
    s_out = score_plan_by_cache(out_dist, cache, min_n=2, max_n=3)
    assert s_in > s_out


def test_score_empty_plan_is_minus_infinity():
    import math
    cache = AdaptorCache()
    # Default empty_penalty is -inf so the reranker drops empty plans.
    assert score_plan_by_cache(_plan([]), cache) == float("-inf")
    # Caller can opt into a finite floor:
    assert score_plan_by_cache(_plan([]), cache, empty_penalty=0.0) == 0.0
