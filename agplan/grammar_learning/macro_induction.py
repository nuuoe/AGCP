"""Update an adaptor cache with macros from successful plans, and score
new plans under the cache."""
from __future__ import annotations

import math

from agplan.grammar_learning.adaptor_cache import AdaptorCache
from agplan.grammar_learning.subtree_extraction import extract_action_ngrams
from agplan.planning.plan_dsl import Plan


def update_cache_from_plan(
    cache: AdaptorCache,
    plan: Plan,
    success: bool,
    min_n: int = 2,
    max_n: int = 6,
) -> None:
    """Add every n-gram of `plan` to `cache`, only if `success=True`.

    Updates are positive-only: failed plans leave the cache unchanged.
    """
    if not success:
        return
    for fragment in extract_action_ngrams(plan, min_n=min_n, max_n=max_n):
        cache.update(fragment, success=True)


def update_cache_from_plans(
    cache: AdaptorCache,
    plans: list[Plan],
    successes: list[bool],
    min_n: int = 2,
    max_n: int = 6,
) -> None:
    if len(plans) != len(successes):
        raise ValueError("plans and successes must have same length")
    for plan, success in zip(plans, successes):
        update_cache_from_plan(cache, plan, success, min_n, max_n)


def score_plan_by_cache(
    plan: Plan,
    cache: AdaptorCache,
    min_n: int = 2,
    max_n: int = 6,
    length_normalize: bool = False,
    empty_penalty: float = -math.inf,
) -> float:
    """Sum (or mean if length_normalize) of cache log-priors over a plan's n-grams.

    Plans whose fragments are well-represented in the cache score
    higher (less negative). With length_normalize=True we use the
    mean instead of the sum, which removes the implicit short-plan
    preference of the unnormalised score. Plans too short to
    produce any fragment of length min_n receive `empty_penalty`
    (default -inf so the reranker drops them); a finite floor keeps
    them rankable.
    """
    fragments = extract_action_ngrams(plan, min_n=min_n, max_n=max_n)
    if not fragments:
        return empty_penalty
    total = sum(cache.log_prior(f) for f in fragments)
    return total / len(fragments) if length_normalize else total


def score_plan_by_mle(
    plan: Plan,
    cache: AdaptorCache,
    min_n: int = 2,
    max_n: int = 6,
    floor: float = -25.0,
    length_normalize: bool = False,
    empty_penalty: float = -math.inf,
) -> float:
    """Sum (or mean) of unsmoothed maximum-likelihood log-priors over a plan's n-grams.

    For each n-gram f with cache count c_f and total N: log(c_f / N) if
    c_f > 0 else `floor`. The Pitman-Yor discount and concentration are
    ignored. With length_normalize=True we return the mean.
    Plans too short to produce any fragment of length min_n receive
    `empty_penalty` (default -inf so the reranker drops them).
    """
    fragments = extract_action_ngrams(plan, min_n=min_n, max_n=max_n)
    if not fragments:
        return empty_penalty
    if cache.total == 0:
        s = floor * len(fragments)
        return s / len(fragments) if length_normalize else s
    score = 0.0
    for f in fragments:
        c = cache.fragments.get(f, 0)
        if c == 0:
            score += floor
        else:
            score += math.log(c / cache.total)
    return score / len(fragments) if length_normalize else score
