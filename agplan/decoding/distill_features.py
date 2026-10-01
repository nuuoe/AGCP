"""Per-candidate feature extraction for distilling the verifier signal
into a learned reranker.

Given a parsed `Plan` and the priors fit on demos, return a fixed-length
feature vector. Used both at training time (to fit the model on
verifier-labelled candidates) and at inference time (to score new
candidates without the env).
"""
from __future__ import annotations

from typing import Optional

from agplan.grammar_learning.adaptor_cache import AdaptorCache
from agplan.grammar_learning.macro_induction import (
    score_plan_by_cache,
    score_plan_by_mle,
)
from agplan.grammar_learning.pcfg import PCFGPrior
from agplan.grammar_learning.rule_estimation import score_plan
from agplan.planning.plan_dsl import Action, Plan


# Feature names in fixed order. Keep stable so saved models can be reused.
FEATURE_NAMES: tuple[str, ...] = (
    "plan_length",
    "n_subgoals",
    "n_turn_left",
    "n_turn_right",
    "n_move_forward",
    "n_pickup",
    "n_drop",
    "n_toggle",
    "n_done",
    "ends_with_done",
    "has_toggle",
    "has_pickup",
    "pcfg_mean",
    "cache_mean",
    "mle_mean",
    "pcfg_sum",
    "cache_sum",
)


def featurise(
    plan: Plan,
    pcfg: Optional[PCFGPrior] = None,
    cache: Optional[AdaptorCache] = None,
    macro_min_n: int = 2,
    macro_max_n: int = 4,
) -> list[float]:
    """Return a list[float] in the order of FEATURE_NAMES.

    Missing priors return 0 for the corresponding feature. Empty plans
    return zeros for length-dependent features and 0 for all log-prior
    features (so they sort low under any positive coefficient)."""
    actions = plan.flat_actions()
    n = len(actions)
    counts = {a: 0 for a in Action}
    for a in actions:
        counts[a] += 1

    pcfg_sum = score_plan(plan, pcfg, length_normalize=False) if pcfg else 0.0
    pcfg_mean = score_plan(plan, pcfg, length_normalize=True) if pcfg else 0.0
    cache_sum = (
        score_plan_by_cache(plan, cache, min_n=macro_min_n, max_n=macro_max_n,
                            length_normalize=False, empty_penalty=-50.0)
        if cache else 0.0
    )
    cache_mean = (
        score_plan_by_cache(plan, cache, min_n=macro_min_n, max_n=macro_max_n,
                            length_normalize=True, empty_penalty=-50.0)
        if cache else 0.0
    )
    mle_mean = (
        score_plan_by_mle(plan, cache, min_n=macro_min_n, max_n=macro_max_n,
                          length_normalize=True, empty_penalty=-50.0)
        if cache else 0.0
    )

    return [
        float(n),
        float(len(plan.subgoals)),
        float(counts[Action.TURN_LEFT]),
        float(counts[Action.TURN_RIGHT]),
        float(counts[Action.MOVE_FORWARD]),
        float(counts[Action.PICKUP]),
        float(counts[Action.DROP]),
        float(counts[Action.TOGGLE]),
        float(counts[Action.DONE]),
        float(actions[-1] == Action.DONE if actions else 0),
        float(any(a == Action.TOGGLE for a in actions)),
        float(any(a == Action.PICKUP for a in actions)),
        float(pcfg_mean),
        float(cache_mean),
        float(mle_mean),
        float(pcfg_sum),
        float(cache_sum),
    ]
