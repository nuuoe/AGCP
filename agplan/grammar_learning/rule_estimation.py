"""Estimate PCFG rule probabilities from a set of plans."""
from __future__ import annotations

from collections.abc import Iterable

from agplan.grammar_learning.pcfg import PCFGPrior
from agplan.planning.plan_dsl import Plan


ACTION_LHS = "action"


def fit_action_unigram(plans: Iterable[Plan], alpha: float = 0.1) -> PCFGPrior:
    """Fit a unigram PCFG over actions: P(action) under LHS 'action'."""
    prior = PCFGPrior(alpha=alpha)
    for plan in plans:
        for action in plan.flat_actions():
            prior.update_rule(ACTION_LHS, (action.value,))
    return prior


def score_plan(
    plan: Plan,
    prior: PCFGPrior,
    length_normalize: bool = False,
    empty_penalty: float = float("-inf"),
) -> float:
    """Sum (or mean) of action log-priors under the action-unigram model.

    Empty plans (zero actions) receive `empty_penalty` (default -inf)
    so the reranker drops them.
    """
    actions = plan.flat_actions()
    if not actions:
        return empty_penalty
    total = sum(prior.logprob(ACTION_LHS, (a.value,)) for a in actions)
    return total / len(actions) if length_normalize else total
