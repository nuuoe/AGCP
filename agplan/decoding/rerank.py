"""Post-hoc reranking of CFG-constrained candidates by PCFG / cache priors."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from agplan.grammar_learning.adaptor_cache import AdaptorCache
from agplan.grammar_learning.macro_induction import score_plan_by_cache
from agplan.grammar_learning.pcfg import PCFGPrior
from agplan.grammar_learning.rule_estimation import score_plan
from agplan.planning.plan_dsl import Plan, PlanParseError, parse_plan


@dataclass
class Scored:
    raw: str
    plan: Optional[Plan]
    parse_error: Optional[str]
    pcfg_score: float
    cache_score: float
    total_score: float
    # If set (e.g. by the verifier-pick condition), holds the env
    # execution result attached during scoring so main() does not have
    # to re-run the verifier.
    cached_verifier_result: object = None


def rerank_candidates(
    candidates: list[str],
    pcfg: Optional[PCFGPrior] = None,
    lambda_pcfg: float = 1.0,
    cache: Optional[AdaptorCache] = None,
    lambda_adaptor: float = 1.0,
    base_logprobs: Optional[list[float]] = None,
    macro_min_n: int = 2,
    macro_max_n: int = 6,
    length_normalize: bool = False,
) -> list[Scored]:
    """Rank candidates by

        base + lambda_pcfg * pcfg_score + lambda_adaptor * cache_score

    Either prior may be None (then its term is 0). Unparseable candidates
    get total_score = -inf so they sort last. Returns sorted descending
    by total_score (best first).
    """
    if base_logprobs is None:
        base_logprobs = [0.0] * len(candidates)
    if len(base_logprobs) != len(candidates):
        raise ValueError("base_logprobs length must match candidates length")

    scored: list[Scored] = []
    for raw, base in zip(candidates, base_logprobs):
        try:
            plan = parse_plan(raw)
            pcfg_s = (
                score_plan(plan, pcfg, length_normalize=length_normalize)
                if pcfg is not None else 0.0
            )
            cache_s = (
                score_plan_by_cache(plan, cache, min_n=macro_min_n, max_n=macro_max_n,
                                    length_normalize=length_normalize)
                if cache is not None else 0.0
            )
            scored.append(
                Scored(
                    raw=raw,
                    plan=plan,
                    parse_error=None,
                    pcfg_score=pcfg_s,
                    cache_score=cache_s,
                    total_score=base + lambda_pcfg * pcfg_s + lambda_adaptor * cache_s,
                )
            )
        except PlanParseError as e:
            scored.append(
                Scored(
                    raw=raw,
                    plan=None,
                    parse_error=str(e),
                    pcfg_score=-math.inf,
                    cache_score=-math.inf,
                    total_score=-math.inf,
                )
            )

    scored.sort(key=lambda s: s.total_score, reverse=True)
    return scored
