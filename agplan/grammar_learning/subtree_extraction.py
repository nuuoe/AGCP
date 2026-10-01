"""Extract reusable plan fragments (action n-grams) from parsed plans.

Fragments are contiguous flat-action n-grams with lengths in a window
[min_n, max_n]; subgoal boundaries are ignored.
"""
from __future__ import annotations

from agplan.planning.plan_dsl import Plan


def extract_action_ngrams(
    plan: Plan,
    min_n: int = 2,
    max_n: int = 6,
) -> list[tuple[str, ...]]:
    """Return every contiguous action n-gram of length in [min_n, max_n].

    Returns string-tuples so they are directly usable as keys in
    `AdaptorCache.fragments`. Order preserved: shorter fragments first,
    then position-wise within each length.
    """
    if min_n < 1 or max_n < min_n:
        raise ValueError(f"need 1 <= min_n <= max_n, got {min_n}, {max_n}")
    actions = tuple(a.value for a in plan.flat_actions())
    out: list[tuple[str, ...]] = []
    for n in range(min_n, max_n + 1):
        if n > len(actions):
            break
        for i in range(len(actions) - n + 1):
            out.append(actions[i : i + n])
    return out
