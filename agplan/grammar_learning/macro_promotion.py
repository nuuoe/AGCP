"""Promote top-K cache fragments into a MacroLibrary.

Used as the static macro-selection step in the dynamic-grammar
experiments: fit the cache on demos, then promote the most frequent
fragments to named macros that the LLM can emit as single action
tokens.
"""
from __future__ import annotations

from agplan.grammar_learning.adaptor_cache import AdaptorCache
from agplan.planning.macro import Macro, MacroLibrary, name_from_actions
from agplan.planning.plan_dsl import Action


def promote_top_k(
    cache: AdaptorCache,
    k: int = 8,
    min_count: int = 2,
) -> MacroLibrary:
    """Return a `MacroLibrary` of the top-`k` most frequent fragments
    in `cache` whose count is at least `min_count`.

    Tied counts are broken by fragment length descending then by
    fragment content ascending, so longer reusable subroutines come
    before shorter prefixes when their counts are equal.
    """
    items = [(f, c) for f, c in cache.fragments.items() if c >= min_count]
    items.sort(key=lambda it: (-it[1], -len(it[0]), it[0]))
    chosen = items[:k]
    library = MacroLibrary()
    seen_names: set[str] = set()
    for fragment, _ in chosen:
        actions = tuple(Action(a) for a in fragment)
        name = name_from_actions(actions)
        if name in seen_names:
            continue
        seen_names.add(name)
        library.add(Macro(name=name, expansion=actions))
    return library
