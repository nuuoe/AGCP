"""Lifted action-model induction with positional argument templating.

When the same object fills several parameter slots (e.g. waypoint1 at
positions 3 and 4), each predicate is templated once per valid binding
(at({a1},{a3}) and at({a1},{a4})) and intersection statistics run over
these template sets rather than a single first-occurrence template.
"""
from __future__ import annotations

import re
from collections import defaultdict
from itertools import product


def positional_templates(pred: str, args: tuple[str, ...]) -> set[str]:
    """Return every valid positional templating of `pred` given the action `args`.

    Each token matching an arg value becomes {a_i}; an arg value that
    occurs at several positions yields one template per binding. Tokens
    matching no arg are left unchanged.
    """
    if not args:
        return {pred}
    # arg value -> 1-indexed positions where it occurs
    pos_map: dict[str, list[int]] = defaultdict(list)
    for i, a in enumerate(args, start=1):
        pos_map[a].append(i)
    matches = []  # list of (start, end, arg_value)
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(a) for a in pos_map) + r")\b"
    )
    for m in pattern.finditer(pred):
        matches.append((m.start(), m.end(), m.group(1)))
    if not matches:
        return {pred}
    # One template per combination of position choices across matches.
    choices = [pos_map[arg] for _, _, arg in matches]
    out = set()
    for combo in product(*choices):
        new = []
        cursor = 0
        for (s, e, _arg), pos in zip(matches, combo):
            new.append(pred[cursor:s])
            new.append(f"{{a{pos}}}")
            cursor = e
        new.append(pred[cursor:])
        out.add("".join(new))
    return out


def induce_lifted_models_positional(transitions) -> dict:
    """Induce schemas over positional template sets.

    A template is a precondition iff it is in the templated state_before
    set of every transition of that action; effects use the same causal
    semantics as induce_pddl_generic.induce_lifted_models.
    """
    def _keep(p: str) -> bool:
        return ("{a" in p) or all(c not in p for c in "()")

    by_action: dict[str, list] = defaultdict(list)
    for sb, action, args, sa in transitions:
        if sb == sa: continue
        by_action[action].append((sb, args, sa))

    models = {}
    for action, items in by_action.items():
        if not items: continue
        sb_t_list = []
        sa_t_list = []
        for sb, args, sa in items:
            sb_t = set()
            for p in sb:
                sb_t |= positional_templates(p, args)
            sb_t = {p for p in sb_t if _keep(p)}
            sa_t = set()
            for p in sa:
                sa_t |= positional_templates(p, args)
            sa_t = {p for p in sa_t if _keep(p)}
            sb_t_list.append(sb_t)
            sa_t_list.append(sa_t)
        pre_inter = (set.intersection(*sb_t_list) if sb_t_list else set())

        # Causal effect semantics: a candidate add is kept iff every
        # transition has p in sb or sa (dels symmetrically).
        candidate_adds: set[str] = set()
        candidate_dels: set[str] = set()
        for sb_t, sa_t in zip(sb_t_list, sa_t_list):
            candidate_adds |= sa_t - sb_t
            candidate_dels |= sb_t - sa_t
        eff_add = set()
        for p in candidate_adds:
            if all((p in sb_t) or (p in sa_t)
                   for sb_t, sa_t in zip(sb_t_list, sa_t_list)):
                eff_add.add(p)
        eff_del = set()
        for p in candidate_dels:
            if all((p not in sb_t) or (p not in sa_t)
                   for sb_t, sa_t in zip(sb_t_list, sa_t_list)):
                eff_del.add(p)
        models[action] = {
            "pre_pos": pre_inter,
            "eff_add": eff_add,
            "eff_del": eff_del,
            "n_examples": len(items),
        }
    return models


if __name__ == "__main__":
    # Example with a repeated argument.
    args = ("rover1", "general", "waypoint1", "waypoint1", "waypoint2")
    print(f"positional_templates('at(rover1,waypoint1)', {args})")
    print("  ->", sorted(positional_templates("at(rover1,waypoint1)", args)))
    print(f"positional_templates('have_soil_analysis(rover1,waypoint1)', {args})")
    print("  ->", sorted(positional_templates(
        "have_soil_analysis(rover1,waypoint1)", args)))
