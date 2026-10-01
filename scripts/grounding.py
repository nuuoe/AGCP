"""Ground lifted predicate templates with concrete action arguments."""
from __future__ import annotations


def ground_lifted(template_set: set, args: tuple) -> set:
    """Substitute {a1}, {a2}, ... with concrete args. Returns the
    ground predicate set."""
    out = set()
    for tmpl in template_set:
        g = tmpl
        for i, a in enumerate(args, start=1):
            g = g.replace(f"{{a{i}}}", a)
        out.add(g)
    return out
