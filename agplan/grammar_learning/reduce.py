"""ReDuce: grammar compression by greedy folding (Muggleton 2025).

Two operations are applied until no adjacent pair occurs more than once:
star_replace collapses runs of identical adjacent symbols into repeat
nodes, and greedy_fold introduces a nonterminal for the most frequent
adjacent pair and replaces every occurrence. The result is a hierarchical
regular grammar that regenerates the input by unfolding. The module also
translates the grammar into XGrammar-compatible EBNF and does not import
xgrammar.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union


# Symbol representation.
# Terminals are strings (e.g. "turn_left").
# Nonterminals are NTSymbol with a unique integer id.
# Star-repeat nodes wrap a single child (terminal or nonterminal).


@dataclass(frozen=True)
class NTSymbol:
    """Unique nonterminal name (e.g. N_0, N_1, ...)."""
    idx: int

    def __repr__(self) -> str:
        return f"N{self.idx}"


@dataclass(frozen=True)
class StarRepeat:
    """One-or-more repetition of a single symbol (regex +); zero-length
    runs of an action carry no information for plans."""
    inner: Union[str, NTSymbol]

    def __repr__(self) -> str:
        return f"({self.inner})+"


Symbol = Union[str, NTSymbol, StarRepeat]


@dataclass
class Grammar:
    """Hierarchical grammar produced by ReDuce.

    productions[nt] is a list of bodies, each a list of Symbol; several
    bodies mean alternation. Folding produces single-body productions;
    only the start symbol of induce_from_trajectories has one body per
    plan.
    """
    start: NTSymbol
    productions: dict[NTSymbol, list[list[Symbol]]] = field(default_factory=dict)
    next_idx: int = 0

    def fresh_nt(self) -> NTSymbol:
        nt = NTSymbol(self.next_idx)
        self.next_idx += 1
        return nt

    def body(self, nt: NTSymbol) -> list[Symbol]:
        bodies = self.productions[nt]
        if len(bodies) != 1:
            raise ValueError(f"expected single-body production, got {bodies}")
        return bodies[0]

    def all_bodies(self):
        for nt, bodies in self.productions.items():
            for body in bodies:
                yield nt, body


def _initial_grammar_from_sequence(seq: list[str]) -> Grammar:
    g = Grammar(start=NTSymbol(0))
    g.next_idx = 1
    g.productions[g.start] = [list(seq)]
    return g


def _star_replace_body(body: list[Symbol]) -> list[Symbol]:
    """Replace maximal runs (length >= 2) of identical adjacent
    symbols with a single StarRepeat node.

    The grammar admits one-or-more repetitions (positive closure),
    a deliberate departure from ReDuce (Muggleton 2025), whose star
    includes the empty repetition (its star_e/star_d clauses have an
    eq(X,X) base case).
    """
    if not body:
        return body
    out: list[Symbol] = []
    i = 0
    n = len(body)
    while i < n:
        sym = body[i]
        j = i + 1
        while j < n and body[j] == sym:
            j += 1
        run_len = j - i
        if run_len >= 2 and not isinstance(sym, StarRepeat):
            # Avoid wrapping star inside star.
            out.append(StarRepeat(inner=sym))  # type: ignore[arg-type]
        else:
            out.extend(body[i:j])
        i = j
    return out


def star_replace(g: Grammar) -> Grammar:
    """Apply star_replace to every production body in-place."""
    for nt, bodies in g.productions.items():
        g.productions[nt] = [_star_replace_body(b) for b in bodies]
    return g


def _max_frequency_pair(g: Grammar) -> Optional[tuple[Symbol, Symbol]]:
    """Most frequent adjacent pair over all bodies, or None if no pair
    occurs twice. A StarRepeat node counts as one symbol."""
    counts: dict[tuple[Symbol, Symbol], int] = {}
    for _nt, body in g.all_bodies():
        for k in range(len(body) - 1):
            pair = (body[k], body[k + 1])
            counts[pair] = counts.get(pair, 0) + 1
    if not counts:
        return None
    best_pair, best_count = max(counts.items(), key=lambda kv: kv[1])
    if best_count < 2:
        return None
    return best_pair


def _substitute_pair_in_body(
    body: list[Symbol], pair: tuple[Symbol, Symbol], replacement: NTSymbol
) -> list[Symbol]:
    """Replace non-overlapping occurrences of pair, scanning left to right."""
    out: list[Symbol] = []
    i = 0
    n = len(body)
    while i < n:
        if i + 1 < n and body[i] == pair[0] and body[i + 1] == pair[1]:
            out.append(replacement)
            i += 2
        else:
            out.append(body[i])
            i += 1
    return out


def define_and_substitute_all(
    g: Grammar, pair: tuple[Symbol, Symbol]
) -> Grammar:
    """Introduce N -> pair and replace pair globally."""
    new_nt = g.fresh_nt()
    g.productions[new_nt] = [list(pair)]
    for nt in list(g.productions.keys()):
        if nt is new_nt:
            continue
        g.productions[nt] = [
            _substitute_pair_in_body(b, pair, new_nt)
            for b in g.productions[nt]
        ]
    return g


def reduce_grammar(seq: list[str]) -> Grammar:
    """ReDuce over one sequence (Muggleton 2025, Algorithm 1)."""
    g = _initial_grammar_from_sequence(seq)
    g = star_replace(g)
    while True:
        pair = _max_frequency_pair(g)
        if pair is None:
            break
        g = define_and_substitute_all(g, pair)
        g = star_replace(g)
    return g


# EBNF translation for XGrammar.


def _ebnf_quote(action: str) -> str:
    """Quote an action terminal as a JSON string literal inside EBNF."""
    return f'"\\"{action}\\""'


def _symbol_to_ebnf(sym: Symbol, ws: str = "ws") -> str:
    """EBNF surface form of one symbol."""
    if isinstance(sym, str):
        return _ebnf_quote(sym)
    if isinstance(sym, NTSymbol):
        return f"n{sym.idx}"
    if isinstance(sym, StarRepeat):
        return f"({_symbol_to_ebnf(sym.inner, ws)})+"
    raise TypeError(f"unknown symbol type: {type(sym)}")


def grammar_to_action_seq_ebnf(g: Grammar) -> str:
    """Translate a ReDuce grammar into the action_seq productions of the plan EBNF.

    Each NTSymbol becomes n{idx}, terminals are JSON-quoted action
    strings, body items are joined by ws "," ws, and a StarRepeat expands
    to inner (ws "," ws inner)* so that a comma separates every pair of
    repeated occurrences.
    """
    lines: list[str] = []

    def _expand_body(body: list[Symbol]) -> str:
        """Join body items with ws "," ws, expanding StarRepeat inline."""
        parts: list[str] = []
        for sym in body:
            if isinstance(sym, StarRepeat):
                inner_str = _symbol_to_ebnf(sym.inner)
                parts.append(f'{inner_str} (ws "," ws {inner_str})*')
            else:
                parts.append(_symbol_to_ebnf(sym))
        return ' ws "," ws '.join(parts)

    for nt in g.productions:
        bodies = g.productions[nt]
        body_strs = [_expand_body(b) for b in bodies]
        lhs = f"n{nt.idx}"
        rhs = " | ".join(body_strs)
        lines.append(f"{lhs} ::= {rhs}")
    return "\n".join(lines)


def wrap_in_plan_schema(action_seq_ebnf: str, start_nt: NTSymbol) -> str:
    """Wrap the action_seq EBNF in the JSON plan schema, giving a complete
    XGrammar grammar for {"subgoals": [{"name": "go", "actions": [...]}]}."""
    header = (
        'root ::= "{" ws "\\"subgoals\\"" ws ":" ws "[" ws subgoal ws "]" ws "}"\n'
        'subgoal ::= "{" ws "\\"name\\"" ws ":" ws "\\"go\\"" ws "," ws '
        '"\\"actions\\"" ws ":" ws "[" ws action_seq ws "]" ws "}"\n'
        f"action_seq ::= n{start_nt.idx}\n"
    )
    footer = '\nws ::= [ \\t\\n\\r]*'
    return header + action_seq_ebnf + footer


def wrap_in_plan_schema_hybrid(
    action_seq_ebnf: str, start_nt: NTSymbol,
    base_actions: tuple[str, ...] = (
        "turn_left", "turn_right", "move_forward",
        "pickup", "drop", "toggle", "done",
    )
) -> str:
    """Wrapper whose action_seq accepts either the ReDuce-induced grammar
    or a free sequence over base_actions, so the mask enforces validity
    without restricting the language to the demonstrated patterns."""
    base_alts = " | ".join(f'"\\"{a}\\""' for a in base_actions)
    header = (
        'root ::= "{" ws "\\"subgoals\\"" ws ":" ws "[" ws subgoal ws "]" ws "}"\n'
        'subgoal ::= "{" ws "\\"name\\"" ws ":" ws "\\"go\\"" ws "," ws '
        '"\\"actions\\"" ws ":" ws "[" ws action_seq ws "]" ws "}"\n'
        f"action_seq ::= n{start_nt.idx} | base_seq\n"
        f"base_seq ::= base_action | base_action ws \",\" ws base_seq\n"
        f"base_action ::= {base_alts}\n"
    )
    footer = '\nws ::= [ \\t\\n\\r]*'
    return header + action_seq_ebnf + footer


# Multi-trajectory induction.


def induce_from_trajectories(plans: list[list[str]]) -> Grammar:
    """Induce one grammar over several action sequences.

    The start symbol gets one body per plan, so folding finds pairs shared
    across plans.
    """
    if not plans:
        raise ValueError("no plans to induce from")
    g = Grammar(start=NTSymbol(0))
    g.next_idx = 1
    g.productions[g.start] = [list(p) for p in plans]
    g = star_replace(g)
    while True:
        pair = _max_frequency_pair(g)
        if pair is None:
            break
        g = define_and_substitute_all(g, pair)
        g = star_replace(g)
    return g


__all__ = [
    "NTSymbol",
    "StarRepeat",
    "Symbol",
    "Grammar",
    "reduce_grammar",
    "induce_from_trajectories",
    "grammar_to_action_seq_ebnf",
    "wrap_in_plan_schema",
    "star_replace",
    "define_and_substitute_all",
]
