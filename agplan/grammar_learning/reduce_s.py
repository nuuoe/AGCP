"""State-class indexed ReDuce (SCI-ReDuce).

Extends ReDuce (Muggleton 2025) by recording the predicate signature of
the states at which each induced macro occurs. The grammar carries one
entry class per macro, and the mask admits a macro only when the current
state entails its entry class, which restores L(G) subseteq L_exec.
State-class refinement over a finite predicate vocabulary terminates,
the class set is finite, and productions have the form N_c -> M N_c',
so the grammar stays right-linear. See Sec. 3.5, App. E and App. P of
the paper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from .reduce import (
    Grammar,
    NTSymbol,
    StarRepeat,
    Symbol,
    induce_from_trajectories,
)


# State and trajectory representations.


@dataclass(frozen=True)
class StateAwareTrajectory:
    """Successful trajectory with logged intermediate states.

    actions: action strings, length L.
    states: predicate sets at each step, length L+1. states[i] is the
            state before action i; states[L] is the final state.
    Each state is represented as a frozenset of true-predicate strings
    (e.g. {"on(A,B)", "clear(C)", "handempty"} for Blocksworld).
    """
    actions: tuple[str, ...]
    states: tuple[frozenset[str], ...]

    def __post_init__(self) -> None:
        if len(self.states) != len(self.actions) + 1:
            raise ValueError(
                f"len(states) = {len(self.states)} != "
                f"len(actions)+1 = {len(self.actions)+1}"
            )


@dataclass(frozen=True)
class StateClass:
    """A class of states described by required positive predicates
    and required absent predicates. A state s entails this class iff
    `positive` is a subset of s.true and `negative` is disjoint from
    s.true. Negative predicates are populated by refinement when a
    coarser class would conflate effect-different entries.
    """
    positive: frozenset[str]
    negative: frozenset[str] = frozenset()

    def entails(self, state_predicates: frozenset[str]) -> bool:
        return (self.positive <= state_predicates
                and self.negative.isdisjoint(state_predicates))


# Macro extraction and state-class computation.


def flatten_macro(nt: NTSymbol, grammar: Grammar) -> tuple[str, ...]:
    """Expand a nonterminal to its action terminals.

    Raises on a production with alternations. A StarRepeat contributes
    its inner symbol once, the canonical form used for substring matching.
    """
    seen: set[NTSymbol] = set()

    def _expand(sym: Symbol) -> list[str]:
        if isinstance(sym, str):
            return [sym]
        if isinstance(sym, NTSymbol):
            if sym in seen:
                raise ValueError(f"recursive macro {sym}")
            seen.add(sym)
            bodies = grammar.productions[sym]
            if len(bodies) != 1:
                raise ValueError(f"macro {sym} has alternations")
            out: list[str] = []
            for s in bodies[0]:
                out.extend(_expand(s))
            return out
        if isinstance(sym, StarRepeat):
            return _expand(sym.inner)
        raise TypeError(f"unknown symbol type: {type(sym)}")

    bodies = grammar.productions[nt]
    if len(bodies) != 1:
        raise ValueError(f"top-level nt {nt} has alternations")
    out: list[str] = []
    for s in bodies[0]:
        out.extend(_expand(s))
    return tuple(out)


def find_substring_positions(seq: tuple[str, ...],
                             pattern: tuple[str, ...]) -> list[int]:
    """Start indices of the non-overlapping occurrences of pattern in seq."""
    if not pattern:
        return []
    out: list[int] = []
    i = 0
    n, m = len(seq), len(pattern)
    while i + m <= n:
        if seq[i:i + m] == pattern:
            out.append(i)
            i += m
        else:
            i += 1
    return out


@dataclass
class MacroOccurrence:
    """One use of a macro in a training trajectory."""
    traj_idx: int
    position: int
    entry_state: frozenset[str]
    exit_state: frozenset[str]


@dataclass
class MacroProfile:
    """Summary of all training occurrences of a macro."""
    nt: NTSymbol
    body: tuple[str, ...]
    occurrences: list[MacroOccurrence] = field(default_factory=list)

    def entry_class(self) -> StateClass:
        """Intersection of positive predicates across all entry states.

        With no occurrences: empty positive set (vacuous match).
        Negative predicates are populated only by refinement.
        """
        if not self.occurrences:
            return StateClass(positive=frozenset())
        common = frozenset.intersection(
            *[o.entry_state for o in self.occurrences]
        )
        return StateClass(positive=common)

    def exit_classes(self) -> list[frozenset[str]]:
        """Distinct exit signatures observed."""
        return list({o.exit_state for o in self.occurrences})


# Refinement: split entry class on predicates that correlate with
# different exit signatures.


def refine_entry_class(profile: MacroProfile,
                       predicate_vocab: frozenset[str]) -> list[MacroProfile]:
    """Split a profile until each part has a single exit signature.

    Each split is on the vocabulary predicate whose presence at entry best
    separates the exit signatures, and both parts are refined recursively.
    Returns the refined profiles, each with a subset of the occurrences;
    empty for a profile without occurrences.
    """
    if not profile.occurrences:
        return []
    if len({o.exit_state for o in profile.occurrences}) <= 1:
        return [profile]

    # Greedy choice: minimise the larger side's number of distinct exit
    # signatures.
    best_split: Optional[tuple[str, list[MacroOccurrence],
                               list[MacroOccurrence]]] = None
    best_drop = -1
    base_exit_count = len({o.exit_state for o in profile.occurrences})
    for p in predicate_vocab:
        with_p = [o for o in profile.occurrences if p in o.entry_state]
        without_p = [o for o in profile.occurrences if p not in o.entry_state]
        if not with_p or not without_p:
            continue  # not informative
        n_with = len({o.exit_state for o in with_p})
        n_without = len({o.exit_state for o in without_p})
        drop = base_exit_count - max(n_with, n_without)
        if drop > best_drop:
            best_drop = drop
            best_split = (p, with_p, without_p)

    if best_split is None:
        # No predicate separates the occurrences; the multi-exit profile
        # compiles to an alternation.
        return [profile]

    p, with_p, without_p = best_split
    prof_with = MacroProfile(nt=profile.nt, body=profile.body,
                             occurrences=with_p)
    prof_without = MacroProfile(nt=profile.nt, body=profile.body,
                                occurrences=without_p)
    # Recurse on each side; the split predicate is carried by the
    # occurrence sets, not stored explicitly.
    out: list[MacroProfile] = []
    for sub in refine_entry_class(prof_with, predicate_vocab - {p}):
        new = MacroProfile(nt=sub.nt, body=sub.body,
                           occurrences=sub.occurrences)
        out.append(new)
    for sub in refine_entry_class(prof_without, predicate_vocab - {p}):
        new = MacroProfile(nt=sub.nt, body=sub.body,
                           occurrences=sub.occurrences)
        out.append(new)
    return out


# Top-level induction.


@dataclass
class StateAwareGrammar:
    """SCI-ReDuce output.

    base_grammar: the underlying ReDuce grammar over actions.
    profiles: per-macro state-class profiles, keyed by either a
              ReDuce NTSymbol (folded macro) or the tuple
              ("action", <action_str>) for atomic-action transitions.
    predicate_vocab: full set of predicates observed across states.
    """
    base_grammar: Grammar
    profiles: dict[object, list[MacroProfile]]
    predicate_vocab: frozenset[str]


def induce_state_aware(
    trajectories: Iterable[StateAwareTrajectory],
) -> StateAwareGrammar:
    """Run SCI-ReDuce over state-annotated trajectories."""
    trajectories = list(trajectories)
    if not trajectories:
        raise ValueError("no trajectories")

    # Base ReDuce on the action sequences only.
    base_grammar = induce_from_trajectories(
        [list(t.actions) for t in trajectories]
    )

    # Predicate vocabulary across all logged states.
    predicate_vocab = frozenset.union(
        *(s for t in trajectories for s in t.states)
    )

    # Macros are the non-start nonterminals (folded sequences) plus every
    # atomic action as a length-1 macro, so the class graph can still
    # move by single actions where no longer macro is admissible.
    macros: dict[object, tuple[str, ...]] = {}
    for nt in base_grammar.productions:
        if nt is base_grammar.start:
            continue
        try:
            macros[nt] = flatten_macro(nt, base_grammar)
        except ValueError:
            continue
    seen_actions: set[str] = set()
    for traj in trajectories:
        for a in traj.actions:
            if a not in seen_actions:
                seen_actions.add(a)
                # Keyed by ("action", a), distinct from the NTSymbol keys.
                macros[("action", a)] = (a,)

    # Find occurrences and record entry/exit states.
    profiles_flat: dict[object, MacroProfile] = {}
    for key, body in macros.items():
        prof = MacroProfile(
            nt=NTSymbol(-1) if isinstance(key, tuple) else key,
            body=body, occurrences=[],
        )
        profiles_flat[key] = prof
        for ti, traj in enumerate(trajectories):
            positions = find_substring_positions(traj.actions, body)
            for pos in positions:
                prof.occurrences.append(MacroOccurrence(
                    traj_idx=ti,
                    position=pos,
                    entry_state=traj.states[pos],
                    exit_state=traj.states[pos + len(body)],
                ))

    # Refine entry classes for each macro.
    profiles: dict[object, list[MacroProfile]] = {}
    for key, prof in profiles_flat.items():
        profiles[key] = refine_entry_class(prof, predicate_vocab)

    return StateAwareGrammar(
        base_grammar=base_grammar,
        profiles=profiles,
        predicate_vocab=predicate_vocab,
    )


# State-class admissibility check (for decoder use).


def macro_admissible(profile: MacroProfile,
                     current_state: frozenset[str]) -> bool:
    """True when current_state entails the macro's entry class; gates macro
    emission in the decoder."""
    return profile.entry_class().entails(current_state)


def predict_exit_states(profile: MacroProfile) -> list[frozenset[str]]:
    """Empirically observed exit signatures for this macro."""
    return profile.exit_classes()


# Blocksworld state -> predicate set helper.


def blocksworld_predicates(
    support: dict[str, str],
    held: Optional[str],
    labels: list[str],
) -> frozenset[str]:
    """Convert a Blocksworld state (support map + held) to the set of
    positive predicates that hold.

    Predicates emitted:
      handempty           if held is None
      holding(b)          if held == b
      ontable(b)          if support[b] == TABLE
      on(b, c)            if support[b] == c (block on block)
      clear(b)            if b has nothing on top and is not held
    """
    out: set[str] = set()
    if held is None:
        out.add("handempty")
    else:
        out.add(f"holding({held})")
    for b in labels:
        sup = support.get(b)
        if sup is None:
            # held block -- already accounted for
            continue
        if sup == "TABLE":
            out.add(f"ontable({b})")
        else:
            out.add(f"on({b},{sup})")
    # clear predicates
    on_top: set[str] = set()  # blocks that have something on top
    for b, sup in support.items():
        if sup != "TABLE":
            on_top.add(sup)
    for b in labels:
        if b == held:
            continue
        if b not in on_top and b in support:
            out.add(f"clear({b})")
    return frozenset(out)


# State-class indexed EBNF compilation.
#
# For a given initial state s_0 and goal predicate check, do BFS
# through (entry_class -> exit_class) transitions defined by the
# induced macros. Output an EBNF where each reachable class is a
# nonterminal and each macro is a labelled transition. This is the
# direct analogue of G_env's PDDL->EBNF, but using observation-induced
# macros instead of base PDDL actions.


def _action_seq_to_json_array(actions: tuple[str, ...]) -> str:
    """EBNF for the inside of a JSON array holding the given actions."""
    return ' ws "," ws '.join(f'"\\"{a}\\""' for a in actions)


def _class_name(class_idx: int) -> str:
    return f"sc{class_idx}"


def compile_state_aware_ebnf(
    grammar: StateAwareGrammar,
    initial_state: frozenset[str],
    goal_check: "callable[[frozenset[str]], bool]",
    max_depth: int = 20,
    include_base_actions: tuple[str, ...] = (),
) -> Optional[str]:
    """Compile the state-class indexed EBNF for one task.

    A BFS from initial_state over the macro edges, capped by max_depth,
    gives the reachable classes. Each becomes a nonterminal that expands
    to a macro body followed by the post-state class, or to the empty
    string at a goal class (goal_check on the predicate set). With
    include_base_actions, every class also admits any of those actions.
    The entry nonterminal is action_seq. Returns None when no goal class
    is reachable.
    """
    # One edge per (entry class, body, observed exit state); the refined
    # entry class is the admissibility precondition.
    edges: list[tuple[StateClass, tuple[str, ...],
                      frozenset[str]]] = []
    for profile_list in grammar.profiles.values():
        for prof in profile_list:
            ec = prof.entry_class()
            if not prof.body:
                continue
            for occ in prof.occurrences:
                edges.append((ec, prof.body, occ.exit_state))

    if not edges:
        return None

    # BFS from initial_state. Each visited node is a concrete state
    # predicate set (frozenset). Any state entailing an entry class is
    # admitted, but the BFS tracks the concrete states reached so the
    # output EBNF only mentions classes that are actually reachable in
    # the test task.
    visited: dict[frozenset[str], int] = {initial_state: 0}
    queue: list[tuple[frozenset[str], int]] = [(initial_state, 0)]
    transitions: list[tuple[int, tuple[str, ...], int]] = []
    goal_idx: set[int] = set()

    next_idx = 1
    while queue:
        state, sidx = queue.pop(0)
        if goal_check(state):
            goal_idx.add(sidx)
            continue
        for ec, body, exit_state in edges:
            if not ec.entails(state):
                continue
            if exit_state in visited:
                tidx = visited[exit_state]
            else:
                visited[exit_state] = next_idx
                tidx = next_idx
                next_idx += 1
                if next_idx <= max_depth * len(edges):
                    queue.append((exit_state, tidx))
            transitions.append((sidx, body, tidx))

    if not any(idx in goal_idx for idx in visited.values()):
        # States reached but never dequeued under the depth cap may still
        # satisfy the goal.
        for st, idx in visited.items():
            if goal_check(st):
                goal_idx.add(idx)
    if not goal_idx:
        return None

    # Emit EBNF.
    lines: list[str] = []
    lines.append(f"action_seq ::= {_class_name(0)}")
    # Group transitions by source class.
    by_src: dict[int, list[tuple[tuple[str, ...], int]]] = {}
    for src, body, dst in transitions:
        by_src.setdefault(src, []).append((body, dst))

    for src, edges_from in by_src.items():
        alts: list[str] = []
        seen: set[str] = set()
        for body, dst in edges_from:
            body_str = _action_seq_to_json_array(body)
            if dst in goal_idx:
                alt = body_str
            else:
                alt = f'{body_str} ws "," ws {_class_name(dst)}'
            if alt not in seen:
                alts.append(alt)
                seen.add(alt)
        # A goal class may also terminate here.
        if src in goal_idx and '""' not in seen:
            alts.append('""')
        if include_base_actions:
            # Any base action, staying in the same class.
            base_alts = " | ".join(
                f'"\\"{a}\\""' for a in include_base_actions
            )
            alts.append(
                f"({base_alts}) ws \",\" ws {_class_name(src)}"
            )
        lines.append(f"{_class_name(src)} ::= " + " | ".join(alts))

    # Goal classes without outgoing transitions:
    for idx in goal_idx:
        if idx not in by_src:
            lines.append(f'{_class_name(idx)} ::= ""')

    return "\n".join(lines)


def wrap_state_aware_ebnf_in_plan_schema(action_seq_ebnf: str) -> str:
    """Wrap the compiled action_seq EBNF in the JSON plan schema; the same
    wrapper as reduce.wrap_in_plan_schema."""
    header = (
        'root ::= "{" ws "\\"subgoals\\"" ws ":" ws "[" ws subgoal ws "]" ws "}"\n'
        'subgoal ::= "{" ws "\\"name\\"" ws ":" ws "\\"go\\"" ws "," ws '
        '"\\"actions\\"" ws ":" ws "[" ws action_seq ws "]" ws "}"\n'
    )
    footer = '\nws ::= [ \\t\\n\\r]*'
    return header + action_seq_ebnf + footer


def wrap_state_aware_ebnf_in_plan_array(action_seq_ebnf: str) -> str:
    """Wrap the action_seq EBNF in the `{"plan": [...]}` schema that
    run_genv_planbench.py and the Blocksworld/Mystery prompts use."""
    header = (
        'root ::= "{" ws "\\"plan\\"" ws ":" ws "[" ws action_seq ws "]" ws "}"\n'
    )
    footer = '\nws ::= [ \\t\\n\\r]*'
    return header + action_seq_ebnf + footer


__all__ = [
    "StateAwareTrajectory",
    "StateClass",
    "MacroProfile",
    "MacroOccurrence",
    "StateAwareGrammar",
    "induce_state_aware",
    "macro_admissible",
    "predict_exit_states",
    "flatten_macro",
    "find_substring_positions",
    "refine_entry_class",
    "blocksworld_predicates",
    "compile_state_aware_ebnf",
    "wrap_state_aware_ebnf_in_plan_schema",
    "wrap_state_aware_ebnf_in_plan_array",
]
