"""Tests for ReDuce-S (state-aware grammar induction)."""
from __future__ import annotations

import pytest

from agplan.grammar_learning.reduce_s import (
    StateAwareTrajectory,
    StateClass,
    blocksworld_predicates,
    find_substring_positions,
    induce_state_aware,
    macro_admissible,
)


def test_find_substring_positions_basic():
    seq = ("a", "b", "c", "a", "b", "d")
    assert find_substring_positions(seq, ("a", "b")) == [0, 3]
    assert find_substring_positions(seq, ("a", "b", "c")) == [0]
    assert find_substring_positions(seq, ("x",)) == []
    assert find_substring_positions(seq, ()) == []


def test_find_substring_positions_no_overlap():
    """Non-overlapping advance: aaa matched as (a,a) at pos 0 only."""
    seq = ("a", "a", "a", "a")
    assert find_substring_positions(seq, ("a", "a")) == [0, 2]


def test_state_class_entails():
    c = StateClass(positive=frozenset({"p", "q"}),
                   negative=frozenset({"r"}))
    assert c.entails(frozenset({"p", "q", "s"}))
    assert not c.entails(frozenset({"p", "s"}))  # missing q
    assert not c.entails(frozenset({"p", "q", "r"}))  # has r


def test_state_aware_trajectory_validates_length():
    with pytest.raises(ValueError):
        StateAwareTrajectory(
            actions=("a", "b"),
            states=(frozenset({"p"}), frozenset({"q"})),  # need 3
        )


def test_blocksworld_predicates_initial():
    """Initial state: A on table, clear, B on top of A, held empty."""
    support = {"A": "TABLE", "B": "A"}
    held = None
    labels = ["A", "B"]
    preds = blocksworld_predicates(support, held, labels)
    assert "handempty" in preds
    assert "ontable(A)" in preds
    assert "on(B,A)" in preds
    assert "clear(B)" in preds
    assert "clear(A)" not in preds  # B is on top of A


def test_blocksworld_predicates_holding():
    """Holding A: hand not empty, A not in support map."""
    support = {"B": "TABLE"}  # A is being held
    held = "A"
    labels = ["A", "B"]
    preds = blocksworld_predicates(support, held, labels)
    assert "holding(A)" in preds
    assert "handempty" not in preds
    assert "clear(B)" in preds


def test_induce_state_aware_smoke():
    """Smoke: two short trajectories, induction returns a grammar
    with macros and at least one occurrence found."""
    # Trajectory 1: A on table, pick A, put A down. Repeat with B.
    traj1 = StateAwareTrajectory(
        actions=("pickup(A)", "putdown(A)", "pickup(B)", "putdown(B)"),
        states=(
            frozenset({"handempty", "ontable(A)", "ontable(B)",
                       "clear(A)", "clear(B)"}),
            frozenset({"holding(A)", "ontable(B)", "clear(B)"}),
            frozenset({"handempty", "ontable(A)", "ontable(B)",
                       "clear(A)", "clear(B)"}),
            frozenset({"holding(B)", "ontable(A)", "clear(A)"}),
            frozenset({"handempty", "ontable(A)", "ontable(B)",
                       "clear(A)", "clear(B)"}),
        ),
    )
    traj2 = StateAwareTrajectory(
        actions=("pickup(A)", "putdown(A)", "pickup(B)", "putdown(B)"),
        states=traj1.states,  # same shape for the smoke test
    )

    g = induce_state_aware([traj1, traj2])
    assert g.base_grammar is not None
    assert len(g.predicate_vocab) > 0
    # At least one macro should have been found and profiled
    profile_lists = list(g.profiles.values())
    assert any(len(plist) > 0 for plist in profile_lists)


def test_compile_state_aware_ebnf_smoke():
    """End-to-end: induce ReDuce-S from 2-block Blocksworld
    trajectories, compile EBNF, check it's syntactically reasonable."""
    # Simple traj: pickup A, stack A B
    s0 = frozenset({"handempty", "ontable(A)", "ontable(B)",
                    "clear(A)", "clear(B)"})
    s1 = frozenset({"holding(A)", "ontable(B)", "clear(B)"})
    s2 = frozenset({"handempty", "ontable(B)", "on(A,B)",
                    "clear(A)"})
    traj = StateAwareTrajectory(
        actions=("pickup(A)", "stack(A,B)"),
        states=(s0, s1, s2),
    )
    grammar = induce_state_aware([traj, traj])  # duplicate for fold

    def goal_check(s: frozenset[str]) -> bool:
        return "on(A,B)" in s

    from agplan.grammar_learning.reduce_s import compile_state_aware_ebnf
    ebnf = compile_state_aware_ebnf(grammar, s0, goal_check)
    assert ebnf is not None
    assert "action_seq" in ebnf
    assert "sc0" in ebnf  # initial class


def test_plan_admits_basic():
    """The _plan_admits parser should recognize plans whose
    action sequence corresponds to a valid path through the
    state-class EBNF productions."""
    from scripts.run_leave_out_n import _plan_admits

    ebnf = (
        'action_seq ::= sc0\n'
        'sc0 ::= "\\"pickup(A)\\"" ws "," ws "\\"stack(A,B)\\""\n'
        'sc1 ::= ""\n'
    )
    assert _plan_admits(ebnf, ("pickup(A)", "stack(A,B)"))
    assert not _plan_admits(ebnf, ("pickup(B)", "stack(A,B)"))
    assert not _plan_admits(ebnf, ("pickup(A)",))


def test_plan_admits_with_intermediate_class():
    """Two-step EBNF: sc0 -> action , sc1 ; sc1 -> action.
    Both intermediate class hops and terminal alternative should
    parse correctly."""
    from scripts.run_leave_out_n import _plan_admits

    ebnf = (
        'action_seq ::= sc0\n'
        'sc0 ::= "\\"a(X)\\"" ws "," ws sc1\n'
        'sc1 ::= "\\"b(Y)\\"" ws "," ws sc2 | "\\"c(Z)\\""\n'
        'sc2 ::= "\\"d(W)\\""\n'
    )
    assert _plan_admits(ebnf, ("a(X)", "b(Y)", "d(W)"))
    assert _plan_admits(ebnf, ("a(X)", "c(Z)"))
    assert not _plan_admits(ebnf, ("a(X)", "b(Y)", "c(Z)"))


def test_macro_admissible_simple():
    """A macro requiring handempty + ontable(A) should be admissible
    at states satisfying both."""
    from agplan.grammar_learning.reduce_s import MacroProfile, MacroOccurrence
    from agplan.grammar_learning.reduce import NTSymbol

    state_with = frozenset({"handempty", "ontable(A)", "clear(A)"})
    state_without_handempty = frozenset({"holding(A)", "clear(A)"})
    prof = MacroProfile(
        nt=NTSymbol(99),
        body=("pickup(A)",),
        occurrences=[
            MacroOccurrence(traj_idx=0, position=0,
                            entry_state=state_with,
                            exit_state=frozenset({"holding(A)"})),
            MacroOccurrence(traj_idx=1, position=0,
                            entry_state=frozenset({"handempty",
                                                   "ontable(A)",
                                                   "clear(A)",
                                                   "ontable(B)"}),
                            exit_state=frozenset({"holding(A)"})),
        ],
    )
    # Entry class intersection: handempty + ontable(A) + clear(A)
    ec = prof.entry_class()
    assert "handempty" in ec.positive
    assert "ontable(A)" in ec.positive

    assert macro_admissible(prof, state_with)
    assert not macro_admissible(prof, state_without_handempty)
