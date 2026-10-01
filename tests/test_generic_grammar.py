"""Tests for the generic action-model compiler."""
from __future__ import annotations

from agplan.grammars.generic_grammar import compile_from_action_models


def test_generic_compiler_smoke_blocksworld():
    """Smoke: hand-write a Blocksworld action model, compile, get an
    EBNF that admits the obvious 2-step plan to put A on B."""
    models = {
        "pickup": {
            "pre_pos": {"handempty", "clear({a1})", "ontable({a1})"},
            "eff_add": {"holding({a1})"},
            "eff_del": {"handempty", "clear({a1})", "ontable({a1})"},
            "arity": 1,
        },
        "putdown": {
            "pre_pos": {"holding({a1})"},
            "eff_add": {"handempty", "clear({a1})", "ontable({a1})"},
            "eff_del": {"holding({a1})"},
            "arity": 1,
        },
        "stack": {
            "pre_pos": {"holding({a1})", "clear({a2})"},
            "eff_add": {"handempty", "clear({a1})", "on({a1},{a2})"},
            "eff_del": {"holding({a1})", "clear({a2})"},
            "arity": 2,
        },
        "unstack": {
            "pre_pos": {"handempty", "clear({a1})", "on({a1},{a2})"},
            "eff_add": {"holding({a1})", "clear({a2})"},
            "eff_del": {"handempty", "clear({a1})", "on({a1},{a2})"},
            "arity": 2,
        },
    }
    initial = frozenset({
        "handempty",
        "ontable(a)", "ontable(b)",
        "clear(a)", "clear(b)",
    })
    goal = {"on(a, b)".replace(" ", "")}
    ebnf = compile_from_action_models(models, initial, goal,
                                       objects=["a", "b"], max_extra=2)
    assert ebnf is not None
    assert "pickup(a)" in ebnf
    assert "stack(a,b)" in ebnf
    assert ebnf.count("::=") >= 3
    # Goal already-satisfied check: empty plan should NOT be accepted
    # (a not on b at start)
    assert '"{\\"plan\\":[]}"' not in ebnf


def test_arity_inference_from_templates():
    """If 'arity' is omitted, inferred from max {aN} index."""
    models = {
        "act": {
            "pre_pos": {"p({a1})", "q({a2})"},
            "eff_add": set(),
            "eff_del": set(),
        },
    }
    initial = frozenset({"p(x)", "q(y)"})
    goal: set[str] = set()  # vacuous; should still compile or return None
    out = compile_from_action_models(models, initial, goal,
                                      objects=["x", "y"], max_extra=1)
    # No transitions reach a non-trivial new state -> goal may not be
    # reachable depending on the empty-goal semantics; this only
    # checks that the arity inference path does not crash.
    # (Goal is empty set, satisfied by any state.)
    assert out is None or "act" in out or out == ""


def test_zero_arity_action():
    """A nullary action (e.g. 'noop') should be admitted."""
    models = {
        "noop": {
            "pre_pos": set(),
            "eff_add": {"flag"},
            "eff_del": set(),
            "arity": 0,
        },
    }
    initial = frozenset()
    goal = {"flag"}
    out = compile_from_action_models(models, initial, goal,
                                      objects=[], max_extra=1)
    assert out is not None
    assert "noop" in out


def test_unreachable_goal_returns_none():
    """Goal that no action sequence can reach -> None."""
    models = {
        "a": {
            "pre_pos": set(),
            "eff_add": {"x"},
            "eff_del": set(),
            "arity": 0,
        },
    }
    initial = frozenset()
    goal = {"never_reached"}
    out = compile_from_action_models(models, initial, goal,
                                      objects=[], max_extra=5)
    assert out is None
