import pytest

from agplan.grammar_learning.adaptor_cache import AdaptorCache
from agplan.grammar_learning.macro_promotion import promote_top_k
from agplan.grammars.build_schema import base_action_names, build_plan_schema
from agplan.planning.macro import Macro, MacroLibrary, name_from_actions
from agplan.planning.plan_dsl import Action


def test_macro_name_validation():
    Macro(name="m_3f", expansion=(Action.MOVE_FORWARD,) * 3)
    with pytest.raises(ValueError):
        Macro(name="bad name", expansion=(Action.MOVE_FORWARD,))
    with pytest.raises(ValueError):
        Macro(name="m_3f", expansion=())


def test_name_from_actions_runs_collapse():
    n = name_from_actions((Action.MOVE_FORWARD,) * 3)
    assert n == "m_3f"
    n2 = name_from_actions((Action.TURN_LEFT, Action.MOVE_FORWARD))
    assert n2 == "m_lf"
    n3 = name_from_actions((Action.MOVE_FORWARD, Action.TURN_LEFT, Action.MOVE_FORWARD,
                            Action.MOVE_FORWARD))
    assert n3 == "m_fl2f"


def test_library_expand_base_action():
    lib = MacroLibrary()
    assert lib.expand_token("move_forward") == (Action.MOVE_FORWARD,)


def test_library_expand_macro():
    lib = MacroLibrary([Macro("m_3f", (Action.MOVE_FORWARD,) * 3)])
    assert lib.expand_token("m_3f") == (Action.MOVE_FORWARD,) * 3


def test_library_expand_mixed_sequence():
    lib = MacroLibrary([
        Macro("m_lf", (Action.TURN_LEFT, Action.MOVE_FORWARD)),
        Macro("m_3f", (Action.MOVE_FORWARD,) * 3),
    ])
    out = lib.expand(["m_lf", "pickup", "m_3f", "done"])
    assert out == (
        Action.TURN_LEFT, Action.MOVE_FORWARD, Action.PICKUP,
        Action.MOVE_FORWARD, Action.MOVE_FORWARD, Action.MOVE_FORWARD,
        Action.DONE,
    )


def test_library_unknown_token_raises():
    lib = MacroLibrary()
    with pytest.raises((KeyError, ValueError)):
        lib.expand_token("not_an_action")


def test_library_vocabulary_lists_base_then_macros():
    lib = MacroLibrary([Macro("m_3f", (Action.MOVE_FORWARD,) * 3)])
    vocab = lib.vocabulary()
    assert "move_forward" in vocab
    assert "m_3f" in vocab
    # macros come after base actions
    assert vocab.index("m_3f") > vocab.index("done")


def test_promote_top_k_picks_most_frequent():
    cache = AdaptorCache()
    cache.update(("move_forward", "move_forward"), success=True)
    cache.update(("move_forward", "move_forward"), success=True)
    cache.update(("turn_left", "move_forward"), success=True)
    cache.update(("toggle", "done"), success=True)
    lib = promote_top_k(cache, k=2, min_count=1)
    names = lib.names()
    # most frequent first; "move_forward, move_forward" has count 2
    assert "m_2f" in names
    # second most: tie 1 between turn_left/move_forward and toggle/done,
    # broken by length desc then content asc
    assert len(names) == 2


def test_build_plan_schema_extends_enum():
    base = base_action_names()
    extended = base + ("m_3f",)
    schema = build_plan_schema(extended)
    assert "m_3f" in schema
    assert "move_forward" in schema
    assert "additionalProperties" in schema
