import json

from agplan.planning.plan_dsl import Action, load_schema_text, schema_path


def test_schema_file_exists():
    assert schema_path().exists()


def test_schema_is_valid_json():
    obj = json.loads(load_schema_text())
    assert obj["type"] == "object"
    assert obj["required"] == ["subgoals"]


def test_schema_action_enum_matches_dsl():
    obj = json.loads(load_schema_text())
    schema_actions = set(
        obj["properties"]["subgoals"]["items"]["properties"]["actions"]["items"]["enum"]
    )
    dsl_actions = {a.value for a in Action}
    assert schema_actions == dsl_actions


def test_ebnf_file_exists():
    from pathlib import Path

    p = Path(__file__).resolve().parent.parent / "agplan" / "grammars" / "babyai_plan.ebnf"
    assert p.exists()
    text = p.read_text()
    assert "subgoals" in text
    assert "move_forward" in text
