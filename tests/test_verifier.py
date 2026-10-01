from agplan.planning.plan_dsl import Action, Plan, Subgoal
from agplan.planning.verifier import verify_plan


def _plan(actions: list[Action]) -> Plan:
    return Plan(subgoals=(Subgoal(name="x", actions=tuple(actions)),))


def test_empty_plan_runs_without_error():
    result = verify_plan(_plan([]), "BabyAI-GoToRedBall-v0", seed=0)
    assert result.steps_taken == 0
    assert result.success is False
    assert result.error is None
    assert result.mission == "go to the red ball"


def test_short_plan_executes_steps_taken():
    plan = _plan([Action.MOVE_FORWARD, Action.TURN_LEFT, Action.MOVE_FORWARD])
    result = verify_plan(plan, "BabyAI-GoToRedBall-v0", seed=0)
    assert result.steps_taken == 3
    assert result.final_pos is not None
    assert isinstance(result.final_dir, int)


def test_max_steps_truncates_action_list():
    plan = _plan([Action.MOVE_FORWARD] * 100)
    result = verify_plan(plan, "BabyAI-GoToRedBall-v0", seed=0, max_steps=5)
    assert result.steps_taken <= 5


def test_seed_determinism():
    plan = _plan([Action.MOVE_FORWARD, Action.MOVE_FORWARD])
    r1 = verify_plan(plan, "BabyAI-GoToRedBall-v0", seed=42)
    r2 = verify_plan(plan, "BabyAI-GoToRedBall-v0", seed=42)
    assert r1.final_pos == r2.final_pos
    assert r1.final_dir == r2.final_dir
    assert r1.total_reward == r2.total_reward
