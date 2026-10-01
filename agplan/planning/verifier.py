"""Execute a parsed plan in a BabyAI MiniGrid environment and report outcome."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym

from agplan.planning.babyai_adapter import to_minigrid_seq
from agplan.planning.plan_dsl import Plan


@dataclass
class ExecutionResult:
    """Outcome of executing a plan in a MiniGrid env."""
    success: bool
    total_reward: float
    steps_taken: int
    terminated: bool
    truncated: bool
    final_pos: Optional[tuple[int, int]]
    final_dir: Optional[int]
    error: Optional[str] = None
    mission: Optional[str] = None


def verify_plan(
    plan: Plan,
    env_name: str,
    seed: int = 0,
    max_steps: Optional[int] = None,
) -> ExecutionResult:
    """Run a plan's flat action sequence in `env_name` from `seed`.

    Returns an ExecutionResult. `success` is true iff the env terminates
    via the goal predicate (not via truncation). Stops early on terminate
    or truncate.
    """
    actions = to_minigrid_seq(plan.flat_actions())
    if max_steps is not None:
        actions = actions[:max_steps]

    env = gym.make(env_name)
    try:
        obs, _info = env.reset(seed=seed)
        mission = obs.get("mission") if isinstance(obs, dict) else None

        total_reward = 0.0
        terminated = False
        truncated = False
        steps_taken = 0
        error: Optional[str] = None

        for action_int in actions:
            try:
                _obs, reward, terminated, truncated, _info = env.step(action_int)
            except Exception as e:
                error = f"env.step failed at step {steps_taken}: {e}"
                break
            total_reward += float(reward)
            steps_taken += 1
            if terminated or truncated:
                break

        unwrapped: Any = env.unwrapped
        try:
            final_pos = (int(unwrapped.agent_pos[0]), int(unwrapped.agent_pos[1]))
            final_dir = int(unwrapped.agent_dir)
        except Exception:
            final_pos = None
            final_dir = None

        return ExecutionResult(
            success=bool(terminated),
            total_reward=total_reward,
            steps_taken=steps_taken,
            terminated=bool(terminated),
            truncated=bool(truncated),
            final_pos=final_pos,
            final_dir=final_dir,
            error=error,
            mission=mission,
        )
    finally:
        env.close()
