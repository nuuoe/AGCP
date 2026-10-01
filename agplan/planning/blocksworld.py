"""Minimal Blocksworld environment.

Pure Python, with a gymnasium-style reset/step interface. Each block is
on the table, on another block or held by the gripper. Actions:
  pickup(b)   : b clear and on the table, gripper empty
  unstack(b,c): b clear and on c, gripper empty
  putdown(b)  : b held; placed on the table
  stack(b,c)  : b held and c clear; placed on c

The goal is a tuple of (block, support) pairs, support being another
block or 'TABLE'; the episode ends with reward 1 when it holds.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Goal:
    """List of (block, support) pairs the goal requires."""
    pairs: tuple[tuple[str, str], ...]


class BlocksworldEnv:
    """Plain-Python Blocksworld."""

    ACTIONS = ("pickup", "unstack", "putdown", "stack")

    def __init__(self, n_blocks: int = 4, max_steps: int = 60):
        self.n_blocks = n_blocks
        self.max_steps = max_steps
        self._labels = [chr(ord('A') + i) for i in range(n_blocks)]
        # state: dict block -> support ('TABLE' or another block);
        #        held_block: Optional[str]
        self.support: dict[str, str] = {}
        self.held: Optional[str] = None
        self.goal: Goal = Goal(())
        self.steps = 0
        self.terminated = False

    @property
    def labels(self) -> list[str]:
        return list(self._labels)

    def is_clear(self, b: str) -> bool:
        if self.held == b:
            return False
        for x, sup in self.support.items():
            if sup == b:
                return False
        return True

    def reset_from(
        self, labels: list[str], initial_support: dict[str, str],
        goal_pairs: tuple, held: Optional[str] = None,
        partial_goal: bool = False,
    ) -> dict:
        """Initialize the env from a specific (initial, goal) spec.

        Used for loading PlanBench instances. If partial_goal is True,
        goal_satisfied() checks only the listed constraints; otherwise
        the full support config must match.
        """
        self._labels = list(labels)
        self.n_blocks = len(labels)
        self.support = dict(initial_support)
        self.held = held
        self.goal = Goal(tuple(goal_pairs))
        self._partial_goal = partial_goal
        self.steps = 0
        self.terminated = False
        return self._obs()

    def reset(self, seed: Optional[int] = None) -> dict:
        rng = random.Random(seed)
        self._partial_goal = False
        # Build a random initial stack configuration: each block on table or
        # on top of a previously placed block.
        order = self._labels[:]
        rng.shuffle(order)
        self.support = {}
        used_supports = {"TABLE"}
        for b in order:
            choices = ["TABLE"] + [
                x for x in self.support if self.is_clear_dict_only(x, self.support)
            ]
            sup = rng.choice(choices)
            self.support[b] = sup
        self.held = None
        # Sample a goal configuration (different from initial).
        for _ in range(20):
            goal_support = self._random_config(rng)
            if goal_support != self.support:
                break
        self.goal = Goal(tuple(sorted(goal_support.items())))
        self.steps = 0
        self.terminated = False
        return self._obs()

    def is_clear_dict_only(self, b: str, support_map: dict[str, str]) -> bool:
        for x, sup in support_map.items():
            if sup == b:
                return False
        return True

    def _random_config(self, rng: random.Random) -> dict[str, str]:
        order = self._labels[:]
        rng.shuffle(order)
        cfg: dict[str, str] = {}
        for b in order:
            choices = ["TABLE"] + [
                x for x in cfg if self.is_clear_dict_only(x, cfg)
            ]
            cfg[b] = rng.choice(choices)
        return cfg

    def _obs(self) -> dict:
        return {
            "support": dict(self.support),
            "held": self.held,
            "goal": self.goal,
            "labels": list(self._labels),
        }

    def render_text(self) -> str:
        """Text rendering of the towers, the held block and the goal, for prompts."""
        bases = [b for b, sup in self.support.items() if sup == "TABLE"]
        towers: list[list[str]] = []
        for base in sorted(bases):
            tower = [base]
            while True:
                top = tower[-1]
                next_block = next(
                    (b for b, sup in self.support.items() if sup == top), None
                )
                if next_block is None:
                    break
                tower.append(next_block)
            towers.append(tower)
        lines = [f"Tower {i + 1}: {' on '.join(reversed(t))} on TABLE"
                 for i, t in enumerate(towers)]
        held_str = f"Holding: {self.held}" if self.held else "Holding: nothing"
        goal_str = "Goal: " + ", ".join(
            f"{b} on {sup}" for b, sup in self.goal.pairs
        )
        return "\n".join(lines + [held_str, goal_str])

    def goal_satisfied(self) -> bool:
        if self.held is not None:
            return False
        if getattr(self, "_partial_goal", False):
            # PlanBench-style: only listed constraints must hold.
            return all(self.support.get(b) == sup
                       for b, sup in self.goal.pairs)
        return tuple(sorted(self.support.items())) == self.goal.pairs

    def step(self, action_name: str, arg1: Optional[str] = None,
             arg2: Optional[str] = None) -> tuple[dict, float, bool, bool, dict]:
        """Apply an action. Returns (obs, reward, terminated, truncated, info)."""
        self.steps += 1
        truncated = self.steps >= self.max_steps
        info = {}

        try:
            if action_name == "pickup":
                if self.held is not None: raise ValueError("gripper not empty")
                if arg1 is None: raise ValueError("pickup needs block")
                if self.support.get(arg1) != "TABLE": raise ValueError("not on table")
                if not self.is_clear(arg1): raise ValueError("not clear")
                del self.support[arg1]
                self.held = arg1
            elif action_name == "unstack":
                if self.held is not None: raise ValueError("gripper not empty")
                if arg1 is None or arg2 is None:
                    raise ValueError("unstack needs two blocks")
                if self.support.get(arg1) != arg2: raise ValueError("not on top")
                if not self.is_clear(arg1): raise ValueError("not clear")
                del self.support[arg1]
                self.held = arg1
            elif action_name == "putdown":
                if self.held is None: raise ValueError("gripper empty")
                self.support[self.held] = "TABLE"
                self.held = None
            elif action_name == "stack":
                if self.held is None: raise ValueError("gripper empty")
                if arg2 is None: raise ValueError("stack needs target block")
                if not self.is_clear(arg2): raise ValueError("target not clear")
                self.support[self.held] = arg2
                self.held = None
            else:
                raise ValueError(f"unknown action {action_name}")
        except ValueError as e:
            info["error"] = str(e)
            return self._obs(), 0.0, False, truncated, info

        if self.goal_satisfied():
            self.terminated = True
            return self._obs(), 1.0, True, False, info
        return self._obs(), 0.0, False, truncated, info
