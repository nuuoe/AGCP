"""Minimal Gripper environment with a raw dict state.

Gripper (IPC benchmark): a two-gripper robot moves between rooms, picking
up and dropping balls. State keys: rooms, balls, grippers, robby_at,
ball_at (ball -> room, absent while carried) and holding (gripper -> ball,
absent when free). Actions: move(from, to), pick(ball, room, gripper),
drop(ball, room, gripper). step() returns (obs, reward, terminated,
truncated, info); on an illegal action info["error"] is set and the state
is unchanged.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GripperEnv:
    rooms: list[str] = field(default_factory=list)
    balls: list[str] = field(default_factory=list)
    grippers: list[str] = field(default_factory=lambda: ["left", "right"])
    robby_at: Optional[str] = None
    ball_at: dict[str, str] = field(default_factory=dict)
    # gripper -> ball; absent means free
    holding: dict[str, str] = field(default_factory=dict)
    goal_ball_at: dict[str, str] = field(default_factory=dict)
    terminated: bool = False

    def reset_from(self, rooms, balls, robby_at, ball_at,
                    goal_ball_at, grippers=None):
        self.rooms = list(rooms)
        self.balls = list(balls)
        if grippers is not None:
            self.grippers = list(grippers)
        self.robby_at = robby_at
        self.ball_at = dict(ball_at)
        self.holding = {}
        self.goal_ball_at = dict(goal_ball_at)
        self.terminated = False
        return self._obs()

    def reset_random(self, n_rooms=2, n_balls=4, seed=None):
        rng = random.Random(seed)
        self.rooms = [f"room{i}" for i in range(n_rooms)]
        self.balls = [f"ball{i}" for i in range(n_balls)]
        self.grippers = ["left", "right"]
        self.robby_at = rng.choice(self.rooms)
        self.ball_at = {b: rng.choice(self.rooms) for b in self.balls}
        self.holding = {}
        # Goal: all balls to a (possibly different) random target
        target_room = rng.choice(self.rooms)
        self.goal_ball_at = {b: target_room for b in self.balls}
        self.terminated = False
        return self._obs()

    def _obs(self):
        return {
            "rooms": list(self.rooms),
            "balls": list(self.balls),
            "grippers": list(self.grippers),
            "robby_at": self.robby_at,
            "ball_at": dict(self.ball_at),
            "holding": dict(self.holding),
        }

    def raw_state(self):
        """Expose the state as a structured dict (the input to
        predicate discovery / induction). No predicate vocabulary."""
        return {
            "robby_at": self.robby_at,
            "ball_at": dict(self.ball_at),
            "holding": dict(self.holding),
        }

    def goal_satisfied(self):
        return all(self.ball_at.get(b) == r
                   for b, r in self.goal_ball_at.items())

    def step(self, action_name, *args):
        info = {}
        if action_name == "move":
            if len(args) != 2:
                info["error"] = "move needs (from, to)"
                return self._obs(), 0.0, False, False, info
            frm, to = args
            if self.robby_at != frm:
                info["error"] = "robby not at from"
                return self._obs(), 0.0, False, False, info
            if to not in self.rooms:
                info["error"] = "to not a room"
                return self._obs(), 0.0, False, False, info
            if frm == to:
                info["error"] = "move to same room"
                return self._obs(), 0.0, False, False, info
            self.robby_at = to
        elif action_name == "pick":
            if len(args) != 3:
                info["error"] = "pick needs (ball, room, gripper)"
                return self._obs(), 0.0, False, False, info
            ball, room, gripper = args
            if self.robby_at != room:
                info["error"] = "robby not at room"
                return self._obs(), 0.0, False, False, info
            if self.ball_at.get(ball) != room:
                info["error"] = "ball not in room"
                return self._obs(), 0.0, False, False, info
            if gripper in self.holding:
                info["error"] = "gripper not free"
                return self._obs(), 0.0, False, False, info
            del self.ball_at[ball]
            self.holding[gripper] = ball
        elif action_name == "drop":
            if len(args) != 3:
                info["error"] = "drop needs (ball, room, gripper)"
                return self._obs(), 0.0, False, False, info
            ball, room, gripper = args
            if self.robby_at != room:
                info["error"] = "robby not at room"
                return self._obs(), 0.0, False, False, info
            if self.holding.get(gripper) != ball:
                info["error"] = "gripper not holding ball"
                return self._obs(), 0.0, False, False, info
            del self.holding[gripper]
            self.ball_at[ball] = room
        else:
            info["error"] = f"unknown action {action_name}"
            return self._obs(), 0.0, False, False, info

        if self.goal_satisfied():
            self.terminated = True
            return self._obs(), 1.0, True, False, info
        return self._obs(), 0.0, False, False, info


__all__ = ["GripperEnv"]
