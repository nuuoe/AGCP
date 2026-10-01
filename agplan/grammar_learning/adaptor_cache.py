"""Pitman-Yor-style nonparametric prior over reusable plan fragments."""
from __future__ import annotations

import math


class AdaptorCache:
    def __init__(self, discount: float = 0.5, strength: float = 1.0):
        self.discount = discount
        self.strength = strength
        self.fragments: dict[tuple[str, ...], int] = {}
        self.total = 0

    def update(self, fragment: tuple[str, ...], success: bool = True) -> None:
        if not success:
            return
        self.fragments[fragment] = self.fragments.get(fragment, 0) + 1
        self.total += 1

    def log_prior(self, fragment: tuple[str, ...]) -> float:
        count = self.fragments.get(fragment, 0)
        numerator = max(count - self.discount, 1e-8)
        denominator = max(self.total + self.strength, 1e-8)
        return math.log(numerator / denominator)
