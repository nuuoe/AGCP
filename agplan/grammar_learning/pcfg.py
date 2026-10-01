"""Smoothed PCFG prior over grammar rule expansions."""
from __future__ import annotations

import math


class PCFGPrior:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.counts: dict[tuple[str, tuple[str, ...]], float] = {}
        self.totals: dict[str, float] = {}

    def update_rule(self, lhs: str, rhs: tuple[str, ...], weight: float = 1.0) -> None:
        key = (lhs, rhs)
        self.counts[key] = self.counts.get(key, 0.0) + weight
        self.totals[lhs] = self.totals.get(lhs, 0.0) + weight

    def logprob(self, lhs: str, rhs: tuple[str, ...]) -> float:
        candidates = [k for k in self.counts if k[0] == lhs]
        vocab = max(1, len(candidates))
        count = self.counts.get((lhs, rhs), 0.0)
        total = self.totals.get(lhs, 0.0)
        return math.log((count + self.alpha) / (total + self.alpha * vocab))
