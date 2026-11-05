"""Elo rating models.

Every team starts at 1500. Before a match the expected score of the home team is

    expected = 1 / (1 + 10 ** (-gap / 400))        where gap = home rating - away rating + hfa

After the match the home team gains (and the away team loses) k * (actual - expected), where
actual is 1 for a win, 0.5 for a draw and 0 for a loss.
"""
from __future__ import annotations

import math
from collections import defaultdict

from ratings.base import RatingModel

START_RATING = 1500.0


def margin_multiplier(kind: str, goal_margin: int) -> float:
    """How much to scale the rating change for a win by `goal_margin` goals."""
    if kind == "none" or goal_margin <= 1:
        return 1.0
    if kind == "sqrt":
        return math.sqrt(goal_margin)
    if kind == "log":
        return math.log(goal_margin + 1) + 1.0
    if kind == "wfe":  # World Football Elo (eloratings.net)
        return 1.5 if goal_margin == 2 else (11 + goal_margin) / 8
    raise ValueError(f"unknown margin-of-victory kind: {kind}")


class Elo(RatingModel):
    """Elo with home advantage.

    k          size of each rating update
    hfa        home advantage, in rating points
    mov        margin-of-victory scaling: "none", "sqrt", "log" or "wfe"
    xg_weight  if above 0, the "actual" score blends xG difference and goal difference
               (see update) instead of using win/draw/loss
    xg_c       how steeply that blended margin is turned into a 0-1 score
    carry      share of a team's rating above/below 1500 kept when a new season starts
    autocorr   shrink the margin bonus when the favourite wins (so strong teams do not keep climbing)
    """

    def __init__(self, k=20.0, hfa=60.0, mov="none", xg_weight=0.0, xg_c=1.2,
                 carry=0.75, autocorr=False, name=None):
        self.k = k
        self.hfa = hfa
        self.mov = mov
        self.xg_weight = xg_weight
        self.xg_c = xg_c
        self.carry = carry
        self.autocorr = autocorr
        self.r = defaultdict(lambda: START_RATING)  # team id -> rating
        self.name = name or f"elo_{mov}"

    def start_season(self, year):
        for team in list(self.r):
            self.r[team] = START_RATING + self.carry * (self.r[team] - START_RATING)

    def gap(self, match):
        return self.r[match.home] - self.r[match.away] + self.hfa

    def update(self, match):
        gap = self.gap(match)
        expected = 1.0 / (1.0 + 10 ** (-gap / 400.0))
        goal_diff = match.hg - match.ag

        has_xg = not (math.isnan(match.hxg) or math.isnan(match.axg))
        if self.xg_weight > 0 and has_xg:
            # Score the match by chance quality as well as goals.
            margin = self.xg_weight * (match.hxg - match.axg) + (1 - self.xg_weight) * goal_diff
            actual = 1.0 / (1.0 + math.exp(-self.xg_c * margin))
            multiplier = 1.0
        else:
            actual = 1.0 if goal_diff > 0 else 0.5 if goal_diff == 0 else 0.0
            multiplier = margin_multiplier(self.mov, abs(goal_diff))
            if self.autocorr and goal_diff != 0:
                winner_gap = gap if goal_diff > 0 else -gap  # how much the winner was favoured
                multiplier *= 2.2 / (0.001 * winner_gap + 2.2)

        change = self.k * multiplier * (actual - expected)
        self.r[match.home] += change
        self.r[match.away] -= change
