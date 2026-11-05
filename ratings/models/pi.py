"""Pi-ratings (Constantinou & Fenton, 2013).

Each team has two ratings measured in goals: how well it does at home and how well it does away.
The expected goal difference of a match is the home team's home rating minus the away team's away
rating. After the match each rating moves toward the result.
"""
from __future__ import annotations

import math
from collections import defaultdict

from ratings.base import RatingModel


def damp(error: float, c: float) -> float:
    """Shrink big errors so one blowout does not move ratings too much."""
    return math.copysign(c * math.log10(1 + abs(error)), error)


class PiRatings(RatingModel):
    """
    lam     learning rate
    gamma   share of a team's update that also goes to its other (home/away) rating
    c       how strongly big errors are damped
    carry   share of each rating kept when a new season starts
    scale   converts goals to Elo-like units so the gap is comparable to the other models
    """
    name = "pi"

    def __init__(self, lam=0.06, gamma=0.5, c=3.0, carry=0.8, scale=200.0):
        self.lam = lam
        self.gamma = gamma
        self.c = c
        self.carry = carry
        self.scale = scale
        self.home_rating = defaultdict(float)  # expected goal diff vs an average team, at home
        self.away_rating = defaultdict(float)  # same, but away

    def start_season(self, year):
        for ratings in (self.home_rating, self.away_rating):
            for team in ratings:
                ratings[team] *= self.carry

    def expected_goal_diff(self, match):
        return self.home_rating[match.home] - self.away_rating[match.away]

    def gap(self, match):
        return self.expected_goal_diff(match) * self.scale

    def update(self, match):
        error = (match.hg - match.ag) - self.expected_goal_diff(match)
        step = damp(error, self.c) * self.lam

        # Home team: its home rating moves by `step`, its away rating by a fraction of that.
        self.home_rating[match.home] += step
        self.away_rating[match.home] += step * self.gamma
        # Away team: moves the opposite way.
        self.away_rating[match.away] -= step
        self.home_rating[match.away] -= step * self.gamma
