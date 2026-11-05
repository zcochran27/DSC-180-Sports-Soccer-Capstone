"""Glicko-style rating: an Elo-like rating plus an uncertainty ("rating deviation", RD).

A team's RD grows while it sits idle and when a new season starts, and shrinks as it plays.
A team with a large RD (few or old games) has its rating moved more by each result.
"""
from __future__ import annotations

import math
from collections import defaultdict

from ratings.base import RatingModel

Q = math.log(10) / 400.0
START_RATING = 1500.0


def opponent_weight(rd: float) -> float:
    """Results against an opponent whose rating is uncertain count for less."""
    return 1.0 / math.sqrt(1.0 + 3 * Q * Q * rd * rd / math.pi ** 2)


class Glicko(RatingModel):
    """
    hfa        home advantage, in rating points
    rd0        starting RD for a new team
    rd_max     RD never grows beyond this
    c_week     how much RD grows per week without a match
    season_c   how much RD jumps at the start of a season
    carry      share of a team's rating above/below 1500 kept when a new season starts
    """
    name = "glicko"

    def __init__(self, hfa=60.0, rd0=200.0, rd_max=250.0, c_week=6.0, season_c=120.0, carry=0.75):
        self.hfa = hfa
        self.rd0 = rd0
        self.rd_max = rd_max
        self.c_week = c_week
        self.season_c = season_c
        self.carry = carry
        self.r = defaultdict(lambda: START_RATING)  # team id -> rating
        self.rd = defaultdict(lambda: rd0)          # team id -> rating deviation
        self.last_played = {}                       # team id -> date of last match

    def start_season(self, year):
        for team in list(self.r):
            self.r[team] = START_RATING + self.carry * (self.r[team] - START_RATING)
            self.rd[team] = min(math.hypot(self.rd[team], self.season_c), self.rd_max)

    def _grow_rd_for_idle_time(self, team, date):
        if team in self.last_played:
            weeks_idle = max((date - self.last_played[team]).days, 0) / 7.0
            new_rd = math.sqrt(self.rd[team] ** 2 + self.c_week ** 2 * weeks_idle)
            self.rd[team] = min(new_rd, self.rd_max)

    def gap(self, match):
        return self.r[match.home] - self.r[match.away] + self.hfa

    def gap_sd(self, match):
        """Uncertainty of the gap, from both teams' RD."""
        return math.hypot(self.rd[match.home], self.rd[match.away])

    def update(self, match):
        for team in (match.home, match.away):
            self._grow_rd_for_idle_time(team, match.date)

        goal_diff = match.hg - match.ag
        home_score = 1.0 if goal_diff > 0 else 0.5 if goal_diff == 0 else 0.0

        # Work out both teams' new values before applying either, so neither uses the other's new rating.
        new_values = {}
        sides = ((match.home, match.away, home_score, self.hfa),
                 (match.away, match.home, 1 - home_score, -self.hfa))
        for team, opponent, score, advantage in sides:
            weight = opponent_weight(self.rd[opponent])
            rating_diff = self.r[team] + advantage - self.r[opponent]
            expected = 1.0 / (1.0 + 10 ** (-weight * rating_diff / 400.0))
            precision = 1.0 / self.rd[team] ** 2 + Q * Q * weight * weight * expected * (1 - expected)
            new_rating = self.r[team] + Q / precision * weight * (score - expected)
            new_values[team] = (new_rating, math.sqrt(1.0 / precision))

        for team, (rating, rd) in new_values.items():
            self.r[team] = rating
            self.rd[team] = rd
            self.last_played[team] = match.date
