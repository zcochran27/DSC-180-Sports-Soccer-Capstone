"""Glicko-style rating with explicit uncertainty (a 1-D Kalman filter per team)."""
from __future__ import annotations

import math
from collections import defaultdict

from ratings.base import RatingModel

Q = math.log(10) / 400.0


def _g(rd: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3 * Q * Q * rd * rd / math.pi ** 2)


class Glicko(RatingModel):
    """RD grows with idle time (`c_week`) and jumps at season boundaries (`season_c`),
    so teams with few/old games move fast. `gap_sd(m)` gives the uncertainty of a matchup."""
    name = "glicko"

    def __init__(self, hfa=60.0, rd0=200.0, rd_max=250.0, c_week=6.0, season_c=120.0, carry=0.75):
        self.hfa, self.rd0, self.rd_max = hfa, rd0, rd_max
        self.c_week, self.season_c, self.carry = c_week, season_c, carry
        self.r = defaultdict(lambda: 1500.0)
        self.rd = defaultdict(lambda: rd0)
        self.last = {}

    def start_season(self, year):
        for t in list(self.r):
            self.r[t] = 1500.0 + self.carry * (self.r[t] - 1500.0)
            self.rd[t] = min(math.hypot(self.rd[t], self.season_c), self.rd_max)

    def _inflate(self, team, date):
        if team in self.last:
            weeks = max((date - self.last[team]).days, 0) / 7.0
            self.rd[team] = min(math.sqrt(self.rd[team] ** 2 + self.c_week ** 2 * weeks), self.rd_max)

    def gap(self, m):
        return self.r[m.home] - self.r[m.away] + self.hfa

    def gap_sd(self, m):
        return math.hypot(self.rd[m.home], self.rd[m.away])

    def update(self, m):
        for t in (m.home, m.away):
            self._inflate(t, m.date)
        gd = m.hg - m.ag
        s_home = 1.0 if gd > 0 else 0.5 if gd == 0 else 0.0
        new = {}
        for team, opp, s, adj in ((m.home, m.away, s_home, self.hfa),
                                  (m.away, m.home, 1 - s_home, -self.hfa)):
            g = _g(self.rd[opp])
            e = 1.0 / (1.0 + 10 ** (-g * (self.r[team] + adj - self.r[opp]) / 400.0))
            prec = 1.0 / self.rd[team] ** 2 + Q * Q * g * g * e * (1 - e)
            new[team] = (self.r[team] + Q / prec * g * (s - e), math.sqrt(1.0 / prec))
        for t, (r, rd) in new.items():  # apply simultaneously
            self.r[t], self.rd[t] = r, rd
            self.last[t] = m.date
