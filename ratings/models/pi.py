"""Pi-ratings (Constantinou & Fenton 2013): separate home/away ratings in goal units."""
from __future__ import annotations

import math
from collections import defaultdict

from ratings.base import RatingModel


def _psi(e: float, c: float) -> float:
    return math.copysign(c * math.log10(1 + abs(e)), e)


class PiRatings(RatingModel):
    name = "pi"

    def __init__(self, lam=0.06, gamma=0.5, c=3.0, carry=0.8, scale=200.0):
        self.lam, self.gamma, self.c, self.carry, self.scale = lam, gamma, c, carry, scale
        self.rh = defaultdict(float)  # expected GD vs an average team when playing at home
        self.ra = defaultdict(float)  # ... when playing away

    def start_season(self, year):
        for d in (self.rh, self.ra):
            for t in d:
                d[t] *= self.carry

    def exp_gd(self, m):
        return self.rh[m.home] - self.ra[m.away]

    def gap(self, m):
        return self.exp_gd(m) * self.scale

    def update(self, m):
        err = (m.hg - m.ag) - self.exp_gd(m)
        psi = _psi(err, self.c)
        dh = psi * self.lam
        self.rh[m.home] += dh
        self.ra[m.home] += dh * self.gamma
        da = -psi * self.lam
        self.ra[m.away] += da
        self.rh[m.away] += da * self.gamma
