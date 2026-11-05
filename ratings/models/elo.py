"""Elo variants: baseline, margin-of-victory (World-Football style), and xG-blended."""
from __future__ import annotations

import math
from collections import defaultdict

from ratings.base import RatingModel


def mov_multiplier(kind: str, n: int) -> float:
    if kind == "none" or n <= 1:
        return 1.0
    if kind == "sqrt":
        return math.sqrt(n)
    if kind == "log":
        return math.log(n + 1) + 1.0
    if kind == "wfe":  # eloratings.net: 1.5 for 2 goals, (11+N)/8 for 3+
        return 1.5 if n == 2 else (11 + n) / 8
    raise ValueError(kind)


class Elo(RatingModel):
    """Elo with home advantage `hfa`, step `k`, optional MOV multiplier and xG blending.

    mov: 'none'|'sqrt'|'log'|'wfe'. xg_weight w>0 replaces the W/D/L result by
    sigmoid(c * (w*xGdiff + (1-w)*GD)) so the update uses chance quality, not just goals.
    autocorr damps the margin boost when the favourite wins (538-style).
    carry: share of (rating - mean) kept across a season boundary.
    """

    def __init__(self, k=20.0, hfa=60.0, mov="none", xg_weight=0.0, xg_c=1.2,
                 carry=0.75, autocorr=False, name=None):
        self.k, self.hfa, self.mov, self.w, self.c = k, hfa, mov, xg_weight, xg_c
        self.carry, self.autocorr = carry, autocorr
        self.r = defaultdict(lambda: 1500.0)
        self.name = name or f"elo_{mov}" + (f"_xg{xg_weight:g}" if xg_weight else "")

    def start_season(self, year):
        for t in list(self.r):
            self.r[t] = 1500.0 + self.carry * (self.r[t] - 1500.0)

    def gap(self, m):
        return self.r[m.home] - self.r[m.away] + self.hfa

    def update(self, m):
        d = self.gap(m)
        e = 1.0 / (1.0 + 10 ** (-d / 400.0))
        gd = m.hg - m.ag
        if self.w > 0 and not math.isnan(m.hxg) and not math.isnan(m.axg):
            margin = self.w * (m.hxg - m.axg) + (1 - self.w) * gd
            s = 1.0 / (1.0 + math.exp(-self.c * margin))
            mult = 1.0
        else:
            s = 1.0 if gd > 0 else 0.5 if gd == 0 else 0.0
            mult = mov_multiplier(self.mov, abs(gd))
            if self.autocorr and gd != 0:
                winner_diff = d if gd > 0 else -d
                mult *= 2.2 / (0.001 * winner_diff + 2.2)
        delta = self.k * mult * (s - e)
        self.r[m.home] += delta
        self.r[m.away] -= delta
