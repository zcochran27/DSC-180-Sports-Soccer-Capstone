"""Shared interface, outcome mapping and walk-forward evaluation for rating models.

Every model exposes `gap(m)`: pre-match strength of home minus away (home advantage included,
native units) and `update(m)` to absorb the result. `OutcomeMapper` turns a gap into W/D/L
probabilities (ordered logit) and an expected goal difference (linear), so all models are
scored on the same footing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit


class RatingModel:
    name = "base"

    def start_season(self, year: int) -> None:  # called when the season year changes
        pass

    def gap(self, m) -> float:  # m is a namedtuple row from itertuples()
        raise NotImplementedError

    def update(self, m) -> None:
        raise NotImplementedError


def walk_forward(model: RatingModel, matches: pd.DataFrame) -> pd.DataFrame:
    """Predict each match from pre-match state, then update. Returns gaps (no look-ahead)."""
    gaps, last_year = [], None
    for m in matches.itertuples(index=False):
        if m.year != last_year:
            model.start_season(m.year)
            last_year = m.year
        gaps.append(model.gap(m))
        model.update(m)
    out = matches[["match_id", "year", "date", "home", "away", "hg", "ag", "hxg", "axg"]].copy()
    out["gap"] = gaps
    return out


@dataclass
class OutcomeMapper:
    """gap -> P(home win/draw/away win) via ordered logit, and gap -> expected goal diff."""
    beta: float = 1.0
    cut: float = 0.5  # half-width of the draw band on the latent scale
    slope: float = 0.0
    intercept: float = 0.0

    def fit(self, gap, hg, ag) -> "OutcomeMapper":
        gap, gd = np.asarray(gap, float), np.asarray(hg, float) - np.asarray(ag, float)
        s = gap.std() or 1.0
        z = gap / s
        res = np.sign(gd)

        def nll(p):
            b, c = p[0], np.exp(p[1])
            ph = expit(b * z - c)  # P(home win) = P(latent > c)
            pa = expit(-b * z - c)
            pd_ = np.clip(1 - ph - pa, 1e-9, 1)
            pr = np.where(res > 0, ph, np.where(res < 0, pa, pd_))
            return -np.log(np.clip(pr, 1e-12, 1)).sum()

        r = minimize(nll, [1.0, np.log(0.5)], method="Nelder-Mead")
        self.beta, self.cut = r.x[0] / s, float(np.exp(r.x[1]))
        A = np.c_[gap, np.ones_like(gap)]
        self.slope, self.intercept = np.linalg.lstsq(A, gd, rcond=None)[0]
        return self

    def probs(self, gap) -> np.ndarray:
        gap = np.asarray(gap, float)
        ph = expit(self.beta * gap - self.cut)
        pa = expit(-self.beta * gap - self.cut)
        return np.c_[ph, np.clip(1 - ph - pa, 0, 1), pa]

    def exp_gd(self, gap) -> np.ndarray:
        return self.slope * np.asarray(gap, float) + self.intercept


def score(df: pd.DataFrame, mapper: OutcomeMapper) -> dict:
    """Log loss, RPS, Brier, accuracy, and goal-diff / xG-diff errors for a predictions frame."""
    p = mapper.probs(df["gap"].values)
    gd = (df["hg"] - df["ag"]).values
    y = np.where(gd > 0, 0, np.where(gd == 0, 1, 2))
    oh = np.eye(3)[y]
    ll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()
    rps = (((np.cumsum(p, 1) - np.cumsum(oh, 1))[:, :2]) ** 2).sum(1).mean() / 2
    brier = ((p - oh) ** 2).sum(1).mean()
    eg = mapper.exp_gd(df["gap"].values)
    out = {"n": len(df), "log_loss": ll, "rps": rps, "brier": brier,
           "accuracy": (p.argmax(1) == y).mean(), "gd_mae": np.abs(eg - gd).mean(),
           "gd_rmse": np.sqrt(((eg - gd) ** 2).mean())}
    xd = (df["hxg"] - df["axg"]).values
    ok = ~np.isnan(xd)
    out["xgd_mae"] = np.abs(eg[ok] - xd[ok]).mean() if ok.any() else np.nan
    return out
