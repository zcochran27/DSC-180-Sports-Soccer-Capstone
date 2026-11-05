"""Batch (refit-periodically) models: Dixon-Coles Poisson and ridge goal-difference ratings."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from ratings.base import RatingModel


class _Batch(RatingModel):
    """Stores history; refits on matches strictly before the current date every `refit_days`."""

    def __init__(self, half_life_days=365.0, refit_days=14, min_matches=200, target="goals", xg_weight=0.5):
        self.hl, self.refit_days, self.min_matches = half_life_days, refit_days, min_matches
        self.target, self.xgw = target, xg_weight
        self.hist = []
        self.idx, self.last_fit = {}, None
        self.fitted = False

    def _y(self, m):
        if self.target == "xg" and not (np.isnan(m.hxg) or np.isnan(m.axg)):
            return (self.xgw * m.hxg + (1 - self.xgw) * m.hg,
                    self.xgw * m.axg + (1 - self.xgw) * m.ag)
        return float(m.hg), float(m.ag)

    def update(self, m):
        self.hist.append((m.date, m.home, m.away, *self._y(m)))

    def _maybe_fit(self, date):
        if len(self.hist) < self.min_matches:
            return
        if self.last_fit is None or (date - self.last_fit).days >= self.refit_days:
            dates = np.array([h[0] for h in self.hist], dtype="datetime64[ns]")
            age = (np.datetime64(date, "ns") - dates) / np.timedelta64(1, "D")
            w = 0.5 ** (age / self.hl)
            teams = sorted({h[1] for h in self.hist} | {h[2] for h in self.hist})
            self.idx = {t: i for i, t in enumerate(teams)}
            hi = np.array([self.idx[h[1]] for h in self.hist])
            ai = np.array([self.idx[h[2]] for h in self.hist])
            yh = np.array([h[3] for h in self.hist])
            ya = np.array([h[4] for h in self.hist])
            self._fit(hi, ai, yh, ya, w, len(teams))
            self.last_fit, self.fitted = date, True

    def gap(self, m):
        self._maybe_fit(m.date)
        if not self.fitted:
            return 0.0
        return self._gap(self.idx.get(m.home), self.idx.get(m.away))


class DixonColes(_Batch):
    """Time-decayed Poisson attack/defence model with ridge shrinkage.

    gap = expected home goals - expected away goals. target='xg' fits on a goals/xG blend.
    (The low-score rho correction is omitted: it barely moves gap, which is what we score.)
    """

    def __init__(self, ridge=2.0, **kw):
        super().__init__(**kw)
        self.ridge = ridge
        self.name = "dixon_coles" + ("_xg" if self.target == "xg" else "")
        self.theta = None

    def _fit(self, hi, ai, yh, ya, w, n):
        # theta = [mu, hfa, att(n), dfn(n)]; log lam_h = mu+hfa+att_h-dfn_a, log lam_a = mu+att_a-dfn_h
        k = 2 + 2 * n
        if self.theta is not None and len(self.theta) == k:
            x0 = self.theta.copy()
        else:
            x0 = np.zeros(k)
            x0[0] = np.log(max(yh.mean(), 0.1))

        def f(th):
            mu, hfa, att, dfn = th[0], th[1], th[2:2 + n], th[2 + n:]
            lh = mu + hfa + att[hi] - dfn[ai]
            la = mu + att[ai] - dfn[hi]
            eh, ea = np.exp(lh), np.exp(la)
            nll = (w * (eh - yh * lh + gammaln(yh + 1))).sum() + (w * (ea - ya * la + gammaln(ya + 1))).sum()
            nll += self.ridge * ((att ** 2).sum() + (dfn ** 2).sum())
            rh, ra = w * (eh - yh), w * (ea - ya)
            g = np.zeros(k)
            g[0] = rh.sum() + ra.sum()
            g[1] = rh.sum()
            g[2:2 + n] = np.bincount(hi, rh, n) + np.bincount(ai, ra, n) + 2 * self.ridge * att
            g[2 + n:] = -np.bincount(ai, rh, n) - np.bincount(hi, ra, n) + 2 * self.ridge * dfn
            return nll, g

        res = minimize(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": 200})
        self.theta, self.n = res.x, n

    def _gap(self, i, j):
        th, n = self.theta, self.n
        att = lambda t: th[2 + t] if t is not None else 0.0
        dfn = lambda t: th[2 + n + t] if t is not None else 0.0
        lh = th[0] + th[1] + att(i) - dfn(j)
        la = th[0] + att(j) - dfn(i)
        return float(np.exp(lh) - np.exp(la))


class RidgeRating(_Batch):
    """Weighted ridge regression of goal difference on team indicators (+1 home, -1 away) and
    a home-advantage intercept. Batch analogue of Elo/Bradley-Terry; gap in goals."""

    def __init__(self, ridge=5.0, **kw):
        super().__init__(**kw)
        self.ridge = ridge
        self.name = "ridge_gd" + ("_xg" if self.target == "xg" else "")

    def _fit(self, hi, ai, yh, ya, w, n):
        m = len(hi)
        X = np.zeros((m, n + 1))
        X[:, 0] = 1.0
        X[np.arange(m), hi + 1] += 1.0
        X[np.arange(m), ai + 1] -= 1.0
        y = yh - ya
        Xw = X * w[:, None]
        P = np.eye(n + 1) * self.ridge
        P[0, 0] = 0.0
        self.beta = np.linalg.solve(Xw.T @ X + P, Xw.T @ y)

    def _gap(self, i, j):
        s = lambda t: self.beta[t + 1] if t is not None else 0.0
        return float(self.beta[0] + s(i) - s(j))
