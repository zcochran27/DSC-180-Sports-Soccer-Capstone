"""Batch models: instead of updating after every match, they refit on all earlier matches
every couple of weeks. Two are here: a Dixon-Coles style Poisson model and a ridge regression.

Both weight recent matches more (an old match counts half as much every `half_life_days`).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from ratings.base import RatingModel


class BatchModel(RatingModel):
    """Keeps every past match and refits when `refit_days` have passed since the last fit.

    Subclasses implement _fit(...) and _team_gap(home_index, away_index).

    half_life_days  an old match loses half its weight after this many days
    refit_days      how often to refit
    min_matches     do not fit until there are at least this many matches
    target          "goals", or "xg" to fit on a blend of xG and goals
    xg_weight       share of xG in that blend
    """

    def __init__(self, half_life_days=365.0, refit_days=14, min_matches=200, target="goals", xg_weight=0.5):
        self.half_life_days = half_life_days
        self.refit_days = refit_days
        self.min_matches = min_matches
        self.target = target
        self.xg_weight = xg_weight
        self.history = []        # (date, home, away, home_value, away_value) for every match so far
        self.team_index = {}     # team id -> position in the arrays used for fitting
        self.last_fit_date = None
        self.is_fitted = False

    def _match_values(self, match):
        """The home and away numbers we try to explain: goals, or a goals/xG blend."""
        if self.target == "xg" and not (np.isnan(match.hxg) or np.isnan(match.axg)):
            home = self.xg_weight * match.hxg + (1 - self.xg_weight) * match.hg
            away = self.xg_weight * match.axg + (1 - self.xg_weight) * match.ag
            return home, away
        return float(match.hg), float(match.ag)

    def update(self, match):
        home_value, away_value = self._match_values(match)
        self.history.append((match.date, match.home, match.away, home_value, away_value))

    def _refit_if_due(self, today):
        if len(self.history) < self.min_matches:
            return
        if self.last_fit_date is not None and (today - self.last_fit_date).days < self.refit_days:
            return

        dates = np.array([row[0] for row in self.history], dtype="datetime64[ns]")
        age_in_days = (np.datetime64(today, "ns") - dates) / np.timedelta64(1, "D")
        weights = 0.5 ** (age_in_days / self.half_life_days)

        teams = sorted({row[1] for row in self.history} | {row[2] for row in self.history})
        self.team_index = {team: i for i, team in enumerate(teams)}
        home_idx = np.array([self.team_index[row[1]] for row in self.history])
        away_idx = np.array([self.team_index[row[2]] for row in self.history])
        home_values = np.array([row[3] for row in self.history])
        away_values = np.array([row[4] for row in self.history])

        self._fit(home_idx, away_idx, home_values, away_values, weights, len(teams))
        self.last_fit_date = today
        self.is_fitted = True

    def gap(self, match):
        self._refit_if_due(match.date)
        if not self.is_fitted:
            return 0.0
        # A team we have never seen gets index None, which the models treat as an average team.
        return self._team_gap(self.team_index.get(match.home), self.team_index.get(match.away))


class DixonColes(BatchModel):
    """Poisson model with an attack and a defence rating for every team.

        home goals ~ Poisson(exp(mu + hfa + attack_home - defence_away))
        away goals ~ Poisson(exp(mu + attack_away - defence_home))

    A ridge penalty pulls every attack/defence rating toward zero. The gap is the expected home
    goals minus the expected away goals. (The original paper's correction for low scores is left
    out; it barely changes the gap.)
    """

    def __init__(self, ridge=2.0, **kwargs):
        super().__init__(**kwargs)
        self.ridge = ridge
        self.name = "dixon_coles" + ("_xg" if self.target == "xg" else "")
        self.params = None  # [mu, hfa, attack (n values), defence (n values)]
        self.n_teams = 0

    def _fit(self, home_idx, away_idx, home_values, away_values, weights, n_teams):
        n_params = 2 + 2 * n_teams

        # Start from the previous fit when the number of teams has not changed.
        if self.params is not None and len(self.params) == n_params:
            start = self.params.copy()
        else:
            start = np.zeros(n_params)
            start[0] = np.log(max(home_values.mean(), 0.1))

        def penalised_loss(params):
            """Returns the loss and its gradient (the optimiser uses both)."""
            mu, hfa = params[0], params[1]
            attack = params[2:2 + n_teams]
            defence = params[2 + n_teams:]

            log_home_rate = mu + hfa + attack[home_idx] - defence[away_idx]
            log_away_rate = mu + attack[away_idx] - defence[home_idx]
            home_rate, away_rate = np.exp(log_home_rate), np.exp(log_away_rate)

            # Weighted Poisson negative log-likelihood plus the ridge penalty.
            loss = (weights * (home_rate - home_values * log_home_rate + gammaln(home_values + 1))).sum()
            loss += (weights * (away_rate - away_values * log_away_rate + gammaln(away_values + 1))).sum()
            loss += self.ridge * ((attack ** 2).sum() + (defence ** 2).sum())

            home_error = weights * (home_rate - home_values)
            away_error = weights * (away_rate - away_values)
            grad = np.zeros(n_params)
            grad[0] = home_error.sum() + away_error.sum()
            grad[1] = home_error.sum()
            grad[2:2 + n_teams] = (np.bincount(home_idx, home_error, n_teams)
                                   + np.bincount(away_idx, away_error, n_teams)
                                   + 2 * self.ridge * attack)
            grad[2 + n_teams:] = (-np.bincount(away_idx, home_error, n_teams)
                                  - np.bincount(home_idx, away_error, n_teams)
                                  + 2 * self.ridge * defence)
            return loss, grad

        result = minimize(penalised_loss, start, jac=True, method="L-BFGS-B", options={"maxiter": 200})
        self.params = result.x
        self.n_teams = n_teams

    def _attack(self, team):
        return self.params[2 + team] if team is not None else 0.0

    def _defence(self, team):
        return self.params[2 + self.n_teams + team] if team is not None else 0.0

    def _team_gap(self, home, away):
        mu, hfa = self.params[0], self.params[1]
        home_rate = np.exp(mu + hfa + self._attack(home) - self._defence(away))
        away_rate = np.exp(mu + self._attack(away) - self._defence(home))
        return float(home_rate - away_rate)


class RidgeRating(BatchModel):
    """Weighted ridge regression of goal difference on which teams played.

    Each match is a row with +1 for the home team, -1 for the away team and a constant column
    for home advantage. The team coefficients are the ratings. The gap is in goals.
    """

    def __init__(self, ridge=5.0, **kwargs):
        super().__init__(**kwargs)
        self.ridge = ridge
        self.name = "ridge_gd" + ("_xg" if self.target == "xg" else "")
        self.coef = None  # [home advantage, team 0, team 1, ...]

    def _fit(self, home_idx, away_idx, home_values, away_values, weights, n_teams):
        n_matches = len(home_idx)
        rows = np.arange(n_matches)
        X = np.zeros((n_matches, n_teams + 1))
        X[:, 0] = 1.0                 # home advantage
        X[rows, home_idx + 1] += 1.0
        X[rows, away_idx + 1] -= 1.0
        y = home_values - away_values

        weighted_X = X * weights[:, None]
        penalty = np.eye(n_teams + 1) * self.ridge
        penalty[0, 0] = 0.0           # do not shrink the home advantage
        self.coef = np.linalg.solve(weighted_X.T @ X + penalty, weighted_X.T @ y)

    def _team_gap(self, home, away):
        def rating(team):
            return self.coef[team + 1] if team is not None else 0.0
        return float(self.coef[0] + rating(home) - rating(away))
