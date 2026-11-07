"""Weighted ridge regression of goal difference on which teams played.

Each match is a row with +1 for the home team, -1 for the away team and a constant column
for home advantage. The team coefficients are the ratings. The gap is in goals.

Extra setting: ridge (strength of the penalty). The shared batch settings are in batch.py.
"""
import numpy as np

from ratings.models import batch
from ratings.models.batch import start_season, update   # same as every batch model


def new_model(ridge=5.0, half_life_days=365.0, refit_days=14, min_matches=200, target="goals",
              xg_weight=0.5, name=None):
    name = name or "ridge_gd" + ("_xg" if target == "xg" else "")
    model = batch.new_batch_model("ridge_gd", name, half_life_days, refit_days, min_matches, target, xg_weight)
    model["ridge"] = ridge
    model["coef"] = None   # [home advantage, team 0, team 1, ...]
    return model


def fit(model, home_idx, away_idx, home_values, away_values, weights, n_teams):
    n_matches = len(home_idx)
    rows = np.arange(n_matches)
    X = np.zeros((n_matches, n_teams + 1))
    X[:, 0] = 1.0                    # home advantage
    X[rows, home_idx + 1] += 1.0
    X[rows, away_idx + 1] -= 1.0
    y = home_values - away_values

    weighted_X = X * weights[:, None]
    penalty = np.eye(n_teams + 1) * model["ridge"]
    penalty[0, 0] = 0.0              # do not shrink the home advantage
    model["coef"] = np.linalg.solve(weighted_X.T @ X + penalty, weighted_X.T @ y)


def rating(model, team):
    return model["coef"][team + 1] if team is not None else 0.0


def team_gap(model, home, away):
    return float(model["coef"][0] + rating(model, home) - rating(model, away))


def gap(model, match):
    return batch.batch_gap(model, match, fit, team_gap)
