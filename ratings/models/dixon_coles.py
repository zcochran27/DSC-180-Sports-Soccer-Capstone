"""Dixon-Coles style Poisson model with an attack and a defence rating for every team.

    home goals ~ Poisson(exp(mu + hfa + attack_home - defence_away))
    away goals ~ Poisson(exp(mu + attack_away - defence_home))

A ridge penalty pulls every attack/defence rating toward zero. The gap is the expected home
goals minus the expected away goals. (The original paper's correction for low scores is left
out; it barely changes the gap.)

Extra setting: ridge (strength of the penalty). The shared batch settings are in batch.py.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from ratings.models import batch
from ratings.models.batch import start_season, update   # same as every batch model


def new_model(ridge=2.0, half_life_days=365.0, refit_days=14, min_matches=200, target="goals",
              xg_weight=0.5, name=None):
    name = name or "dixon_coles" + ("_xg" if target == "xg" else "")
    model = batch.new_batch_model("dixon_coles", name, half_life_days, refit_days, min_matches, target, xg_weight)
    model["ridge"] = ridge
    model["params"] = None   # [mu, hfa, attack (n values), defence (n values)]
    model["n_teams"] = 0
    return model


def penalised_loss(params, ridge, n_teams, home_idx, away_idx, home_values, away_values, weights):
    """Weighted Poisson negative log-likelihood plus the ridge penalty. Returns (loss, gradient)."""
    mu, hfa = params[0], params[1]
    attack = params[2:2 + n_teams]
    defence = params[2 + n_teams:]

    log_home_rate = mu + hfa + attack[home_idx] - defence[away_idx]
    log_away_rate = mu + attack[away_idx] - defence[home_idx]
    home_rate, away_rate = np.exp(log_home_rate), np.exp(log_away_rate)

    loss = (weights * (home_rate - home_values * log_home_rate + gammaln(home_values + 1))).sum()
    loss += (weights * (away_rate - away_values * log_away_rate + gammaln(away_values + 1))).sum()
    loss += ridge * ((attack ** 2).sum() + (defence ** 2).sum())

    home_error = weights * (home_rate - home_values)
    away_error = weights * (away_rate - away_values)
    grad = np.zeros(len(params))
    grad[0] = home_error.sum() + away_error.sum()
    grad[1] = home_error.sum()
    grad[2:2 + n_teams] = (np.bincount(home_idx, home_error, n_teams)
                           + np.bincount(away_idx, away_error, n_teams)
                           + 2 * ridge * attack)
    grad[2 + n_teams:] = (-np.bincount(away_idx, home_error, n_teams)
                          - np.bincount(home_idx, away_error, n_teams)
                          + 2 * ridge * defence)
    return loss, grad


def fit(model, home_idx, away_idx, home_values, away_values, weights, n_teams):
    n_params = 2 + 2 * n_teams

    # Start from the previous fit when the number of teams has not changed.
    if model["params"] is not None and len(model["params"]) == n_params:
        start = model["params"].copy()
    else:
        start = np.zeros(n_params)
        start[0] = np.log(max(home_values.mean(), 0.1))

    result = minimize(penalised_loss, start,
                      args=(model["ridge"], n_teams, home_idx, away_idx, home_values, away_values, weights),
                      jac=True, method="L-BFGS-B", options={"maxiter": 200})
    model["params"] = result.x
    model["n_teams"] = n_teams


def team_attack(model, team):
    return model["params"][2 + team] if team is not None else 0.0


def team_defence(model, team):
    return model["params"][2 + model["n_teams"] + team] if team is not None else 0.0


def team_gap(model, home, away):
    mu, hfa = model["params"][0], model["params"][1]
    home_rate = np.exp(mu + hfa + team_attack(model, home) - team_defence(model, away))
    away_rate = np.exp(mu + team_attack(model, away) - team_defence(model, home))
    return float(home_rate - away_rate)


def gap(model, match):
    return batch.batch_gap(model, match, fit, team_gap)
