"""Shared pieces for every rating model: the walk-forward runner, the mapping from rating gap
to match outcomes, and the scoring metrics.

A rating model is a plain dict (see ratings/specs.py and the files in ratings/models/). Every kind of
model has three functions:
    gap(model, match)           -> home strength minus away strength before the match (home advantage included)
    update(model, match)        -> learn from the result of the match
    start_season(model, year)   -> called when the first match of a new season is reached
                                   (most models pull their ratings towards the average here)
`match` is one row from matches.itertuples().

Because all models produce a "gap", we can turn any of them into win/draw/loss probabilities
with the same outcome mapper and score them all the same way.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from ratings.models import constant, dixon_coles, elo, glicko, pi, ridge_gd

# model["kind"] -> the file with that kind's gap / update / start_season functions
MODEL_MODULES = {"constant": constant, "elo": elo, "glicko": glicko, "pi": pi,
                 "dixon_coles": dixon_coles, "ridge_gd": ridge_gd}


# ---------------------------------------------------------------- calling any model

def start_season(model, year):
    MODEL_MODULES[model["kind"]].start_season(model, year)


def gap(model, match):
    return MODEL_MODULES[model["kind"]].gap(model, match)


def update(model, match):
    MODEL_MODULES[model["kind"]].update(model, match)


def walk_forward(model, matches):
    """Go through matches in date order, record the model's gap *before* each match, then update.

    Recording the gap before updating means the model never sees a result it is predicting.
    """
    gaps = []
    last_year = None
    for match in matches.itertuples(index=False):
        if match.year != last_year:
            start_season(model, match.year)
            last_year = match.year
        gaps.append(gap(model, match))
        update(model, match)

    predictions = matches[["match_id", "year", "date", "home", "away", "hg", "ag", "hxg", "axg"]].copy()
    predictions["gap"] = gaps
    return predictions


# ---------------------------------------------------------------- gap -> outcomes

# An outcome mapper is a dict with four numbers:
#   beta, cut         ordered logit: a larger gap makes a home win more likely, and a draw happens when
#                     the latent result falls inside a band of half-width `cut` around zero
#   slope, intercept  expected goal difference = slope * gap + intercept (a straight line)

def ordered_logit_loss(params, z, result):
    """Negative log-likelihood of the results given a standardised gap z (what the optimiser minimises)."""
    beta, cut = params[0], np.exp(params[1])   # exp keeps the cut positive
    p_home = expit(beta * z - cut)
    p_away = expit(-beta * z - cut)
    p_draw = np.clip(1 - p_home - p_away, 1e-9, 1)
    p_actual = np.where(result > 0, p_home, np.where(result < 0, p_away, p_draw))
    return -np.log(np.clip(p_actual, 1e-12, 1)).sum()


def fit_outcome_mapper(gap_values, home_goals, away_goals):
    """Fit the mapper from rating gaps to match results."""
    gap_values = np.asarray(gap_values, float)
    goal_diff = np.asarray(home_goals, float) - np.asarray(away_goals, float)
    result = np.sign(goal_diff)   # 1 = home win, 0 = draw, -1 = away win

    # Work with a standardised gap so the optimiser starts from sensible numbers.
    spread = gap_values.std() or 1.0
    z = gap_values / spread
    best = minimize(ordered_logit_loss, [1.0, np.log(0.5)], args=(z, result), method="Nelder-Mead")

    # Expected goal difference: least-squares line through (gap, goal difference).
    design = np.c_[gap_values, np.ones_like(gap_values)]
    slope, intercept = np.linalg.lstsq(design, goal_diff, rcond=None)[0]

    return {"beta": best.x[0] / spread, "cut": float(np.exp(best.x[1])),
            "slope": slope, "intercept": intercept}


def outcome_probs(mapper, gap_values):
    """One row per gap: [P(home win), P(draw), P(away win)]."""
    gap_values = np.asarray(gap_values, float)
    p_home = expit(mapper["beta"] * gap_values - mapper["cut"])
    p_away = expit(-mapper["beta"] * gap_values - mapper["cut"])
    p_draw = np.clip(1 - p_home - p_away, 0, 1)
    return np.c_[p_home, p_draw, p_away]


def expected_goal_diff(mapper, gap_values):
    """Expected goal difference (home minus away)."""
    return mapper["slope"] * np.asarray(gap_values, float) + mapper["intercept"]


# ---------------------------------------------------------------- scoring

def score(predictions, mapper):
    """Score predictions (a frame from walk_forward). Lower is better for everything except accuracy."""
    probs = outcome_probs(mapper, predictions["gap"].values)
    goal_diff = (predictions["hg"] - predictions["ag"]).values

    # Actual result as a column index: 0 = home win, 1 = draw, 2 = away win.
    actual = np.where(goal_diff > 0, 0, np.where(goal_diff == 0, 1, 2))
    actual_onehot = np.eye(3)[actual]

    log_loss = -np.log(np.clip(probs[np.arange(len(actual)), actual], 1e-12, 1)).mean()
    brier = ((probs - actual_onehot) ** 2).sum(axis=1).mean()
    # Ranked probability score: compares cumulative probabilities (home, home+draw).
    cumulative_gap = np.cumsum(probs, axis=1) - np.cumsum(actual_onehot, axis=1)
    rps = (cumulative_gap[:, :2] ** 2).sum(axis=1).mean() / 2

    expected_gd = expected_goal_diff(mapper, predictions["gap"].values)
    results = {
        "n": len(predictions),
        "log_loss": log_loss,
        "rps": rps,
        "brier": brier,
        "accuracy": (probs.argmax(axis=1) == actual).mean(),
        "gd_mae": np.abs(expected_gd - goal_diff).mean(),
        "gd_rmse": np.sqrt(((expected_gd - goal_diff) ** 2).mean()),
    }

    # Same comparison against xG difference, only for matches that have xG.
    xg_diff = (predictions["hxg"] - predictions["axg"]).values
    has_xg = ~np.isnan(xg_diff)
    results["xgd_mae"] = np.abs(expected_gd[has_xg] - xg_diff[has_xg]).mean() if has_xg.any() else np.nan
    return results
