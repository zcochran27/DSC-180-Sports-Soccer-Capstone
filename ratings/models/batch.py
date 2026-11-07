"""Shared code for the batch models (dixon_coles.py and ridge_gd.py).

Batch models do not update after every match. They keep every past match and refit on all of them
every couple of weeks. Both weight recent matches more (an old match counts half as much every
`half_life_days`).

Settings shared by the batch models (new_batch_model arguments):
    half_life_days  an old match loses half its weight after this many days
    refit_days      how often to refit
    min_matches     do not fit until there are at least this many matches
    target          "goals", or "xg" to fit on a blend of xG and goals
    xg_weight       share of xG in that blend

Each batch model file provides a `fit_function` that fits its parameters on the weighted matches and a
`team_gap_function` that turns the fitted parameters into a gap for two teams.
"""
import numpy as np


def new_batch_model(kind, name, half_life_days, refit_days, min_matches, target, xg_weight):
    return {"kind": kind, "name": name, "half_life_days": half_life_days, "refit_days": refit_days,
            "min_matches": min_matches, "target": target, "xg_weight": xg_weight,
            "history": [],        # (date, home, away, home_value, away_value) for every match so far
            "team_index": {},     # team id -> position in the arrays used for fitting
            "last_fit_date": None,
            "is_fitted": False}


def match_values(model, match):
    """The home and away numbers we try to explain: goals, or a goals/xG blend."""
    if model["target"] == "xg" and not (np.isnan(match.hxg) or np.isnan(match.axg)):
        home = model["xg_weight"] * match.hxg + (1 - model["xg_weight"]) * match.hg
        away = model["xg_weight"] * match.axg + (1 - model["xg_weight"]) * match.ag
        return home, away
    return float(match.hg), float(match.ag)


def start_season(model, year):
    pass   # batch models have no ratings to regress; old matches simply lose weight with age


def update(model, match):
    home_value, away_value = match_values(model, match)
    model["history"].append((match.date, match.home, match.away, home_value, away_value))


def refit_if_due(model, today, fit_function):
    """Refit on all earlier matches if enough days have passed since the last fit."""
    history = model["history"]
    if len(history) < model["min_matches"]:
        return
    if model["last_fit_date"] is not None and (today - model["last_fit_date"]).days < model["refit_days"]:
        return

    # older matches get less weight
    dates = np.array([row[0] for row in history], dtype="datetime64[ns]")
    age_in_days = (np.datetime64(today, "ns") - dates) / np.timedelta64(1, "D")
    weights = 0.5 ** (age_in_days / model["half_life_days"])

    teams = sorted({row[1] for row in history} | {row[2] for row in history})
    model["team_index"] = {team: i for i, team in enumerate(teams)}
    home_idx = np.array([model["team_index"][row[1]] for row in history])
    away_idx = np.array([model["team_index"][row[2]] for row in history])
    home_values = np.array([row[3] for row in history])
    away_values = np.array([row[4] for row in history])

    fit_function(model, home_idx, away_idx, home_values, away_values, weights, len(teams))
    model["last_fit_date"] = today
    model["is_fitted"] = True


def batch_gap(model, match, fit_function, team_gap_function):
    """Gap for a match: refit if due, then ask the fitted parameters."""
    refit_if_due(model, match.date, fit_function)
    if not model["is_fitted"]:
        return 0.0
    # A team we have never seen gets index None, which the models treat as an average team.
    home = model["team_index"].get(match.home)
    away = model["team_index"].get(match.away)
    return team_gap_function(model, home, away)
