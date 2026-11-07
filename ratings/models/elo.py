"""Elo rating models.

Every team starts at 1500. Before a match the expected score of the home team is

    expected = 1 / (1 + 10 ** (-gap / 400))        where gap = home rating - away rating + hfa

After the match the home team gains (and the away team loses) k * (actual - expected), where
actual is 1 for a win, 0.5 for a draw and 0 for a loss.

Settings of a model (new_model arguments):
    k          size of each rating update
    hfa        home advantage, in rating points
    mov        margin-of-victory scaling: "none", "sqrt", "log" or "wfe"
    xg_weight  if above 0, the "actual" score blends xG difference and goal difference
               (see update) instead of using win/draw/loss
    xg_c       how steeply that blended margin is turned into a 0-1 score
    carry      share of a team's rating above/below 1500 kept when a new season starts
    autocorr   shrink the margin bonus when the favourite wins (so strong teams do not keep climbing)
"""
import math

START_RATING = 1500.0


def new_model(k=20.0, hfa=60.0, mov="none", xg_weight=0.0, xg_c=1.2, carry=0.75, autocorr=False, name=None):
    return {"kind": "elo", "name": name or f"elo_{mov}", "k": k, "hfa": hfa, "mov": mov,
            "xg_weight": xg_weight, "xg_c": xg_c, "carry": carry, "autocorr": autocorr,
            "ratings": {}}   # team id -> rating (a team not in here has START_RATING)


def get_rating(model, team):
    return model["ratings"].get(team, START_RATING)


def margin_multiplier(kind, goal_margin):
    """How much to scale the rating change for a win by `goal_margin` goals."""
    if kind == "none" or goal_margin <= 1:
        return 1.0
    if kind == "sqrt":
        return math.sqrt(goal_margin)
    if kind == "log":
        return math.log(goal_margin + 1) + 1.0
    if kind == "wfe":   # World Football Elo (eloratings.net)
        return 1.5 if goal_margin == 2 else (11 + goal_margin) / 8
    raise ValueError(f"unknown margin-of-victory kind: {kind}")


def start_season(model, year):
    """Pull every rating part of the way back towards 1500."""
    ratings = model["ratings"]
    for team in ratings:
        ratings[team] = START_RATING + model["carry"] * (ratings[team] - START_RATING)


def gap(model, match):
    return get_rating(model, match.home) - get_rating(model, match.away) + model["hfa"]


def update(model, match):
    rating_gap = gap(model, match)
    expected = 1.0 / (1.0 + 10 ** (-rating_gap / 400.0))
    goal_diff = match.hg - match.ag

    has_xg = not (math.isnan(match.hxg) or math.isnan(match.axg))
    if model["xg_weight"] > 0 and has_xg:
        # Score the match by chance quality as well as goals.
        margin = model["xg_weight"] * (match.hxg - match.axg) + (1 - model["xg_weight"]) * goal_diff
        actual = 1.0 / (1.0 + math.exp(-model["xg_c"] * margin))
        multiplier = 1.0
    else:
        actual = 1.0 if goal_diff > 0 else 0.5 if goal_diff == 0 else 0.0
        multiplier = margin_multiplier(model["mov"], abs(goal_diff))
        if model["autocorr"] and goal_diff != 0:
            winner_gap = rating_gap if goal_diff > 0 else -rating_gap   # how much the winner was favoured
            multiplier *= 2.2 / (0.001 * winner_gap + 2.2)

    change = model["k"] * multiplier * (actual - expected)
    model["ratings"][match.home] = get_rating(model, match.home) + change
    model["ratings"][match.away] = get_rating(model, match.away) - change
