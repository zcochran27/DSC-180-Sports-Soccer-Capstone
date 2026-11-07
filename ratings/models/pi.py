"""Pi-ratings (Constantinou & Fenton, 2013).

Each team has two ratings measured in goals: how well it does at home and how well it does away.
The expected goal difference of a match is the home team's home rating minus the away team's away
rating. After the match each rating moves toward the result.

Settings of a model (new_model arguments):
    lam     learning rate
    gamma   share of a team's update that also goes to its other (home/away) rating
    c       how strongly big errors are damped
    carry   share of each rating kept when a new season starts
    scale   converts goals to Elo-like units so the gap is comparable to the other models
"""
import math
from collections import defaultdict


def new_model(lam=0.06, gamma=0.5, c=3.0, carry=0.8, scale=200.0, name="pi"):
    return {"kind": "pi", "name": name, "lam": lam, "gamma": gamma, "c": c, "carry": carry, "scale": scale,
            "home_rating": defaultdict(float),   # expected goal diff vs an average team, at home
            "away_rating": defaultdict(float)}   # same, but away


def damp(error, c):
    """Shrink big errors so one blowout does not move ratings too much."""
    return math.copysign(c * math.log10(1 + abs(error)), error)


def start_season(model, year):
    for ratings in (model["home_rating"], model["away_rating"]):
        for team in ratings:
            ratings[team] *= model["carry"]


def expected_goal_diff(model, match):
    return model["home_rating"][match.home] - model["away_rating"][match.away]


def gap(model, match):
    return expected_goal_diff(model, match) * model["scale"]


def update(model, match):
    error = (match.hg - match.ag) - expected_goal_diff(model, match)
    step = damp(error, model["c"]) * model["lam"]

    # Home team: its home rating moves by `step`, its away rating by a fraction of that.
    model["home_rating"][match.home] += step
    model["away_rating"][match.home] += step * model["gamma"]
    # Away team: moves the opposite way.
    model["away_rating"][match.away] -= step
    model["home_rating"][match.away] -= step * model["gamma"]
