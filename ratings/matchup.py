"""Fit a rating model on all matches, rank the teams, and compare any two opponents.

From a notebook:
    model, mapper, matches = fit_final("male", "elo_sqrt")
    names = team_names()
    rankings(model, matches, names).head(15)
    predict_matchup(model, mapper, home_team_id, away_team_id, neutral=False)
"""
from collections import namedtuple

import pandas as pd

from ratings import base
from ratings.data import connect, load_matches
from ratings.specs import build
from ratings.tune import load_best

# The smallest "match" a model needs to produce a gap.
Fixture = namedtuple("Fixture", "date home away")

# Stands in for "an average team we have never seen".
AVERAGE_TEAM = -1


def fit_final(gender, model_name="dixon_coles_xg", matches=None):
    """Run the model through every match. Returns (model, mapper, matches).

    The mapper is fit on the model's gaps for every season after the first (the burn-in season).
    """
    matches = load_matches(gender) if matches is None else matches
    params = load_best(gender).get(model_name, {}).get("params")
    model = build(model_name, params)
    predictions = base.walk_forward(model, matches)

    after_burn_in = predictions[predictions["year"] > predictions["year"].min()]
    mapper = base.fit_outcome_mapper(after_burn_in["gap"], after_burn_in["hg"], after_burn_in["ag"])
    return model, mapper, matches


def strength(model, team, date):
    """How strong a team is against an average opponent, with home advantage removed."""
    vs_average = base.gap(model, Fixture(date, team, AVERAGE_TEAM))
    average_vs_average = base.gap(model, Fixture(date, AVERAGE_TEAM, AVERAGE_TEAM))
    return vs_average - average_vs_average


def rankings(model, matches, names, year=None):
    """Teams that played in `year` (default: latest season), strongest first."""
    year = year or int(matches["year"].max())
    this_year = matches[matches["year"] == year]
    date = matches["date"].max()
    team_ids = sorted(set(this_year["home"]) | set(this_year["away"]))

    table = pd.DataFrame({"team_id": team_ids,
                          "strength": [strength(model, team, date) for team in team_ids]})
    table["team"] = table["team_id"].map(names)
    return table.sort_values("strength", ascending=False).reset_index(drop=True)


def predict_matchup(model, mapper, home, away, date=None, neutral=False):
    """Rating gap, win/draw/loss probabilities and expected goal difference for home vs away.

    neutral=True removes home advantage by averaging the gap over both venues.
    """
    date = date or pd.Timestamp.now()
    gap = base.gap(model, Fixture(date, home, away))
    if neutral:
        gap = (gap - base.gap(model, Fixture(date, away, home))) / 2

    p_home, p_draw, p_away = base.outcome_probs(mapper, [gap])[0]
    return {"gap": float(gap), "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
            "exp_goal_diff": float(base.expected_goal_diff(mapper, [gap])[0])}


def team_names(con=None):
    """{team_id: team name}"""
    con = con or connect()
    return dict(con.execute("SELECT team_id, name FROM wyscout.teams").fetchall())
