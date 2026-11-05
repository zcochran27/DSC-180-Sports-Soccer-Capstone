"""Fit a rating model on all matches, rank the teams, and compare any two opponents.

    python -m ratings.matchup --gender male --model dixon_coles_xg --home "Duke Blue Devils" --away "UCLA Bruins"
"""
from __future__ import annotations

import argparse
from collections import namedtuple

import pandas as pd

from ratings.base import OutcomeMapper, walk_forward
from ratings.data import connect, load_matches
from ratings.specs import build
from ratings.tune import load_best

# The smallest "match" a model needs to produce a gap.
Fixture = namedtuple("Fixture", "date home away")

# Stands in for "an average team we have never seen".
AVERAGE_TEAM = -1


def fit_final(gender: str, model_name: str = "dixon_coles_xg", matches: pd.DataFrame | None = None):
    """Run the model through every match. Returns (model, mapper, matches).

    The mapper is fit on the model's gaps for every season after the first (the burn-in season).
    """
    matches = load_matches(gender) if matches is None else matches
    params = load_best(gender).get(model_name, {}).get("params")
    model = build(model_name, params)
    predictions = walk_forward(model, matches)

    after_burn_in = predictions[predictions["year"] > predictions["year"].min()]
    mapper = OutcomeMapper().fit(after_burn_in["gap"], after_burn_in["hg"], after_burn_in["ag"])
    return model, mapper, matches


def strength(model, team, date) -> float:
    """How strong a team is against an average opponent, with home advantage removed."""
    vs_average = model.gap(Fixture(date, team, AVERAGE_TEAM))
    average_vs_average = model.gap(Fixture(date, AVERAGE_TEAM, AVERAGE_TEAM))
    return vs_average - average_vs_average


def rankings(model, matches: pd.DataFrame, names: dict, year: int | None = None) -> pd.DataFrame:
    """Teams that played in `year` (default: latest season), strongest first."""
    year = year or int(matches["year"].max())
    this_year = matches[matches["year"] == year]
    date = matches["date"].max()
    team_ids = sorted(set(this_year["home"]) | set(this_year["away"]))

    table = pd.DataFrame({"team_id": team_ids,
                          "strength": [strength(model, team, date) for team in team_ids]})
    table["team"] = table["team_id"].map(names)
    return table.sort_values("strength", ascending=False).reset_index(drop=True)


def predict_matchup(model, mapper, home: int, away: int, date=None, neutral: bool = False) -> dict:
    """Rating gap, win/draw/loss probabilities and expected goal difference for home vs away.

    neutral=True removes home advantage by averaging the gap over both venues.
    """
    date = date or pd.Timestamp.now()
    gap = model.gap(Fixture(date, home, away))
    if neutral:
        gap = (gap - model.gap(Fixture(date, away, home))) / 2

    p_home, p_draw, p_away = mapper.probs([gap])[0]
    return {"gap": float(gap), "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
            "exp_goal_diff": float(mapper.exp_gd([gap])[0])}


def team_names(con=None) -> dict:
    """{team_id: team name}"""
    con = con or connect()
    return dict(con.execute("SELECT team_id, name FROM wyscout.teams").fetchall())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gender", default="male", choices=["male", "female"])
    parser.add_argument("--model", default="dixon_coles_xg")
    parser.add_argument("--home", help="team name, e.g. 'Georgetown Hoyas'")
    parser.add_argument("--away")
    parser.add_argument("--neutral", action="store_true", help="remove home advantage")
    args = parser.parse_args()

    model, mapper, matches = fit_final(args.gender, args.model)
    names = team_names()
    print(rankings(model, matches, names).head(15).round(3).to_string())

    if args.home and args.away:
        ids = {name: team_id for team_id, name in names.items()}
        print(predict_matchup(model, mapper, ids[args.home], ids[args.away], neutral=args.neutral))


if __name__ == "__main__":
    main()
