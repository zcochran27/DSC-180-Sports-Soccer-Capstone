"""Fit a rating model on all matches, then rank teams and compare any two opponents.

    python -m ratings.matchup --gender male --model dixon_coles_xg --home "Duke Blue Devils" --away "UCLA Bruins"
"""
from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import argparse
from collections import namedtuple

import pandas as pd

from ratings.base import OutcomeMapper, walk_forward
from ratings.data import connect, load_matches
from ratings.specs import build
from ratings.tune import load_best

Row = namedtuple("Row", "date home away")
_NONE = -1  # sentinel "average / unseen team"


def fit_final(gender: str, model_name: str = "dixon_coles_xg", matches: pd.DataFrame | None = None):
    """Walk the model through every match; return (model, mapper, matches).
    The mapper is fit on all out-of-sample gaps after the first (burn-in) season."""
    matches = load_matches(gender) if matches is None else matches
    params = load_best(gender).get(model_name, {}).get("params")
    model = build(model_name, params)
    preds = walk_forward(model, matches)
    fit = preds[preds["year"] > preds["year"].min()]
    mapper = OutcomeMapper().fit(fit["gap"], fit["hg"], fit["ag"])
    return model, mapper, matches


def strength(model, team, date) -> float:
    """Team strength vs an average unseen opponent, home advantage removed (native units)."""
    return model.gap(Row(date, team, _NONE)) - model.gap(Row(date, _NONE, _NONE))


def rankings(model, matches: pd.DataFrame, names: dict, year: int | None = None) -> pd.DataFrame:
    year = year or int(matches["year"].max())
    cur = matches[matches["year"] == year]
    date = matches["date"].max()
    teams = sorted(set(cur["home"]) | set(cur["away"]))
    df = pd.DataFrame({"team_id": teams, "strength": [strength(model, t, date) for t in teams]})
    df["team"] = df["team_id"].map(names)
    return df.sort_values("strength", ascending=False).reset_index(drop=True)


def predict_matchup(model, mapper, home: int, away: int, date=None, neutral: bool = False) -> dict:
    """Rating gap and W/D/L probabilities / expected goal diff for home vs away.
    neutral=True removes home advantage (gap averaged over both venues)."""
    date = date or pd.Timestamp.now()
    g = model.gap(Row(date, home, away))
    if neutral:
        g = (g - model.gap(Row(date, away, home))) / 2
    p = mapper.probs([g])[0]
    return {"gap": float(g), "p_home": p[0], "p_draw": p[1], "p_away": p[2],
            "exp_goal_diff": float(mapper.exp_gd([g])[0])}


def team_names(con=None) -> dict:
    con = con or connect()
    return dict(con.execute("SELECT team_id, name FROM wyscout.teams").fetchall())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gender", default="male", choices=["male", "female"])
    ap.add_argument("--model", default="dixon_coles_xg")
    ap.add_argument("--home")
    ap.add_argument("--away")
    ap.add_argument("--neutral", action="store_true")
    args = ap.parse_args()
    model, mapper, matches = fit_final(args.gender, args.model)
    names = team_names()
    print(rankings(model, matches, names).head(15).round(3).to_string())
    if args.home and args.away:
        ids = {v: k for k, v in names.items()}
        print(predict_matchup(model, mapper, ids[args.home], ids[args.away], neutral=args.neutral))


if __name__ == "__main__":
    main()
