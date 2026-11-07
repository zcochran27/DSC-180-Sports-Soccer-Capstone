"""Write each team's pre-match Elo rating (elo_sqrt model) onto wyscout.matches.

Adds `home_team_elo` and `away_team_elo`: each team's rating just BEFORE the match, so the
result is not leaked into the rating. Men and women are rated separately with the tuned
parameters in best_params.json. A team starts at 1500 in its first match in the data, so the
earliest seasons are a warm-up. Only played regular-season matches get a rating; friendlies,
cancelled matches and matches without a score stay NULL.

Note: wyscout/load.py reloads matches with INSERT OR REPLACE, which sets these two columns back
to NULL. Run the export again after loading matches.

From a notebook:
    elo = both_genders_elo()                 # work out the ratings, writes nothing
    write_to_database(con, elo)              # add the columns if needed and fill them
"""
import pandas as pd

from ratings import base
from ratings.data import load_matches
from ratings.models import elo as elo_model
from ratings.specs import build
from ratings.tune import load_best

MODEL = "elo_sqrt"


def pre_match_elo(gender, model_name=MODEL, matches=None):
    """One row per match: match_id, home_team_elo, away_team_elo (ratings before the match)."""
    matches = load_matches(gender) if matches is None else matches
    model = build(model_name, load_best(gender).get(model_name, {}).get("params"))

    rows = []
    last_year = None
    for match in matches.itertuples(index=False):
        if match.year != last_year:
            base.start_season(model, match.year)
            last_year = match.year
        home_rating = elo_model.get_rating(model, match.home)
        away_rating = elo_model.get_rating(model, match.away)
        rows.append((match.match_id, home_rating, away_rating))
        base.update(model, match)
    return pd.DataFrame(rows, columns=["match_id", "home_team_elo", "away_team_elo"])


def both_genders_elo():
    """Pre-match Elo for every men's and women's match."""
    frames = [pre_match_elo(gender) for gender in ("male", "female")]
    return pd.concat(frames, ignore_index=True)


def write_to_database(con, elo):
    """Fill the Elo columns on wyscout.matches. Returns the number of matches written."""
    for column in ("home_team_elo", "away_team_elo"):
        con.execute(f"ALTER TABLE wyscout.matches ADD COLUMN IF NOT EXISTS {column} DOUBLE")
    con.execute("UPDATE wyscout.matches SET home_team_elo = NULL, away_team_elo = NULL")

    con.register("elo_table", elo)
    try:
        con.execute("""
            UPDATE wyscout.matches AS m
            SET home_team_elo = e.home_team_elo, away_team_elo = e.away_team_elo
            FROM elo_table AS e
            WHERE m.match_id = e.match_id
        """)
    finally:
        con.unregister("elo_table")
    return len(elo)
