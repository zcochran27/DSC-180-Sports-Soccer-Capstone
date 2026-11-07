"""Load matches from MotherDuck for the rating models."""
import os

import duckdb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()   # reads MOTHERDUCK_TOKEN and MOTHERDUCK_DB from the .env file

MATCH_SQL = """
    SELECT m.match_id, m.date_utc AS date, st.year AS year, st.term AS term,
           th.gender AS gender, m.home_team_id AS home, m.away_team_id AS away,
           m.home_score AS hg, m.away_score AS ag,
           hx.total_xg_shot AS hxg, ax.total_xg_shot AS axg
    FROM wyscout.matches m
    JOIN wyscout.season_terms st ON m.season_id = st.season_id AND m.comp_id = st.comp_id
    JOIN wyscout.teams th ON m.home_team_id = th.team_id
    LEFT JOIN wyscout.team_match_stats hx ON hx.match_id = m.match_id AND hx.team_id = m.home_team_id
    LEFT JOIN wyscout.team_match_stats ax ON ax.match_id = m.match_id AND ax.team_id = m.away_team_id
    WHERE st.kind = 'regular' AND m.status IN ('Played', 'Awarded')
      AND m.home_score IS NOT NULL AND m.away_score IS NOT NULL
      AND th.gender = '{gender}'
    ORDER BY m.date_utc, m.match_id
"""


def connect():
    """Open a MotherDuck connection."""
    database = os.getenv("MOTHERDUCK_DB", "wyscout_dev")
    token = os.getenv("MOTHERDUCK_TOKEN")
    return duckdb.connect(f"md:{database}?motherduck_token={token}")


def load_matches(gender, con=None):
    """One row per played regular-season match, oldest first.

    gender is "male" or "female". Columns: home / away are team ids, hg / ag are goals,
    hxg / axg are expected goals (NaN when the match has no team stats).
    """
    con = con or connect()
    matches = con.execute(MATCH_SQL.format(gender=gender)).df()
    matches["date"] = pd.to_datetime(matches["date"])
    return matches.drop_duplicates("match_id").reset_index(drop=True)
