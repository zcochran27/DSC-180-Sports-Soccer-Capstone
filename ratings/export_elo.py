"""Write each team's pre-match Elo rating (elo_sqrt model) onto wyscout.matches.

    python -m ratings.export_elo --dry-run      # work out the ratings and print a summary, write nothing
    python -m ratings.export_elo                # add the columns if needed and fill them

Adds `home_team_elo` and `away_team_elo`: each team's rating just BEFORE the match, so the
result is not leaked into the rating. Men and women are rated separately with the tuned
parameters in best_params.json. A team starts at 1500 in its first match in the data, so the
earliest seasons are a warm-up. Only played regular-season matches get a rating; friendlies,
cancelled matches and matches without a score stay NULL.

Note: wyscout/load.py reloads matches with INSERT OR REPLACE, which sets these two columns back
to NULL. Run this script again after loading matches.
"""
from __future__ import annotations

import argparse

import pandas as pd

from ratings.data import connect, load_matches
from ratings.specs import build
from ratings.tune import load_best

MODEL = "elo_sqrt"


def pre_match_elo(gender: str, model_name: str = MODEL, matches: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per match: match_id, home_team_elo, away_team_elo (ratings before the match)."""
    matches = load_matches(gender) if matches is None else matches
    model = build(model_name, load_best(gender).get(model_name, {}).get("params"))

    rows = []
    last_year = None
    for match in matches.itertuples(index=False):
        if match.year != last_year:
            model.start_season(match.year)
            last_year = match.year
        rows.append((match.match_id, model.r[match.home], model.r[match.away]))
        model.update(match)
    return pd.DataFrame(rows, columns=["match_id", "home_team_elo", "away_team_elo"])


def write_to_database(con, elo: pd.DataFrame) -> int:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    con = connect()
    both_genders = []
    for gender in ("male", "female"):
        elo = pre_match_elo(gender)
        print(f"{gender}: {len(elo)} matches rated; home Elo mean {elo.home_team_elo.mean():.0f}, "
              f"away Elo mean {elo.away_team_elo.mean():.0f}")
        both_genders.append(elo)
    elo = pd.concat(both_genders, ignore_index=True)

    if args.dry_run:
        print(elo.describe().round(1))
        return

    n_written = write_to_database(con, elo)
    total, with_elo = con.execute("SELECT count(*), count(home_team_elo) FROM wyscout.matches").fetchone()
    print(f"wrote {n_written} rows; wyscout.matches has {with_elo}/{total} rows with Elo")


if __name__ == "__main__":
    main()
