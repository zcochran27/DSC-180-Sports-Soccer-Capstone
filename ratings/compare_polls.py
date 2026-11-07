"""Compare our team rankings with the United Soccer Coaches Top 25 (saved in ratings/reference/).

From a notebook:   table, summary = compare("male", "elo_xg")

Our ratings only use matches through the poll's record date (2026-10-25), so the model knows
the same things the coaches did. The poll comes from SoccerWire's copy of the United Soccer
Coaches polls (men's Week 10, women's Week 11).
"""
from pathlib import Path

import pandas as pd
from scipy.stats import kendalltau, spearmanr

from ratings.data import load_matches
from ratings.matchup import fit_final, rankings

POLL_PATH = Path(__file__).parent / "reference" / "coaches_poll_2026.csv"
AS_OF = pd.Timestamp("2026-10-25 23:59")


def compare(gender, model_name, matches=None):
    """Returns (table of poll rank vs our rank, dict of summary numbers)."""
    poll = pd.read_csv(POLL_PATH).query("gender == @gender")
    matches = load_matches(gender) if matches is None else matches
    matches = matches[matches["date"] <= AS_OF].reset_index(drop=True)

    model, _, _ = fit_final(gender, model_name, matches)
    year = int(matches["year"].max())
    ours = rankings(model, matches, {}, year)  # every team that played this season, strongest first
    ours["model_rank"] = range(1, len(ours) + 1)

    table = poll.merge(ours[["team_id", "strength", "model_rank"]], on="team_id", how="left")
    table["rank_diff"] = table["model_rank"] - table["rank"]

    found = table.dropna(subset=["model_rank"])
    summary = {
        "model": model_name,
        "gender": gender,
        "teams_matched": len(found),
        "n_ranked_teams": len(ours),
        "spearman": spearmanr(found["rank"], found["model_rank"])[0],
        "kendall": kendalltau(found["rank"], found["model_rank"])[0],
        "poll_top10_in_model_top10": int((found[found["rank"] <= 10]["model_rank"] <= 10).sum()),
        "poll_top25_in_model_top25": int((found["model_rank"] <= 25).sum()),
        "poll_top25_in_model_top50": int((found["model_rank"] <= 50).sum()),
        "median_model_rank_of_poll": found["model_rank"].median(),
    }
    return table, summary

