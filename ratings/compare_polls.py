"""Validate model rankings against the United Soccer Coaches Top 25 (ratings/reference/).

    python -m ratings.compare_polls --model elo_xg --model dixon_coles_xg

Ratings use only matches through the poll's record date (2026-10-25), so the model sees the
same information the coaches did. Poll source: SoccerWire's republication of the United Soccer
Coaches polls (men's Week 10, women's Week 11).
"""
from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import kendalltau, spearmanr

from ratings.data import load_matches
from ratings.matchup import fit_final, rankings

POLL_PATH = Path(__file__).parent / "reference" / "coaches_poll_2026.csv"
AS_OF = pd.Timestamp("2026-10-25 23:59")


def compare(gender: str, model_name: str, matches: pd.DataFrame | None = None):
    poll = pd.read_csv(POLL_PATH).query("gender == @gender")
    matches = load_matches(gender) if matches is None else matches
    matches = matches[matches["date"] <= AS_OF].reset_index(drop=True)
    model, _, _ = fit_final(gender, model_name, matches)
    year = int(matches["year"].max())
    # restrict to teams that played this season, ranked by model strength
    rk = rankings(model, matches, {}, year)
    rk["model_rank"] = range(1, len(rk) + 1)
    out = poll.merge(rk[["team_id", "strength", "model_rank"]], on="team_id", how="left")
    out["rank_diff"] = out["model_rank"] - out["rank"]
    ok = out.dropna(subset=["model_rank"])
    stats = {
        "model": model_name, "gender": gender, "teams_matched": len(ok), "n_ranked_teams": len(rk),
        "spearman": spearmanr(ok["rank"], ok["model_rank"])[0],
        "kendall": kendalltau(ok["rank"], ok["model_rank"])[0],
        "poll_top10_in_model_top10": int((ok[ok["rank"] <= 10]["model_rank"] <= 10).sum()),
        "poll_top25_in_model_top25": int((ok["model_rank"] <= 25).sum()),
        "poll_top25_in_model_top50": int((ok["model_rank"] <= 50).sum()),
        "median_model_rank_of_poll": ok["model_rank"].median(),
    }
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append")
    args = ap.parse_args()
    pd.set_option("display.width", 200)
    rows = []
    for g in ("male", "female"):
        m = load_matches(g)
        for name in args.model or ["elo_xg", "dixon_coles_xg"]:
            out, stats = compare(g, name, m)
            rows.append(stats)
            print(f"\n== {g} / {name} ==")
            print(out[["rank", "team", "model_rank", "rank_diff", "strength"]].round(2).to_string(index=False))
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
