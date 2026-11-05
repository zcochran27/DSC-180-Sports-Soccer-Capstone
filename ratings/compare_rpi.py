"""Validate model rankings against the NCAA men's D1 RPI (full ~213-team field).

    python -m ratings.compare_rpi --model elo_xg --model dixon_coles_xg --as-of 2026-11-03

Source: https://www.ncaa.com/rankings/soccer-men/d1/ncaa-mens-soccer-rpi (rankings through
games of Nov. 3, 2026), saved in ratings/reference/. Ratings use only matches up to --as-of.
NCAA short names are matched to Wyscout team names; unmatched schools are reported.
"""
from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import argparse
import re
from pathlib import Path

import pandas as pd
from scipy.stats import kendalltau, spearmanr

from ratings.data import connect, load_matches
from ratings.matchup import fit_final, rankings

REF_DIR = Path(__file__).parent / "reference"
REFS = {  # name -> csv with columns rank, school
    "rpi": REF_DIR / "ncaa_rpi_men_2026-11-03.csv",
    "shortlistr": REF_DIR / "shortlistr_men_2026fall.csv",
}

ABBREV = {"st": "state", "fla": "florida", "ga": "georgia", "col": "college", "ky": "kentucky",
          "ill": "illinois", "ind": "indiana", "ark": "arkansas", "conn": "connecticut",
          "mich": "michigan", "mich.": "michigan", "u": "u", "n": "north", "s": "south"}
ALIASES = {  # NCAA short name -> Wyscout name prefix
    "kansas city": "umkc", "ualbany": "albany", "uncw": "uncw", "unc greensboro": "uncg",
    "ucf": "ucf", "fiu": "fiu", "uconn": "uconn", "csun": "csun", "south fla": "south florida",
    "lmu ca": "loyola marymount", "uic": "uic", "iu indy": "iu indianapolis",
    "fdu": "fairleigh dickinson", "njit": "njit", "vcu": "vcu", "etsu": "east tennessee",
    "uab": "uab", "siue": "siu edwardsville", "niu": "northern illinois", "uiw": "incarnate word",
    "fgcu": "florida gulf coast", "utrgv": "utrgv", "liu": "liu", "umbc": "umbc",
    "st johns ny": "st johns", "queens nc": "queens", "saint marys ca": "saint marys",
    "ga southern": "georgia southern", "central conn state": "central connecticut",
    "northern ky": "northern kentucky", "southern ind": "southern indiana",
    "col of charleston": "charleston", "fla atlantic": "florida atlantic",
    "boston u": "boston university", "uc san diego": "uc san diego", "seattle u": "seattle",
    "massachusetts": "umass", "penn": "penn", "oklahoma": "oklahoma",
    # long-form names used by Shortlistr
    "unc wilmington": "uncw", "east tennessee state": "etsu", "iupui": "iu indianapolis",
    "fairleigh dickinson": "fairleigh dickinson", "james madison": "jmu", "cal state northridge": "csun",
    "cal state bakersfield": "csu bakersfield", "cal state fullerton": "cal st fullerton",
    "florida gulf coast": "fgcu", "loyola marymount": "lmu", "umass": "umass",
    "saint marys": "st marys", "west florida argos": "west florida", "boston university": "boston",
    "northern illinois": "northern illinois", "niu": "northern illinois", "florida atlantic": "fau",
    "fla atlantic": "fau", "south florida": "usf", "south fla": "usf", "central connecticut": "ccsu",
    "central conn st": "ccsu", "northern kentucky": "nku", "northern ky": "nku",
    "purdue fort wayne": "ipfw", "loyola maryland": "loyola greyhounds",
    "southern illinois edwardsville": "siue", "siue": "siue", "loyola marymount": "lmu",
}


def norm(s: str, expand: bool = True) -> str:
    s = s.lower().replace("&", "and")
    s = re.sub(r"[().'`]", "", s)
    s = re.sub(r"-", " ", s)
    toks = [(ABBREV.get(t, t) if expand else t) for t in s.split()]
    return " ".join(toks)


def match_teams(schools: list[str], teams: pd.DataFrame) -> dict:
    """school -> team_id. Candidates share a token prefix with the school; ties go to the
    name with fewest tokens, then the most matches played this season."""
    teams = teams.assign(n=teams["name"].map(norm))  # abbreviations expanded on both sides
    out = {}
    for sch in schools:
        key = norm(ALIASES.get(norm(sch, expand=False), sch))
        cands = teams[teams["n"].map(lambda n: n == key or n.startswith(key + " "))]
        if cands.empty:
            continue
        cands = cands.assign(k=cands["n"].str.split().str.len()).sort_values(["k", "played"], ascending=[True, False])
        out[sch] = int(cands.iloc[0]["team_id"])
    return out


def compare(model_name: str, as_of: str, year: int, matches: pd.DataFrame | None = None, con=None,
            ref: str = "rpi"):
    matches = load_matches("male") if matches is None else matches
    matches = matches[matches["date"] <= pd.Timestamp(as_of) + pd.Timedelta(days=1)].reset_index(drop=True)
    con = con or connect()
    played = pd.concat([matches.loc[matches.year == year, "home"], matches.loc[matches.year == year, "away"]]).value_counts()
    teams = con.execute("SELECT team_id, name FROM wyscout.teams WHERE gender = 'male'").df()
    teams = teams[teams["team_id"].isin(played.index)].assign(played=lambda d: d["team_id"].map(played))
    rpi = pd.read_csv(REFS[ref])
    ids = match_teams(rpi["school"].tolist(), teams)
    rpi["team_id"] = rpi["school"].map(ids)
    model, _, _ = fit_final("male", model_name, matches)
    rk = rankings(model, matches, {}, year)
    rk["model_rank"] = range(1, len(rk) + 1)
    out = rpi.merge(rk[["team_id", "strength", "model_rank"]], on="team_id", how="left")
    ok = out.dropna(subset=["model_rank"])
    stats = {
        "ref": ref, "model": model_name, "ref_teams": len(rpi), "matched": len(ok),
        "spearman": spearmanr(ok["rank"], ok["model_rank"])[0],
        "kendall": kendalltau(ok["rank"], ok["model_rank"])[0],
        "top10_overlap": len(set(ok.nsmallest(10, "rank")["team_id"]) & set(ok.nsmallest(10, "model_rank")["team_id"])),
        "top25_overlap": len(set(ok.nsmallest(25, "rank")["team_id"]) & set(ok.nsmallest(25, "model_rank")["team_id"])),
        "mean_abs_rank_diff": (ok["rank"] - ok["model_rank"]).abs().mean(),
    }
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append")
    ap.add_argument("--as-of", default="2026-11-03")
    ap.add_argument("--year", type=int, default=2026, help="season year of the reference snapshot")
    ap.add_argument("--ref", action="append", choices=list(REFS), help="default: all")
    args = ap.parse_args()
    pd.set_option("display.width", 200)
    matches = load_matches("male")
    print("data through", matches["date"].max())
    rows = []
    for ref in args.ref or list(REFS):
        for name in args.model or ["elo_xg", "dixon_coles_xg"]:
            out, stats = compare(name, args.as_of, args.year, matches, ref=ref)
            rows.append(stats)
        print(f"[{ref}] unmatched:", out.loc[out.team_id.isna(), "school"].tolist())
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
