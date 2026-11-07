"""Compare our team rankings with published men's D1 rankings (NCAA RPI and Shortlistr).

    table, summary = compare("elo_xg", "2026-11-03", 2026, ref="rpi")

The RPI is from https://www.ncaa.com/rankings/soccer-men/d1/ncaa-mens-soccer-rpi (games through
Nov. 3, 2026). Both files are saved in ratings/reference/. Our ratings only use matches up to
`as_of`. School names are matched to Wyscout team names; schools that cannot be matched are
listed in the table (team_id is empty).
"""
import re
from pathlib import Path

import pandas as pd
from scipy.stats import kendalltau, spearmanr

from ratings.data import connect, load_matches
from ratings.matchup import fit_final, rankings

REFERENCE_DIR = Path(__file__).parent / "reference"
REFERENCES = {  # name -> csv file with columns: rank, school
    "rpi": REFERENCE_DIR / "ncaa_rpi_men_2026-11-03.csv",
    "shortlistr": REFERENCE_DIR / "shortlistr_men_2026fall.csv",
}

# Short forms that appear in school names. Expanded on both sides before comparing.
ABBREVIATIONS = {"st": "state", "fla": "florida", "ga": "georgia", "col": "college", "ky": "kentucky",
                 "ill": "illinois", "ind": "indiana", "ark": "arkansas", "conn": "connecticut",
                 "mich": "michigan", "mich.": "michigan", "n": "north", "s": "south"}

# School name in the ranking -> start of the team name in Wyscout.
# Only needed where the two names are too different to match automatically.
ALIASES = {
    "boston u": "boston university",
    "boston university": "boston",
    "cal state bakersfield": "csu bakersfield",
    "cal state fullerton": "cal st fullerton",
    "cal state northridge": "csun",
    "central conn st": "ccsu",
    "central conn state": "central connecticut",
    "central connecticut": "ccsu",
    "col of charleston": "charleston",
    "east tennessee state": "etsu",
    "etsu": "east tennessee",
    "fdu": "fairleigh dickinson",
    "fgcu": "florida gulf coast",
    "fla atlantic": "fau",
    "florida atlantic": "fau",
    "florida gulf coast": "fgcu",
    "ga southern": "georgia southern",
    "iu indy": "iu indianapolis",
    "iupui": "iu indianapolis",
    "james madison": "jmu",
    "kansas city": "umkc",
    "lmu ca": "loyola marymount",
    "loyola maryland": "loyola greyhounds",
    "loyola marymount": "lmu",
    "massachusetts": "umass",
    "niu": "northern illinois",
    "northern kentucky": "nku",
    "northern ky": "nku",
    "purdue fort wayne": "ipfw",
    "queens nc": "queens",
    "saint marys": "st marys",
    "saint marys ca": "saint marys",
    "seattle u": "seattle",
    "south fla": "usf",
    "south florida": "usf",
    "southern illinois edwardsville": "siue",
    "southern ind": "southern indiana",
    "st johns ny": "st johns",
    "ualbany": "albany",
    "uiw": "incarnate word",
    "unc greensboro": "uncg",
    "unc wilmington": "uncw",
    "west florida argos": "west florida",
}


def normalize(name, expand=True):
    """Lowercase a school name, drop punctuation, and (optionally) spell out abbreviations."""
    name = name.lower().replace("&", "and")
    name = re.sub(r"[().'`]", "", name)
    name = name.replace("-", " ")
    words = name.split()
    if expand:
        words = [ABBREVIATIONS.get(word, word) for word in words]
    return " ".join(words)


def match_teams(schools, teams):
    """Match each school name to a team id. Returns {school: team_id}; unmatched schools are left out.

    A Wyscout team is a candidate when its name equals the school name or starts with it
    (e.g. "duke" matches "duke blue devils"). If several teams are candidates we pick the one
    with the fewest words, then the one that played the most matches this season.
    """
    teams = teams.assign(clean_name=teams["name"].map(normalize))
    matched = {}
    for school in schools:
        key = normalize(ALIASES.get(normalize(school, expand=False), school))
        is_candidate = teams["clean_name"].map(lambda n: n == key or n.startswith(key + " "))
        candidates = teams[is_candidate]
        if candidates.empty:
            continue
        candidates = candidates.assign(n_words=candidates["clean_name"].str.split().str.len())
        best = candidates.sort_values(["n_words", "played"], ascending=[True, False]).iloc[0]
        matched[school] = int(best["team_id"])
    return matched


def compare(model_name, as_of, year, matches=None, con=None, ref="rpi"):
    """Rank teams with `model_name` using matches up to `as_of` and compare with the reference.

    Returns (table of reference rank vs our rank, dict of summary numbers).
    """
    matches = load_matches("male") if matches is None else matches
    matches = matches[matches["date"] <= pd.Timestamp(as_of) + pd.Timedelta(days=1)].reset_index(drop=True)
    con = con or connect()

    # Teams that played this season, with how many matches they played (used to break name ties).
    this_year = matches[matches["year"] == year]
    played = pd.concat([this_year["home"], this_year["away"]]).value_counts()
    teams = con.execute("SELECT team_id, name FROM wyscout.teams WHERE gender = 'male'").df()
    teams = teams[teams["team_id"].isin(played.index)]
    teams = teams.assign(played=teams["team_id"].map(played))

    reference = pd.read_csv(REFERENCES[ref])
    reference["team_id"] = reference["school"].map(match_teams(reference["school"].tolist(), teams))

    model, _, _ = fit_final("male", model_name, matches)
    ours = rankings(model, matches, {}, year)
    ours["model_rank"] = range(1, len(ours) + 1)

    table = reference.merge(ours[["team_id", "strength", "model_rank"]], on="team_id", how="left")
    found = table.dropna(subset=["model_rank"])
    top10_ref = set(found.nsmallest(10, "rank")["team_id"])
    top10_ours = set(found.nsmallest(10, "model_rank")["team_id"])
    top25_ref = set(found.nsmallest(25, "rank")["team_id"])
    top25_ours = set(found.nsmallest(25, "model_rank")["team_id"])
    summary = {
        "ref": ref,
        "model": model_name,
        "ref_teams": len(reference),
        "matched": len(found),
        "spearman": spearmanr(found["rank"], found["model_rank"])[0],
        "kendall": kendalltau(found["rank"], found["model_rank"])[0],
        "top10_overlap": len(top10_ref & top10_ours),
        "top25_overlap": len(top25_ref & top25_ours),
        "mean_abs_rank_diff": (found["rank"] - found["model_rank"]).abs().mean(),
    }
    return table, summary

