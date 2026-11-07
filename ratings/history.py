"""Compare our men's D1 team rankings with published rankings over several years (2021-2026).

    results = run()      # from a notebook; writes ratings/history_results.csv (takes a while)

Published rankings (saved in ratings/reference/): Shortlistr Team Strength (2023-26), United
Soccer Coaches polls (Week 11, the last regular-season poll, and the Final poll after the
tournament; 2026 only has Week 10) and the 2026 NCAA RPI.

Each model goes through all matches once. We save its team strengths on the date of each
published ranking, so the model only uses games that had been played by then.
"""
from pathlib import Path

import pandas as pd
from scipy.stats import kendalltau, spearmanr

from ratings.compare_rpi import match_teams
from ratings.data import connect, load_matches
from ratings.evaluate import default_models, evaluate
from ratings import base
from ratings.matchup import strength
from ratings.specs import SPECS, build
from ratings.tune import load_best

REFERENCE_DIR = Path(__file__).parent / "reference"
OUTPUT_PATH = Path(__file__).with_name("history_results.csv")

# Date of the Week 11 coaches poll each year.
WEEK_11_DATES = {2021: "2021-11-09", 2022: "2022-11-08", 2023: "2023-11-06",
                 2024: "2024-11-05", 2025: "2025-11-04"}


def load_references():
    """Every published ranking as {"ref", "year", "as_of", "df"}; df has columns rank and school."""
    refs = []

    for year in (2023, 2024, 2025):
        refs.append({"ref": "shortlistr", "year": year, "as_of": f"{year}-12-31",
                     "df": pd.read_csv(REFERENCE_DIR / f"shortlistr_men_{year}fall.csv")})
    refs.append({"ref": "shortlistr", "year": 2026, "as_of": "2026-11-03",
                 "df": pd.read_csv(REFERENCE_DIR / "shortlistr_men_2026fall.csv")})
    refs.append({"ref": "rpi", "year": 2026, "as_of": "2026-11-03",
                 "df": pd.read_csv(REFERENCE_DIR / "ncaa_rpi_men_2026-11-03.csv")})

    coaches = pd.read_csv(REFERENCE_DIR / "coaches_poll_history_men.csv")
    for (year, poll), rows in coaches.groupby(["year", "poll"]):
        as_of = WEEK_11_DATES[year] if poll == "wk11" else f"{year}-12-31"
        refs.append({"ref": f"usc_{poll}", "year": int(year), "as_of": as_of,
                     "df": rows[["rank", "school"]]})

    poll_2026 = pd.read_csv(REFERENCE_DIR / "coaches_poll_2026.csv").query("gender == 'male'")
    refs.append({"ref": "usc_wk11", "year": 2026, "as_of": "2026-10-25",
                 "df": poll_2026.rename(columns={"team": "school"})[["rank", "school"]]})
    return refs


def take_snapshot(model, matches_by_year, results, date, key, year):
    """Record every team's strength on `date`, for the teams that played in `year` before that date."""
    year_matches = matches_by_year[year]
    played_before = year_matches[year_matches["date"] < date]
    team_ids = sorted(set(played_before["home"]) | set(played_before["away"]))
    results[key] = pd.DataFrame({"team_id": team_ids,
                                 "strength": [strength(model, team, date) for team in team_ids]})


def strengths_on_each_date(model_name, matches, snapshots):
    """Go through all matches once and record team strengths on each requested date.

    snapshots is a list of (key, year, as_of_date). Returns {key: DataFrame(team_id, strength)},
    where each frame has the teams that played in that year before the date.
    """
    model = build(model_name, load_best("male").get(model_name, {}).get("params"))
    matches_by_year = {year: rows for year, rows in matches.groupby("year")}

    # Snapshots still to take, earliest first. We take one when we reach the first match after its date.
    waiting = sorted(((pd.Timestamp(as_of) + pd.Timedelta(days=1), key, year)
                      for key, year, as_of in snapshots), key=lambda s: s[0])
    results = {}

    last_year = None
    for match in matches.itertuples(index=False):
        while waiting and match.date >= waiting[0][0]:
            date, key, year = waiting.pop(0)
            take_snapshot(model, matches_by_year, results, date, key, year)
        if match.year != last_year:
            base.start_season(model, match.year)
            last_year = match.year
        base.gap(model, match)   # some models need to see the gap before the update
        base.update(model, match)

    for date, key, year in waiting:   # dates after the last match
        take_snapshot(model, matches_by_year, results, date, key, year)
    return results


def compare_to_reference(strengths, reference, school_ids):
    """How closely our ranking agrees with one published ranking."""
    ours = strengths.sort_values("strength", ascending=False).reset_index(drop=True)
    ours["model_rank"] = ours.index + 1

    reference = reference.assign(team_id=reference["school"].map(school_ids))
    table = reference.merge(ours[["team_id", "model_rank"]], on="team_id", how="left")
    found = table.dropna(subset=["model_rank"])
    top25 = found[found["rank"] <= 25]

    return {
        "ref_teams": len(table),
        "matched": len(found),
        "unmatched": ";".join(table.loc[table.team_id.isna(), "school"]),
        "spearman": spearmanr(found["rank"], found["model_rank"])[0],
        "kendall": kendalltau(found["rank"], found["model_rank"])[0],
        "ref_top10_in_model_top10": int((found[found["rank"] <= 10]["model_rank"] <= 10).sum()),
        "ref_top25_in_model_top25": int((top25["model_rank"] <= 25).sum()),
        "ref_top25_in_model_top50": int((top25["model_rank"] <= 50).sum()),
        "median_model_rank_of_ref_top25": top25["model_rank"].median(),
    }


def match_schools_by_year(refs, matches, con):
    """{year: {school name: team_id}}, matching only teams that played that year."""
    teams = con.execute("SELECT team_id, name FROM wyscout.teams WHERE gender = 'male'").df()
    school_ids = {}
    for year in sorted({ref["year"] for ref in refs}):
        this_year = matches[matches["year"] == year]
        played = pd.concat([this_year["home"], this_year["away"]]).value_counts()
        teams_this_year = teams[teams["team_id"].isin(played.index)]
        teams_this_year = teams_this_year.assign(played=teams_this_year["team_id"].map(played))
        schools = sorted({school for ref in refs if ref["year"] == year for school in ref["df"]["school"]})
        school_ids[year] = match_teams(schools, teams_this_year)
    return school_ids


def make_ensembles(all_strengths, matches):
    """Combine models into two extra "models": a weighted blend and a plain average.

    Strengths are standardised within each snapshot first. Blend weights come from the
    ensemble fit in ratings.evaluate.
    """
    _, predictions, _, _ = evaluate(matches, default_models("male"), return_preds=True)
    weights = {name: w for name, w in predictions["ensemble_ols"].attrs["coef"].items()
               if name in all_strengths}
    print("ensemble weights:", {name: round(float(w), 3) for name, w in weights.items()})

    ensembles = {"ensemble_ols": {}, "ensemble_mean": {}}
    snapshot_keys = next(iter(all_strengths.values())).keys()
    for key in snapshot_keys:
        table = pd.DataFrame({name: snaps[key].set_index("team_id")["strength"]
                              for name, snaps in all_strengths.items()})
        z = (table - table.mean()) / table.std()
        blend = (z[list(weights)] * pd.Series(weights)).sum(axis=1)
        ensembles["ensemble_ols"][key] = pd.DataFrame({"team_id": z.index, "strength": blend.values})
        ensembles["ensemble_mean"][key] = pd.DataFrame({"team_id": z.index, "strength": z.mean(axis=1).values})
    return ensembles


def run(models=None, min_year=2021):
    matches = load_matches("male")
    refs = [ref for ref in load_references() if ref["year"] >= min_year]
    school_ids = match_schools_by_year(refs, matches, connect())
    snapshots = [((ref["ref"], ref["year"]), ref["year"], ref["as_of"]) for ref in refs]

    all_strengths = {}
    for name in models or list(SPECS):
        print("rating", name, flush=True)
        all_strengths[name] = strengths_on_each_date(name, matches, snapshots)
    if len(all_strengths) > 1:
        all_strengths.update(make_ensembles(all_strengths, matches))

    rows = []
    for model_name, snaps in all_strengths.items():
        for ref in refs:
            result = compare_to_reference(snaps[(ref["ref"], ref["year"])], ref["df"], school_ids[ref["year"]])
            rows.append({"model": model_name, "ref": ref["ref"], "year": ref["year"], **result})

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_PATH, index=False)
    return results

