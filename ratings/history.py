"""Multi-year validation of model rankings against published ones (men's D1, 2021-2026).

    python -m ratings.history            # writes ratings/history_results.csv

References (ratings/reference/): Shortlistr Team Strength (2023-26), United Soccer Coaches
polls (Week 11 = last regular-season poll, and the post-tournament Final; 2026 = Week 10), and
the 2026 NCAA RPI. Each model is walked through all matches once; its strengths are snapshotted
at each reference's as-of date, so the model only sees what existed at that time.
"""
from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import kendalltau, spearmanr

from ratings.compare_rpi import match_teams
from ratings.data import connect, load_matches
from ratings.matchup import strength
from ratings.specs import SPECS, build
from ratings.tune import load_best

REF = Path(__file__).parent / "reference"
OUT = Path(__file__).with_name("history_results.csv")
WK11 = {2021: "2021-11-09", 2022: "2022-11-08", 2023: "2023-11-06", 2024: "2024-11-05", 2025: "2025-11-04"}


def build_refs() -> list[dict]:
    """[{ref, year, as_of, df(rank, school)}]"""
    refs = []
    for y in (2023, 2024, 2025):
        refs.append(dict(ref="shortlistr", year=y, as_of=f"{y}-12-31",
                         df=pd.read_csv(REF / f"shortlistr_men_{y}fall.csv")))
    refs.append(dict(ref="shortlistr", year=2026, as_of="2026-11-03",
                     df=pd.read_csv(REF / "shortlistr_men_2026fall.csv")))
    refs.append(dict(ref="rpi", year=2026, as_of="2026-11-03",
                     df=pd.read_csv(REF / "ncaa_rpi_men_2026-11-03.csv")))
    hist = pd.read_csv(REF / "coaches_poll_history_men.csv")
    for (y, poll), g in hist.groupby(["year", "poll"]):
        refs.append(dict(ref=f"usc_{poll}", year=int(y), as_of=WK11[y] if poll == "wk11" else f"{y}-12-31",
                         df=g[["rank", "school"]]))
    p26 = pd.read_csv(REF / "coaches_poll_2026.csv").query("gender == 'male'")
    refs.append(dict(ref="usc_wk11", year=2026, as_of="2026-10-25",
                     df=p26.rename(columns={"team": "school"})[["rank", "school"]]))
    return refs


def snapshots(model_name: str, matches: pd.DataFrame, snaps: list[tuple]) -> dict:
    """snaps = [(key, year, as_of)] -> {key: DataFrame(team_id, strength)} from one pass."""
    model = build(model_name, load_best("male").get(model_name, {}).get("params"))
    pend = sorted(((pd.Timestamp(a) + pd.Timedelta(days=1), k, y) for k, y, a in snaps), key=lambda t: t[0])
    teams_by_year = {y: g for y, g in matches.groupby("year")}
    out, last_year = {}, None

    def take(dt, key, yr):
        g = teams_by_year[yr]
        g = g[g["date"] < dt]
        ids = sorted(set(g["home"]) | set(g["away"]))
        out[key] = pd.DataFrame({"team_id": ids, "strength": [strength(model, t, dt) for t in ids]})

    for m in matches.itertuples(index=False):
        while pend and m.date >= pend[0][0]:
            dt, key, yr = pend.pop(0)
            take(dt, key, yr)
        if m.year != last_year:
            model.start_season(m.year)
            last_year = m.year
        model.gap(m)
        model.update(m)
    for dt, key, yr in pend:
        take(dt, key, yr)
    return out


def score_ref(strengths: pd.DataFrame, ref_df: pd.DataFrame, ids: dict) -> dict:
    rk = strengths.sort_values("strength", ascending=False).reset_index(drop=True)
    rk["model_rank"] = rk.index + 1
    d = ref_df.assign(team_id=ref_df["school"].map(ids)).merge(rk[["team_id", "model_rank"]], on="team_id", how="left")
    ok = d.dropna(subset=["model_rank"])
    n_ref = len(d)
    return {
        "ref_teams": n_ref, "matched": len(ok), "unmatched": ";".join(d.loc[d.team_id.isna(), "school"]),
        "spearman": spearmanr(ok["rank"], ok["model_rank"])[0],
        "kendall": kendalltau(ok["rank"], ok["model_rank"])[0],
        "ref_top10_in_model_top10": int((ok[ok["rank"] <= 10]["model_rank"] <= 10).sum()),
        "ref_top25_in_model_top25": int((ok[ok["rank"] <= 25]["model_rank"] <= 25).sum()),
        "ref_top25_in_model_top50": int((ok[ok["rank"] <= 25]["model_rank"] <= 50).sum()),
        "median_model_rank_of_ref_top25": ok[ok["rank"] <= 25]["model_rank"].median(),
    }


def run(models=None, min_year: int = 2021) -> pd.DataFrame:
    matches = load_matches("male")
    con = connect()
    names = con.execute("SELECT team_id, name FROM wyscout.teams WHERE gender = 'male'").df()
    refs = [r for r in build_refs() if r["year"] >= min_year]
    # per-year team-name matching (only teams that played that year)
    ids_by_year = {}
    for y in sorted({r["year"] for r in refs}):
        yr = matches[matches["year"] == y]
        played = pd.concat([yr["home"], yr["away"]]).value_counts()
        t = names[names["team_id"].isin(played.index)].assign(played=lambda d: d["team_id"].map(played))
        schools = sorted({s for r in refs if r["year"] == y for s in r["df"]["school"]})
        ids_by_year[y] = match_teams(schools, t)
    snaps = [((r["ref"], r["year"]), r["year"], r["as_of"]) for r in refs]
    rows, all_snaps = [], {}
    for name in models or list(SPECS):
        print("rating", name, flush=True)
        all_snaps[name] = snapshots(name, matches, snaps)

    def add_rows(name, snap):
        for r in refs:
            rows.append({"model": name, "ref": r["ref"], "year": r["year"],
                         **score_ref(snap[(r["ref"], r["year"])], r["df"], ids_by_year[r["year"]])})

    for name, snap in all_snaps.items():
        add_rows(name, snap)

    if len(all_snaps) > 1:  # ensembles: combine per-snapshot z-scored strengths
        from ratings.evaluate import default_models, evaluate
        _, preds, _, _ = evaluate(matches, default_models("male"), return_preds=True)
        w = {k: v for k, v in preds["ensemble_ols"].attrs["coef"].items() if k in all_snaps}
        print("ensemble OLS weights:", {k: round(v, 3) for k, v in w.items()})
        ens = {"ensemble_ols": {}, "ensemble_mean": {}}
        for key in next(iter(all_snaps.values())):
            z = pd.DataFrame({n: (s[key].set_index("team_id")["strength"]) for n, s in all_snaps.items()})
            z = (z - z.mean()) / z.std()
            ens["ensemble_ols"][key] = pd.DataFrame({"team_id": z.index, "strength": (z[list(w)] * pd.Series(w)).sum(axis=1).values})
            ens["ensemble_mean"][key] = pd.DataFrame({"team_id": z.index, "strength": z.mean(axis=1).values})
        for name, snap in ens.items():
            add_rows(name, snap)
    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append")
    ap.add_argument("--min-year", type=int, default=2021)
    args = ap.parse_args()
    res = run(args.model, args.min_year)
    pd.set_option("display.width", 220)
    print(res.pivot_table(index="model", columns=["ref", "year"], values="spearman").round(2).to_string())
    print("\nmean spearman by model/ref:")
    print(res.pivot_table(index="model", columns="ref", values="spearman").round(3).to_string())
    um = res.drop_duplicates(["ref", "year"])[["ref", "year", "matched", "ref_teams", "unmatched"]]
    print(um.to_string(index=False))


if __name__ == "__main__":
    main()
