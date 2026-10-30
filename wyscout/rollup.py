"""Roll season-level stats up across competitions into one row per team/player per year+term.

A Wyscout season belongs to a single competition, so a team that plays a conference schedule,
non-conference matches and a tournament in the same fall has three season-stat rows. This
combines them into one row per (entity, year, term) where term is Fall or Spring.

Run (after wyscout.load --stages season-stats):
    python -m wyscout.rollup                 # team and player rollups + validation
    python -m wyscout.rollup --scope team --include-friendlies

How rows are combined (columns come from the API as total_*, average_*, percent_*):
  total_*    summed across competitions.
  average_*  The API's averages are per "exposure" (roughly matches, scaled by tagged time), not
             per match. Each row's exposure is recovered as sum(total)/sum(average) over a few
             high-volume stats; exposures are summed and average = summed total / summed exposure.
             Averages that are not totals-per-exposure (pass length, possession %, ...) are
             exposure-weighted means.
  percent_*  Recomputed as 100 * sum(numerator) / sum(denominator). The numerator/denominator
             pair comes from column names (successful_X / X, X_won / X, ...) and is kept only if
             it reproduces the API's own percentages on the data; otherwise the percent falls
             back to a mean weighted by its own denominator (matches played if unknown). Chosen methods are written to
             wyscout.stat_formulas so nothing is hidden.

Season name -> (year, term): "2025 Fall", "Spring 2021" use the name. Plain years ("2019") are
classed by the season's dates (ends by June = Spring, starts July or later = Fall, else Full).
"CANCELLED" seasons are dropped; "Friendlies" seasons are dropped unless --include-friendlies.
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from . import load as L

# High-volume stats used to recover each row's exposure (the divisor behind the averages).
EXPOSURE_STATS = ["passes", "successful_passes", "received_pass", "duels", "duels_won",
                  "defensive_duels", "recoveries"]
# average_X whose totals live under another name.
AVG_ALIASES = {"ball_recoveries": "recoveries", "ball_losses": "losses",
               "conceded_goals": "gk_conceded_goals", "shots_against": "gk_shots_against",
               "gk_goal_kicks": "goal_kicks", "gk_goal_kicks_success": "successful_goal_kicks",
               "short_goal_kicks": "goal_kicks_short", "long_goal_kicks": "goal_kicks_long"}
# Averages that are means of a per-event quantity, weighted by the events they average over.
AVG_WEIGHTS = {"pass_length": "passes", "long_pass_length": "long_passes",
               "dribble_distance_from_opponent_goal": "dribbles"}
# percent_X -> (numerator, denominator) where the name rules don't give it.
PCT_OVERRIDES = {"shots_on_target": ("shots_on_target", "shots"),
                 "goal_conversion": ("goals", "shots"),
                 "penalties_conversion": ("successful_penalties", "penalties"),
                 "yellow_cards_per_foul": ("yellow_cards", "fouls"),
                 "gk_saves": ("gk_saves", "gk_shots_against"),
                 "gk_successful_exits": ("gk_successful_exits", "gk_exits"),
                 "new_duels_won": ("new_duels_won", "duels"),
                 "new_defensive_duels_won": ("new_defensive_duels_won", "defensive_duels"),
                 "new_offensive_duels_won": ("new_offensive_duels_won", "offensive_duels"),
                 "new_successful_dribbles": ("new_successful_dribbles", "dribbles"),
                 "win": (None, None)}  # no wins column; percent_win is exact as a matches-weighted mean
MIN_FIT = 0.95


# ── Season name -> year / term ────────────────────────────────────────────────

def parse_season(name: str, start, end) -> dict:
    """Return year, term (Fall/Spring/Full), kind (regular/friendly/cancelled), term_source."""
    upper = (name or "").upper()
    year = re.search(r"(?:19|20)\d{2}", name or "")
    kind = "cancelled" if "CANCEL" in upper else "friendly" if "FRIENDL" in upper else "regular"
    if "FALL" in upper:
        term, source = "Fall", "name"
    elif "SPRING" in upper:
        term, source = "Spring", "name"
    else:
        s, e = pd.to_datetime(start, errors="coerce"), pd.to_datetime(end, errors="coerce")
        if pd.isna(s) or pd.isna(e):
            term = "Unknown"
        elif e.month <= 6:
            term = "Spring"
        elif s.month >= 7:
            term = "Fall"
        else:
            term = "Full"
        source = "dates"
    return {"year": int(year.group()) if year else None, "term": term, "kind": kind,
            "term_source": source}


def build_season_terms(con) -> pd.DataFrame:
    seasons = con.execute("SELECT season_id, comp_id, name, start_date, end_date FROM wyscout.seasons").df()
    parsed = pd.DataFrame([parse_season(r.name, r.start_date, r.end_date) for r in seasons.itertuples()])
    out = pd.concat([seasons[["season_id", "comp_id", "name"]], parsed], axis=1)
    L.upsert(con, "season_terms", out)
    return out


# ── Formula discovery ─────────────────────────────────────────────────────────

def _num(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").to_numpy(float) if col in df else np.full(len(df), np.nan)


def exposure(df: pd.DataFrame) -> np.ndarray:
    """Per-row divisor behind the average_* columns: sum(total)/sum(average) over EXPOSURE_STATS."""
    num = np.zeros(len(df))
    den = np.zeros(len(df))
    for s in EXPOSURE_STATS:
        t, a = _num(df, f"total_{s}"), _num(df, f"average_{s}")
        ok = ~np.isnan(t) & ~np.isnan(a) & (a > 0)
        num[ok] += t[ok]
        den[ok] += a[ok]
    return np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)


def discover_averages(df: pd.DataFrame, d: np.ndarray) -> dict[str, dict]:
    """For each average_X pick: per-exposure (total/exposure), weighted mean, or exposure-weighted."""
    out = {}
    for col in [c for c in df if c.startswith("average_")]:
        stem = col[len("average_"):]
        if stem in AVG_WEIGHTS:
            out[col] = {"method": "weighted_mean", "weight": f"total_{AVG_WEIGHTS[stem]}", "fit": None}
            continue
        total = f"total_{AVG_ALIASES.get(stem, stem)}"
        a, t = _num(df, col), _num(df, total)
        ok = (d > 0) & ~np.isnan(a) & ~np.isnan(t)
        fit = float((np.abs(t[ok] / d[ok] - a[ok]) <= 0.006 + 0.003 * np.abs(a[ok])).mean()) if ok.sum() >= 50 else 0.0
        if fit >= MIN_FIT:
            out[col] = {"method": "per_exposure", "numerator": total, "fit": fit}
        else:
            out[col] = {"method": "exposure_weighted", "fit": fit}
    return out


def pct_candidate(stem: str) -> tuple[str | None, str | None]:
    """(numerator, denominator) stat names for percent_<stem> from naming conventions."""
    if stem in PCT_OVERRIDES:
        return PCT_OVERRIDES[stem]
    if stem.startswith("successful_"):
        return stem, stem[len("successful_"):]
    if stem.endswith("_won"):
        return stem, stem[:-len("_won")]
    return None, None


def discover_percents(df: pd.DataFrame) -> dict[str, dict]:
    out = {}
    for col in [c for c in df if c.startswith("percent_")]:
        stem = col[len("percent_"):]
        n, dn = pct_candidate(stem)
        fit = 0.0
        if n and f"total_{n}" in df and f"total_{dn}" in df:
            p, num, den = _num(df, col), _num(df, f"total_{n}"), _num(df, f"total_{dn}")
            ok = ~np.isnan(p) & (den > 0) & ~np.isnan(num)
            if ok.sum() >= 50:
                fit = float((np.abs(100 * num[ok] / den[ok] - p[ok]) <= 0.06).mean())
        if fit >= MIN_FIT:
            out[col] = {"method": "ratio", "numerator": f"total_{n}", "denominator": f"total_{dn}", "fit": fit}
        else:  # weight by the events the percent is over, if known; percent_win is matches-based
            weight = f"total_{dn}" if dn and f"total_{dn}" in df else "total_matches"
            out[col] = {"method": "weighted_mean", "weight": weight, "fit": fit}
    return out


# ── Rollup ────────────────────────────────────────────────────────────────────

def rollup(df: pd.DataFrame, key: str, formulas: dict[str, dict]) -> pd.DataFrame:
    """df: season-stat rows with year/term/comp_id; returns one row per (key, year, term)."""
    df = df.copy()
    df["_exposure"] = exposure(df)
    totals = [c for c in df if c.startswith("total_")]
    groups = df.groupby([key, "year", "term"], sort=False)
    out = groups[totals + ["_exposure"]].sum(min_count=1)
    out["n_comps"] = groups["comp_id"].nunique()
    out["n_seasons"] = groups["season_id"].nunique()
    out["comp_ids"] = groups["comp_id"].agg(lambda s: sorted({int(x) for x in s}))
    out["exposure"] = out.pop("_exposure")

    def weighted(col: str, weight: np.ndarray) -> pd.Series:
        v = pd.to_numeric(df[col], errors="coerce")
        w = pd.Series(weight, index=df.index).where(v.notna(), 0.0)
        num = (v.fillna(0) * w).groupby([df[key], df["year"], df["term"]], sort=False).sum()
        den = w.groupby([df[key], df["year"], df["term"]], sort=False).sum()
        return (num / den.where(den > 0)).reindex(out.index)

    derived = {}
    for col, f in formulas.items():
        if f["method"] == "per_exposure":
            derived[col] = out[f["numerator"]] / out["exposure"].where(out["exposure"] > 0)
        elif f["method"] == "ratio":
            derived[col] = 100 * out[f["numerator"]] / out[f["denominator"]].where(out[f["denominator"]] > 0)
        elif f["method"] == "weighted_mean":
            derived[col] = weighted(col, _num(df, f["weight"]))
        else:  # exposure_weighted
            derived[col] = weighted(col, df["_exposure"].to_numpy())
    return pd.concat([out, pd.DataFrame(derived)], axis=1).reset_index()


def load_season_rows(con, table: str, key: str, include_friendlies: bool) -> pd.DataFrame:
    kinds = "('regular','friendly')" if include_friendlies else "('regular')"
    return con.execute(f"""
        SELECT t.term, t.year, s.* FROM wyscout.{table} s
        JOIN wyscout.season_terms t USING (season_id)
        WHERE t.kind IN {kinds} AND t.term <> 'Unknown'""").df()


def run_scope(con, scope: str, include_friendlies: bool) -> pd.DataFrame:
    key, table, out_table = (("team_id", "team_season_stats", "team_term_stats") if scope == "team"
                             else ("player_id", "player_season_stats", "player_term_stats"))
    df = load_season_rows(con, table, key, include_friendlies)
    if df.empty:
        print(f"{scope}: no season stats loaded yet")
        return df
    d = exposure(df)
    formulas = {**discover_averages(df, d), **discover_percents(df)}
    log_rows = [{"scope": scope, "stat": c, **{k: v for k, v in f.items()}} for c, f in formulas.items()]
    L.upsert(con, "stat_formulas", pd.DataFrame(log_rows).rename(columns={"stat": "stat_col"}))
    methods = pd.Series([f["method"] for f in formulas.values()]).value_counts().to_dict()
    result = rollup(df, key, formulas)
    L.upsert(con, out_table, result)
    print(f"{scope}: {len(df):,} season rows -> {len(result):,} {key.split('_')[0]}-terms; "
          f"{(result['n_comps'] > 1).sum():,} combine multiple competitions; methods {methods}")
    return result


# ── Validation ────────────────────────────────────────────────────────────────

def validate_team(con, include_friendlies: bool = False) -> None:
    """Compare rolled-up team totals against sums of the per-match rows already loaded."""
    kinds = "('regular','friendly')" if include_friendlies else "('regular')"
    cmp = con.execute(f"""
        WITH m AS (
            SELECT s.team_id, t.year, t.term, sum(s.total_shots) shots, sum(s.total_passes) passes,
                   sum(s.total_matches_tagged) tagged
            FROM wyscout.team_match_stats s
            JOIN wyscout.matches x USING (match_id)
            JOIN wyscout.season_terms t ON t.season_id = x.season_id
            WHERE t.kind IN {kinds} GROUP BY ALL)
        SELECT count(*) n,
               avg((r.total_shots = m.shots)::INT) shots_match,
               avg((r.total_passes = m.passes)::INT) passes_match,
               avg((r.total_matches_tagged = m.tagged)::INT) tagged_match
        FROM wyscout.team_term_stats r JOIN m USING (team_id, year, term)""").fetchone()
    print(f"validation vs match-level sums over {cmp[0]:,} team-terms: shots equal {cmp[1]:.1%}, "
          f"passes equal {cmp[2]:.1%}, tagged matches equal {cmp[3]:.1%}")
    print("(goals / total_matches are not compared: season totals include matches without event data)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scope", choices=["team", "player", "both"], default="both")
    ap.add_argument("--include-friendlies", action="store_true")
    args = ap.parse_args()
    con = L.connect()
    terms = build_season_terms(con)
    print(terms.groupby(["term", "kind"]).size().to_string())
    for scope in (["team", "player"] if args.scope == "both" else [args.scope]):
        run_scope(con, scope, args.include_friendlies)
    if args.scope in ("team", "both"):
        validate_team(con, args.include_friendlies)


if __name__ == "__main__":
    main()
