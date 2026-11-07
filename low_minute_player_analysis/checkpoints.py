"""Data for the reconstruction validation: the truth and one training/test prefix per match played.

Agreed rules:
  * Unit: player-season = (player_id, year) over all Fall regular-season matches with minutes > 0.
  * Group: the tactical group with the most minutes. Player-seasons with any unmapped/NULL
    position_code or a tied top group are dropped (see audit_counts.assign_groups).
  * Truth set: player-seasons with at least TRUTH_MIN (900) minutes. Truth = per-90 over the whole season.
  * Prefix: a player's first k matches in date order (matches.date_utc, then match_id). One row per k as
    long as the cumulative minutes are at most MAX_MINUTES (750). Stats are summed over the prefix and
    turned into per 90 with the prefix's actual minutes.

Volume stats = every total_* count except the match/minute counters, plus the derived counts from
wyscout/derived.py that apply to players. Rates/ratios/percents are NOT built here (see rate_rebuild.py).

Outputs (parquet, in low_minute_player_analysis/results/):
    truth_<gender>.parquet     player_id, year, group, full_minutes, full_matches, full_<stat> (per 90)
    prefixes_<gender>.parquet  player_id, year, group, n_matches, actual_minutes, low_<stat> (per 90)

Build and save everything from a notebook:
    con = common.connect()
    checkpoints.build_and_save_data(con)
"""
from pathlib import Path

import pandas as pd

from low_minute_player_analysis import audit_counts, common
from wyscout import derived

TRUTH_MIN = 900
MAX_MINUTES = 750
RESULTS_DIR = Path(__file__).parent / "results"
KEYS = ["player_id", "year"]
EXCLUDE_TOTALS = {"total_matches", "total_matches_in_start", "total_matches_substituted",
                  "total_matches_coming_off", "total_minutes_on_field", "total_minutes_tagged"}


def total_columns(con):
    cols = con.execute("""SELECT column_name FROM duckdb_columns()
                          WHERE schema_name = 'wyscout' AND table_name = 'player_match_stats'
                          AND column_name LIKE 'total\\_%' ESCAPE '\\'""").df()["column_name"]
    return [c for c in cols if c not in EXCLUDE_TOTALS]


def derived_count_columns():
    """Names (total_<name>) of derived count stats that apply to players."""
    return [f"total_{e.name}" for e in derived.REGISTRY
            if e.kind == "count" and e.scope in ("both", "player") and e.origin == "totals"]


def load_match_rows(con, keep, totals):
    """Per-match stats for the kept player-seasons, with the match date."""
    con.register("keep_ps", keep[KEYS + ["gender", "group"]])
    stat_sql = ", ".join(f"m.{c}" for c in totals)
    sql = f"""
        SELECT m.player_id, t.year, m.match_id, ma.date_utc, m.total_minutes_on_field AS minutes,
               {stat_sql}
        FROM wyscout.player_match_stats m
        JOIN wyscout.season_terms t ON m.season_id = t.season_id AND m.comp_id = t.comp_id
        JOIN wyscout.matches ma ON m.match_id = ma.match_id
        JOIN keep_ps k ON k.player_id = m.player_id AND k.year = t.year
        WHERE t.term = 'Fall' AND t.kind = 'regular' AND m.total_minutes_on_field > 0
    """
    df = con.execute(sql).df()
    con.unregister("keep_ps")
    df[totals] = df[totals].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float64")
    return df


def to_p90(sums, minutes, totals):
    """Add derived counts to summed totals and convert every volume stat to per 90."""
    frame = sums[totals].copy()
    frame["exposure"] = minutes.to_numpy()  # unused by counts; add_derived expects the column
    frame, _ = derived.add_derived(frame, "player")
    cols = totals + derived_count_columns()
    cols = [c for c in dict.fromkeys(cols) if c in frame.columns]
    p90 = frame[cols].div(minutes.to_numpy(), axis=0) * 90
    p90.columns = [c.removeprefix("total_") for c in cols]
    return p90


def build_truth(rows, totals):
    """Full-season per-90 per player-season."""
    full = rows.groupby(KEYS)[["minutes"] + totals].sum()
    truth = to_p90(full, full["minutes"], totals).add_prefix("full_")
    truth.insert(0, "full_minutes", full["minutes"])
    truth["full_matches"] = rows.groupby(KEYS).size()
    return truth.reset_index()


def build_prefixes(rows, totals, max_minutes=MAX_MINUTES):
    """One row per (player-season, number of matches played) with cumulative minutes <= max_minutes."""
    rows = rows.sort_values(KEYS + ["date_utc", "match_id"]).reset_index(drop=True)
    cum = rows.groupby(KEYS)[["minutes"] + totals].cumsum()
    cum["n_matches"] = rows.groupby(KEYS).cumcount() + 1
    cum[KEYS] = rows[KEYS]
    cum = cum[cum["minutes"] <= max_minutes].reset_index(drop=True)
    p90 = to_p90(cum, cum["minutes"], totals).add_prefix("low_")
    return pd.concat([cum[KEYS + ["n_matches"]], cum["minutes"].rename("actual_minutes"), p90], axis=1)


def build_and_save_data(con):
    """Build the truth and prefix tables for both genders from the database and save them as parquet files."""
    RESULTS_DIR.mkdir(exist_ok=True)

    # player-seasons with at least TRUTH_MIN minutes, each with its main group
    keep, _ = audit_counts.assign_groups(audit_counts.load_matches(con))
    keep = keep[keep["minutes"] >= TRUTH_MIN]

    totals = total_columns(con)
    print(f"{len(totals)} total_* volume columns + {len(derived_count_columns())} derived counts")

    for gender, kept_for_gender in keep.groupby("gender"):
        rows = load_match_rows(con, kept_for_gender, totals)
        group_of_season = kept_for_gender[KEYS + ["group"]]

        truth = build_truth(rows, totals).merge(group_of_season, on=KEYS)
        prefixes = build_prefixes(rows, totals).merge(group_of_season, on=KEYS)

        truth.to_parquet(RESULTS_DIR / f"truth_{gender}.parquet")
        prefixes.to_parquet(RESULTS_DIR / f"prefixes_{gender}.parquet")
        print(f"{gender}: {len(truth)} truth player-seasons, {len(prefixes)} prefix rows "
              f"({len(prefixes) / len(truth):.1f} per season), {len(rows)} match rows")
