"""How many player-seasons can serve as the "truth" set?

A player-season is (player_id, year) over all Fall regular-season matches with minutes > 0.
Its group is the tactical group with the most minutes. Per the agreed rules the player-season is
dropped if any of its matches has a NULL / unmapped position_code, or if the top group is tied.

Typical use (read-only):
    con = common.connect()
    keep, reasons = audit_counts.assign_groups(audit_counts.load_matches(con))
    audit_counts.counts_table(keep)
"""
import pandas as pd

from low_minute_player_analysis import common

THRESHOLDS = [600, 750, 900, 1200]   # minute cut-offs shown in the count table


def load_matches(con):
    """One row per player-match: player, year, gender, position_code, minutes."""
    return con.execute("""
        SELECT m.player_id, t.year, p.gender, m.position_code, m.total_minutes_on_field AS minutes
        FROM wyscout.player_match_stats m
        JOIN wyscout.season_terms t ON m.season_id = t.season_id AND m.comp_id = t.comp_id
        JOIN wyscout.players p ON m.player_id = p.player_id
        WHERE t.term = 'Fall' AND t.kind = 'regular' AND m.total_minutes_on_field > 0
    """).df()


def assign_groups(matches):
    """One row per player-season with its main group and total minutes, after dropping unusable seasons.

    Returns (kept player-seasons, dict with the number dropped for each reason).
    """
    matches = matches.copy()
    matches["group"] = matches["position_code"].map(common.TACTICAL_MAPPING)
    keys = ["player_id", "year", "gender"]

    # one row per player-season: total minutes, matches, and how many matches had an unmapped position
    seasons = matches.groupby(keys).agg(
        minutes=("minutes", "sum"),
        n_matches=("minutes", "size"),
        n_unmapped=("group", lambda s: s.isna().sum()),
    ).reset_index()

    # minutes spent in each group; keep the top two groups of every player-season
    group_minutes = matches.dropna(subset=["group"]).groupby(keys + ["group"])["minutes"].sum().reset_index()
    group_minutes = group_minutes.sort_values(keys + ["minutes"], ascending=[True, True, True, False])
    top_two = group_minutes.groupby(keys).head(2)
    first = top_two.groupby(keys).nth(0).set_index(keys)
    second = top_two.groupby(keys).nth(1).set_index(keys)["minutes"].rename("second_minutes")

    seasons = seasons.merge(first[["group", "minutes"]].rename(columns={"minutes": "top_minutes"}),
                            left_on=keys, right_index=True, how="left")
    seasons = seasons.merge(second, left_on=keys, right_index=True, how="left")
    seasons["tied"] = seasons["second_minutes"].notna() & (seasons["second_minutes"] == seasons["top_minutes"])

    reasons = {
        "player_seasons": len(seasons),
        "with_unmapped_position": int((seasons["n_unmapped"] > 0).sum()),
        "tied_top_group": int(seasons["tied"].sum()),
    }
    usable = (seasons["n_unmapped"] == 0) & (~seasons["tied"]) & seasons["group"].notna()
    keep = seasons[usable].copy()
    reasons["kept"] = len(keep)
    return keep, reasons


def counts_table(keep):
    """Number of player-seasons per gender and group at each minute threshold."""
    rows = []
    for (gender, group), part in keep.groupby(["gender", "group"]):
        row = {"gender": gender, "group": group, "all_player_seasons": len(part),
               "median_minutes": part["minutes"].median()}
        for minutes in THRESHOLDS:
            row[f">={minutes}"] = int((part["minutes"] >= minutes).sum())
        rows.append(row)
    return pd.DataFrame(rows)
