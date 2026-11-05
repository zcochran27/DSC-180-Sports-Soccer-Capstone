"""Shared helpers for the small-sample (low-minutes) player evaluation experiments.

The question: a player with 150 minutes has a very noisy goals/90 or key-passes/90. How do we
estimate what that player is really like? Three methods try to answer it (see the other files in
this folder). They all use the helpers here so they are scored the same way.

How we score a method ("next-season stability"):
    1. Estimate each player's per-90 rates from their Fall season in year t.
    2. Compare the estimate with the same player's rates in year t+1 (only for players who
       played enough minutes in t+1 for that target to be trustworthy).
    3. Report Spearman (do we rank players correctly?) and RMSE (are the values close?)
       split by how many minutes the player had in year t.

A better estimate of "true ability" should predict next season better than the raw per-90.

Usage:
    from low_minute_player_analysis import common
    con = common.connect()
    terms = common.load_player_terms(con, gender="female", role="FW")
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import spearmanr

load_dotenv()

# ── Settings ──────────────────────────────────────────────────────────────────

# Stats (without the total_ prefix) we track for each position group.
STAT_COLS = {
    "FW": ["goals", "xg_shot", "shots", "shots_on_target", "assists", "xg_assist",
           "key_passes", "touch_in_box", "successful_dribbles", "offensive_duels_won"],
    "MD": ["passes", "progressive_passes", "key_passes", "xg_assist", "interceptions",
           "recoveries", "duels_won", "successful_dribbles"],
    "DF": ["interceptions", "clearances", "defensive_duels_won", "aerial_duels_won",
           "recoveries", "progressive_passes", "passes", "fouls"],
}

# Wyscout position codes (per match) grouped into tactical roles.
TACTICAL_MAPPING = {
    "gk": "GK",
    "lb": "Outside Backs", "rb": "Outside Backs", "lwb": "Outside Backs", "rwb": "Outside Backs",
    "rb5": "Outside Backs", "lb5": "Outside Backs",
    "cb": "Center Backs", "lcb": "Center Backs", "rcb": "Center Backs",
    "lcb3": "Center Backs", "rcb3": "Center Backs",
    "lw": "Wingers", "rw": "Wingers", "lwf": "Wingers", "rwf": "Wingers",
    "cf": "Center Forwards",
    "amf": "Attacking Midfielders", "lamf": "Attacking Midfielders", "ramf": "Attacking Midfielders",
    "dmf": "Central Midfielders", "ldmf": "Central Midfielders", "rdmf": "Central Midfielders",
    "lcmf": "Central Midfielders", "rcmf": "Central Midfielders",
    "lcmf3": "Central Midfielders", "rcmf3": "Central Midfielders",
}
GROUPS = list(dict.fromkeys(TACTICAL_MAPPING.values()))  # unique groups, in the order above

# Stats tracked for each tactical group. They are added to STAT_COLS so every helper below
# accepts a group name wherever it accepts a role.
STAT_COLS.update({
    "GK": ["gk_saves", "gk_shots_against", "gk_conceded_goals", "xg_save", "gk_exits",
           "passes", "successful_passes", "long_passes"],
    "Outside Backs": ["crosses", "successful_crosses", "progressive_passes", "successful_dribbles",
                      "defensive_duels_won", "interceptions", "recoveries", "key_passes",
                      "touch_in_box", "progressive_run"],
    "Center Backs": ["interceptions", "clearances", "defensive_duels_won", "aerial_duels_won",
                     "recoveries", "progressive_passes", "passes", "fouls"],
    "Wingers": ["goals", "xg_shot", "shots", "assists", "xg_assist", "key_passes",
                "successful_dribbles", "touch_in_box", "crosses", "offensive_duels_won"],
    "Center Forwards": ["goals", "xg_shot", "shots", "shots_on_target", "assists", "xg_assist",
                        "key_passes", "touch_in_box", "successful_dribbles", "offensive_duels_won"],
    "Attacking Midfielders": ["goals", "xg_shot", "shots", "assists", "xg_assist", "key_passes",
                              "progressive_passes", "successful_dribbles", "touch_in_box",
                              "passes_to_final_third"],
    "Central Midfielders": ["passes", "progressive_passes", "key_passes", "interceptions",
                            "recoveries", "duels_won", "successful_dribbles",
                            "passes_to_final_third"],
})

# Minute buckets used to report results. Players in the last bucket have "enough" minutes.
BUCKET_EDGES = [0, 250, 500, 900, np.inf]
BUCKET_LABELS = ["0-250", "250-500", "500-900", "900+"]

# A player's next-season rate is only used as the target if they played at least this long.
MIN_TARGET_MINUTES = 600


# ── Loading data ──────────────────────────────────────────────────────────────

def connect() -> duckdb.DuckDBPyConnection:
    """Open a MotherDuck connection using MOTHERDUCK_TOKEN / MOTHERDUCK_DB from .env."""
    db = os.getenv("MOTHERDUCK_DB", "wyscout_dev")
    token = os.getenv("MOTHERDUCK_TOKEN")
    return duckdb.connect(f"md:{db}?motherduck_token={token}")


def load_player_terms(con, gender: str, role: str) -> pd.DataFrame:
    """One row per player per Fall season: minutes plus total_<stat> columns for `role`.

    Spring terms are almost empty in this data, so each year is just its Fall season.
    """
    stat_sql = ", ".join(f"s.total_{c}" for c in STAT_COLS[role])
    sql = f"""
        SELECT s.player_id, s.year, s.total_minutes_on_field AS minutes, {stat_sql}
        FROM wyscout.player_term_stats s
        JOIN wyscout.players p USING (player_id)
        WHERE s.term = 'Fall' AND p.gender = '{gender}' AND p.role_code2 = '{role}'
          AND s.total_minutes_on_field > 0
    """
    return con.execute(sql).df()


def load_group_terms(con, gender: str, group: str) -> pd.DataFrame:
    """One row per player per Fall season for a tactical group, built from match rows.

    Only matches where the player's position code belongs to `group` are counted, so minutes
    and stats describe how the player did *in that role*. A player who split a season between
    two groups appears once in each, with only the minutes spent there.
    """
    codes = ", ".join(f"'{code}'" for code, g in TACTICAL_MAPPING.items() if g == group)
    stat_sql = ", ".join(f"SUM(m.total_{c}) AS total_{c}" for c in STAT_COLS[group])
    sql = f"""
        SELECT m.player_id, t.year, SUM(m.total_minutes_on_field) AS minutes, {stat_sql}
        FROM wyscout.player_match_stats m
        JOIN wyscout.season_terms t ON m.season_id = t.season_id AND m.comp_id = t.comp_id
        JOIN wyscout.players p ON m.player_id = p.player_id
        WHERE t.term = 'Fall' AND t.kind = 'regular' AND p.gender = '{gender}'
          AND m.position_code IN ({codes}) AND m.total_minutes_on_field > 0
        GROUP BY m.player_id, t.year
    """
    return con.execute(sql).df()


def load_player_matches(con, gender: str, role: str) -> pd.DataFrame:
    """One row per player per match (regular-season Fall matches), with year and minutes."""
    stat_sql = ", ".join(f"m.total_{c}" for c in STAT_COLS[role])
    sql = f"""
        SELECT m.player_id, t.year, m.match_id, m.total_minutes_on_field AS minutes, {stat_sql}
        FROM wyscout.player_match_stats m
        JOIN wyscout.season_terms t ON m.season_id = t.season_id AND m.comp_id = t.comp_id
        JOIN wyscout.players p ON m.player_id = p.player_id
        WHERE t.term = 'Fall' AND t.kind = 'regular'
          AND p.gender = '{gender}' AND p.role_code2 = '{role}'
          AND m.total_minutes_on_field > 0
    """
    return con.execute(sql).df()


# ── Per-90 rates and the evaluation pairs ─────────────────────────────────────

def per90(df: pd.DataFrame, role: str) -> pd.DataFrame:
    """Return a frame of per-90 rates (one column per stat, no total_ prefix)."""
    rates = {c: df[f"total_{c}"] / df["minutes"] * 90 for c in STAT_COLS[role]}
    return pd.DataFrame(rates, index=df.index)


def make_next_year_pairs(df: pd.DataFrame, role: str) -> pd.DataFrame:
    """Pair each player's year t with their year t+1.

    Returns one row per (player, t) with columns: player_id, year, minutes (in t),
    raw_<stat> (per-90 in t) and next_<stat> (per-90 in t+1). Rows where the player played
    fewer than MIN_TARGET_MINUTES in t+1 are dropped, because their target is itself noisy.
    """
    rates = per90(df, role).add_prefix("raw_")
    now = pd.concat([df[["player_id", "year", "minutes"]], rates], axis=1)

    nxt = now.rename(columns=lambda c: c.replace("raw_", "next_") if c.startswith("raw_") else c)
    nxt = nxt.rename(columns={"minutes": "next_minutes"})
    nxt["year"] = nxt["year"] - 1  # so that "year" means the earlier season when we merge

    pairs = now.merge(nxt, on=["player_id", "year"])
    return pairs[pairs["next_minutes"] >= MIN_TARGET_MINUTES].reset_index(drop=True)


# ── Scoring ───────────────────────────────────────────────────────────────────

def minute_bucket(minutes: pd.Series) -> pd.Series:
    """Label each player with the minute bucket they fall in."""
    return pd.cut(minutes, BUCKET_EDGES, labels=BUCKET_LABELS, right=False)


def evaluate(estimate: pd.DataFrame, target: pd.DataFrame, minutes: pd.Series) -> pd.DataFrame:
    """Score estimates against next-season targets, overall and by minute bucket.

    `estimate` and `target` have the same index and one column per stat. For each stat we
    compute Spearman and an RMSE on the target's standard-deviation scale (so stats with big
    numbers do not dominate), then average over stats.
    """
    buckets = minute_bucket(minutes)
    scale = target.std()
    rows = []
    for name, mask in [("all", np.ones(len(minutes), bool))] + [
        (b, (buckets == b).to_numpy()) for b in BUCKET_LABELS
    ]:
        if mask.sum() < 20:
            continue
        est, tgt = estimate[mask], target[mask]
        rho = [spearmanr(est[c], tgt[c])[0] for c in est.columns]
        rmse = (((est.to_numpy() - tgt.to_numpy()) / scale.to_numpy()) ** 2).mean(axis=0) ** 0.5
        rows.append({"bucket": name, "n": int(mask.sum()),
                     "spearman": np.nanmean(rho), "rmse": rmse.mean()})
    return pd.DataFrame(rows).set_index("bucket")


def split_pairs(pairs: pd.DataFrame, role: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Split a pairs frame into (raw per-90, next-year per-90, minutes), columns named by stat."""
    stats = STAT_COLS[role]
    raw = pairs[[f"raw_{c}" for c in stats]].set_axis(stats, axis=1)
    target = pairs[[f"next_{c}" for c in stats]].set_axis(stats, axis=1)
    return raw, target, pairs["minutes"]


# ── One tactical group, ready to use ──────────────────────────────────────────

@dataclass
class GroupData:
    """Everything the experiments need for one tactical group."""
    group: str
    terms: pd.DataFrame      # one row per player-season (minutes + total_<stat>)
    all_rates: pd.DataFrame  # per-90 for every player-season
    pairs: pd.DataFrame      # season t paired with season t+1
    raw: pd.DataFrame        # per-90 in season t, for the paired players
    target: pd.DataFrame     # per-90 in season t+1, for the paired players
    minutes: pd.Series       # minutes in season t, for the paired players


def prepare_group(con, gender: str, group: str) -> GroupData:
    """Load and prepare the data for one tactical group."""
    terms = load_group_terms(con, gender, group)
    pairs = make_next_year_pairs(terms, group)
    raw, target, minutes = split_pairs(pairs, group)
    return GroupData(group, terms, per90(terms, group), pairs, raw, target, minutes)
