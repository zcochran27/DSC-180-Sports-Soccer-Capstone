"""Use the final model (ridge, reconstruct_methods.ridge) on real low-minute player-seasons.

The validation fits on 900+ minute player-seasons (their prefixes) and scores held-out ones. Here the model is fitted
on ALL of a group's 900+ minute player-seasons and applied to player-seasons that ended with fewer minutes (by
default 150-750), using their whole season as the low-minute sample. Needs the database (common.connect()).

    pool, stems = fit_data("male", "Wingers")                 # training prefixes with truth
    low = low_minute_seasons(con, "male", "Wingers", stems)   # real low-minute player-seasons
    est = predict(pool, low, stems)                           # estimated full-season per-90 volumes
    rates = rebuild(con, est, low, stems)                     # rates / percents rebuilt from those volumes
"""
from low_minute_player_analysis import audit_counts, checkpoints, rate_rebuild, reconstruction_eval as ev
from low_minute_player_analysis import reconstruct_methods as rm

KEYS = ["player_id", "year"]


def fit_data(gender, group):
    """(training pool, stats). The pool holds every prefix of the group's 900+ minute player-seasons with its truth."""
    stems, _ = ev.group_setup(gender, group)
    return ev.load_pool(gender, group, stems), stems


def low_minute_seasons(con, gender, group, stems, min_minutes=150, max_minutes=750):
    """One row per real player-season of the group with min_minutes <= season minutes <= max_minutes.

    The row is the whole season: low_<stat> is the season's per-90, actual_minutes its minutes, n_matches its matches.
    The model was trained on prefixes of up to 750 minutes, so max_minutes should not exceed 750.
    """
    # player-seasons of this gender and group with a season total in the minutes range
    keep, _ = audit_counts.assign_groups(audit_counts.load_matches(con))
    in_range = keep["minutes"].between(min_minutes, max_minutes)
    seasons = keep[(keep["gender"] == gender) & (keep["group"] == group) & in_range]

    # per-match stats -> running per-90 after each match
    totals = checkpoints.total_columns(con)
    match_rows = checkpoints.load_match_rows(con, seasons, totals)
    prefixes = checkpoints.build_prefixes(match_rows, totals, max_minutes=max_minutes)

    # keep the last prefix of each season, and only seasons seen in full
    last = prefixes.sort_values(KEYS + ["n_matches"]).groupby(KEYS).tail(1)
    last = last.merge(seasons[KEYS + ["minutes"]], on=KEYS)
    last = last[(last["actual_minutes"] - last["minutes"]).abs() < 1e-6]

    names = con.execute("SELECT player_id, short_name FROM wyscout.players").df()
    last = last.merge(names, on="player_id", how="left")
    needed = [f"low_{s}" for s in stems]
    return last.drop(columns="minutes").dropna(subset=needed).sort_values(KEYS).reset_index(drop=True)


def predict(pool, low, stems):
    """Estimated full-season per-90 volumes (index = low.index) from the final ridge fitted on `pool`."""
    return rm.ridge(pool, low, stems, {})


def rebuild(con, est, low, stems):
    """Rates, ratios and percents rebuilt from the estimated volumes (NaN where the denominator is too small)."""
    specs, _ = rate_rebuild.load_specs(con, set(stems))
    specs = [s for s in specs if rate_rebuild.stats_used(s["num"], s["den"]) <= set(stems)]
    return rate_rebuild.rebuild_rates(est, specs, low["actual_minutes"])
