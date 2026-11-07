"""Rebuild every rate / ratio / percent from per-90 volume stats.

Input: a frame of per-90 volume stats (columns are stems without `total_`, e.g. `passes`, `successful_passes`,
`np_xg`), as produced by checkpoints.py or by any reconstruction method.
Output: one column per rebuildable rate, named like the database column (`percent_*`, `ratio_*`).

Formulas come from the repo, nothing is re-invented:
  * derived    - wyscout/derived.py REGISTRY (percent / ratio entries that apply to players).
  * api_exact  - API percents the roll-up reproduces as 100 * num / den (stat_formulas.method = 'ratio').
  * api_approx - API percents whose roll-up is only a weighted mean (fit < 0.95); here rebuilt as
                 100 * num / den using rollup.pct_candidate naming rules. These are NOT guaranteed to
                 equal the stored value; check_on_truth reports how far off they are.

Each rate is described by a "spec", a plain dict:
    name     database column name, e.g. "percent_pass_accuracy"
    num      numerator:   list of (coefficient, stat) pairs
    den      denominator: list of (coefficient, stat) pairs
    scale    100 for percent_*, 1 for ratio_*
    min_den  smallest denominator (on summed totals) for which the rate is defined
    source   "derived", "api_exact" or "api_approx"

Ratios are scale-free, so per-90 values can be used directly. The minimum-denominator rule in derived.py is
defined on summed totals, so pass `minutes` (the minutes the profile represents) to apply it; with
`minutes=None` a rate is only NaN when its denominator is <= 0.

Not rebuildable from volumes (they come back in the "skipped" table of load_specs): playing-time rates (need
matches), average pass length / dribble distance (need distances), and API percents with no count pair.

Typical use:
    specs, skipped = rate_rebuild.load_specs(con, set(stems))
    rates = rate_rebuild.rebuild_rates(p90, specs, minutes)
"""
import numpy as np
import pandas as pd

from low_minute_player_analysis import checkpoints
from wyscout import derived, rollup

CHECK_YEAR = 2026   # stored rates for older years are stale (they disagree with their own totals)


def make_spec(name, num, den, scale, min_den, source):
    """One rate description (see the module docstring)."""
    return {"name": name, "num": num, "den": den, "scale": scale, "min_den": min_den, "source": source}


def stored_api_percents(con):
    """The API percent columns stored in the database, with how well the roll-up reproduces them."""
    return con.execute("""SELECT stat_col, method, fit FROM wyscout.stat_formulas
                          WHERE scope = 'player' AND (origin IS NULL OR origin <> 'derived')
                            AND stat_col LIKE 'percent\\_%' ESCAPE '\\'""").df()


def stats_used(num, den):
    """All stat names that appear in a numerator and denominator."""
    return {stat for _, stat in num} | {stat for _, stat in den}


def add_spec_if_possible(specs, skipped, volume_stems, spec):
    """Append the spec if every stat it needs is a volume stat we have; otherwise record why it was skipped."""
    missing = sorted(stats_used(spec["num"], spec["den"]) - volume_stems)
    if missing:
        skipped.append({"stat": spec["name"], "reason": f"needs non-volume stat(s): {missing}"})
    else:
        specs.append(spec)


def load_specs(con, volume_stems):
    """Return (list of rebuildable specs, table of skipped stats with the reason)."""
    specs, skipped = [], []

    # 1) rates defined in wyscout/derived.py
    for entry in derived.REGISTRY:
        is_rate = entry.kind in ("percent", "ratio")
        applies_to_players = entry.scope in ("both", "player") and entry.origin == "totals"
        if is_rate and applies_to_players:
            prefix, scale = ("percent_", 100.0) if entry.kind == "percent" else ("ratio_", 1.0)
            spec = make_spec(prefix + entry.name, entry.num, entry.den, scale, entry.min_den, "derived")
            add_spec_if_possible(specs, skipped, volume_stems, spec)

    # 2) percents delivered by the API, rebuilt as 100 * numerator / denominator
    for row in stored_api_percents(con).itertuples():
        stem = row.stat_col[len("percent_"):]
        numerator, denominator = rollup.pct_candidate(stem)
        if not numerator:
            skipped.append({"stat": row.stat_col, "reason": "no count pair known"})
            continue
        source = "api_exact" if row.method == "ratio" else "api_approx"
        spec = make_spec(row.stat_col, ((1.0, numerator),), ((1.0, denominator),), 100.0, 0.0, source)
        add_spec_if_possible(specs, skipped, volume_stems, spec)

    return specs, pd.DataFrame(skipped)


def linear_combination(p90, terms):
    """sum of coefficient * stat over the (coefficient, stat) pairs in `terms`."""
    total = 0.0
    for coefficient, stat in terms:
        total = total + coefficient * p90[stat]
    return total


def rebuild_rates(p90, specs, minutes=None):
    """Compute every spec from per-90 volumes. Rows whose denominator is too small become NaN."""
    rates = {}
    for spec in specs:
        numerator = linear_combination(p90, spec["num"])
        denominator = linear_combination(p90, spec["den"])

        # the minimum-denominator rule is on season totals, so convert the per-90 denominator to a total
        if minutes is None:
            denominator_total = denominator
        else:
            denominator_total = denominator * minutes / 90.0

        if minutes is not None and spec["min_den"] > 0:
            big_enough = denominator_total > spec["min_den"] - 1e-9
        else:
            big_enough = denominator_total > 0

        usable_denominator = denominator.where(big_enough & (denominator > 0))
        rates[spec["name"]] = (spec["scale"] * numerator / usable_denominator).astype("float64")
    return pd.DataFrame(rates, index=p90.index)


def check_on_truth(con, gender, specs):
    """Sanity check: rebuild the rates from the truth profiles and compare them with the stored database values.

    Returns one row per rate with the largest and mean absolute difference (rows with matching minutes only).
    """
    truth = pd.read_parquet(checkpoints.RESULTS_DIR / f"truth_{gender}.parquet")
    truth = truth[truth["year"] == CHECK_YEAR].reset_index(drop=True)
    stems = [c[len("full_"):] for c in truth.columns
             if c.startswith("full_") and c not in ("full_minutes", "full_matches")]
    p90 = truth[[f"full_{s}" for s in stems]].set_axis(stems, axis=1)
    rebuilt = rebuild_rates(p90, specs, minutes=truth["full_minutes"])

    # the stored values for the same player-seasons
    con.register("t", truth[["player_id", "year"]])
    columns = ", ".join(f"p.{spec['name']}" for spec in specs)
    stored = con.execute(f"""SELECT p.player_id, p.year, p.total_minutes_on_field AS db_minutes, {columns}
                             FROM wyscout.player_term_stats p JOIN t USING (player_id, year)
                             WHERE p.term = 'Fall'""").df()
    con.unregister("t")

    mine = pd.concat([truth[["player_id", "year", "full_minutes"]], rebuilt], axis=1)
    both = mine.merge(stored, on=["player_id", "year"], suffixes=("", "_db"))
    # the stored table is missing a match for some 2026 rows; compare only where the minutes agree
    both = both[(both["full_minutes"] - both["db_minutes"]).abs() <= 1]

    rows = []
    for spec in specs:
        ours, theirs = both[spec["name"]], both[spec["name"] + "_db"]
        comparable = ours.notna() & theirs.notna()
        difference = (ours[comparable] - theirs[comparable]).abs()
        rows.append({"stat": spec["name"], "source": spec["source"], "n_compared": int(comparable.sum()),
                     "n_null_mismatch": int((ours.isna() != theirs.isna()).sum()),
                     "max_abs_diff": difference.max() if comparable.any() else np.nan,
                     "mean_abs_diff": difference.mean() if comparable.any() else np.nan})
    return pd.DataFrame(rows)
