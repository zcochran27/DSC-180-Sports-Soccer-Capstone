"""Method 2: optimal transport ("reshape the noisy distribution to look like a reliable one").

Idea: players with many minutes give a reliable picture of how a stat is spread across players.
Players with few minutes show a wider spread, because luck pushes some far too high and others
far too low. Optimal transport (OT) finds the cheapest way to move one pile of values onto
another. We use it to move each low-minute group onto the high-minute (reference) distribution,
which pulls the extreme values back to realistic ones.

Two versions:
* 1-D (one stat at a time): the cheapest move simply matches ranks. The player ranked at the
  30th percentile among low-minute players gets the 30th percentile value of the reference.
* Multi-stat (Sinkhorn): moves whole stat profiles at once, so a player's other stats influence
  where they land. `reg` controls how blurry the matching is (bigger = closer to the reference
  average, smaller = stricter one-to-one matching).

Caveat: the 1-D version never changes how players are ordered within a group; it only rescales
the values. Only the multi-stat version can change the order.

Run a quick self-check (needs .env with MotherDuck credentials):
    python -m low_minute_player_analysis.optimal_transport
"""
from __future__ import annotations

import numpy as np
import ot  # the POT library: "Python Optimal Transport"
import pandas as pd

from . import common

REFERENCE_MINUTES = 900  # players with at least this many minutes define the reference


# --- Reference distribution ---

def fit_reference(rates: pd.DataFrame, minutes: pd.Series) -> pd.DataFrame:
    """Per-90 rates of the players with enough minutes to be trusted."""
    return rates[minutes >= REFERENCE_MINUTES]


# --- 1-D transport (one stat at a time) ---

def transport_1d(values: pd.Series, reference: pd.Series) -> pd.Series:
    """Replace each value with the reference value at the same percentile."""
    percentile = (values.rank(method="average") - 0.5) / len(values)
    return pd.Series(np.quantile(reference, percentile), index=values.index)


# --- Multi-stat transport (Sinkhorn) ---

def sinkhorn_coupling(low: np.ndarray, ref: np.ndarray, reg: float) -> np.ndarray:
    """How much of each low-minute player is sent to each reference player.

    Rows are low-minute players, columns are reference players; each row of the result shows
    where that player's "mass" goes. Both inputs must already be on the same scale.
    """
    cost = ot.dist(low, ref)  # squared distance between every pair
    cost = cost / cost.max()  # keeps `reg` on a comparable scale across data sets
    weights_low = np.full(len(low), 1 / len(low))  # every player counts equally
    weights_ref = np.full(len(ref), 1 / len(ref))
    return ot.sinkhorn(weights_low, weights_ref, cost, reg)


def transport_sinkhorn(group: pd.DataFrame, reference: pd.DataFrame, reg: float = 0.05) -> pd.DataFrame:
    """Move a group of stat profiles onto the reference cloud.

    Steps: put every stat on the same scale, let Sinkhorn decide how much of each player goes
    to each reference player (the "coupling"), then take the coupling-weighted average of the
    reference profiles as the player's new profile.
    """
    mean, std = reference.mean(), reference.std()
    low = ((group - mean) / std).to_numpy()
    ref = ((reference - mean) / std).to_numpy()

    coupling = sinkhorn_coupling(low, ref, reg)
    moved = coupling @ ref / coupling.sum(axis=1, keepdims=True)
    return pd.DataFrame(moved * std.to_numpy() + mean.to_numpy(),
                        index=group.index, columns=group.columns)


# --- Apply to every low-minute bucket ---

def transport_low_minutes(rates: pd.DataFrame, minutes: pd.Series, reference: pd.DataFrame,
                          how: str = "1d", reg: float = 0.05) -> pd.DataFrame:
    """Move each low-minute bucket onto the reference. Players with enough minutes are untouched.

    Buckets are moved separately because the lower the minutes, the wider the raw spread, so
    each bucket needs its own correction.
    """
    out = rates.copy()
    buckets = common.minute_bucket(minutes)
    for bucket in common.BUCKET_LABELS[:-1]:  # every bucket except the last (900+)
        rows = rates.index[buckets == bucket]
        if len(rows) < 20:
            continue
        if how == "1d":
            for stat in rates.columns:
                out.loc[rows, stat] = transport_1d(rates.loc[rows, stat], reference[stat])
        else:
            out.loc[rows] = transport_sinkhorn(rates.loc[rows], reference, reg)
    return out


# --- Quick self-check ---

def main() -> None:
    group = "Center Forwards"
    data = common.prepare_group(common.connect(), "female", group)

    # The reference comes from all high-minute player-seasons, not only those with a next season.
    reference = fit_reference(data.all_rates, data.terms["minutes"])

    print(f"{group}: raw per-90:\n", common.evaluate(data.raw, data.target, data.minutes).round(3), "\n")
    for how in ["1d", "sinkhorn"]:
        moved = transport_low_minutes(data.raw, data.minutes, reference, how=how)
        print(f"OT ({how}):\n", common.evaluate(moved, data.target, data.minutes).round(3), "\n")


if __name__ == "__main__":
    main()
