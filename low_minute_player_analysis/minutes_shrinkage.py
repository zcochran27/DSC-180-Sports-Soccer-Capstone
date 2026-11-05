"""Method 1: minutes-based shrinkage ("pull small samples toward the average").

Idea: a player's raw per-90 rate is true ability plus luck. With few minutes the luck part is big,
so we trust the raw number less and lean on the group average more:

    estimate = w * raw + (1 - w) * group_mean        where  w = minutes / (minutes + k)

* Many minutes -> w near 1 -> keep the player's own number.
* Few minutes  -> w near 0 -> estimate is close to the group average.
* k is the number of minutes at which we trust the player and the group average equally.
  A bigger k means the stat is noisier and needs more minutes to be believed.

We pick k per stat from the data (see estimate_k).

Run a quick self-check (needs .env with MotherDuck credentials):
    python -m low_minute_player_analysis.minutes_shrinkage
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import common


# --- Estimating k ---

def group_mean(rate: pd.Series, minutes: pd.Series) -> float:
    """Minutes-weighted average per-90 rate (more minutes = more weight)."""
    return float((rate * minutes).sum() / minutes.sum())


def estimate_k(rate: pd.Series, minutes: pd.Series) -> float:
    """Estimate k for one stat by splitting the spread of rates into signal and noise.

    We assume each player's observed rate is the group mean plus two independent parts:
        real differences between players   -> variance `signal` (the same for everyone)
        luck                               -> variance `noise / minutes` (shrinks with minutes)
    so  rate ~ Normal(mean, signal + noise / minutes).
    We find the signal and noise that make the observed rates most likely (maximum
    likelihood). Then k = noise / signal.

    A plain straight-line fit of squared distances on 1/minutes is simpler but is thrown off by
    the handful of players with only a few minutes, so we use the likelihood instead.
    """
    mean = group_mean(rate, minutes)
    distance = (rate - mean).to_numpy()
    inv_minutes = 1 / minutes.to_numpy()

    def negative_log_likelihood(log_params: np.ndarray) -> float:
        signal, noise = np.exp(log_params)  # exp keeps both variances positive
        variance = signal + noise * inv_minutes
        return 0.5 * np.sum(np.log(variance) + distance ** 2 / variance)

    start = np.log([rate.var() / 2 + 1e-9, rate.var() * minutes.median() / 2 + 1e-9])
    best = minimize(negative_log_likelihood, start, method="Nelder-Mead")
    signal, noise = np.exp(best.x)
    return float(noise / signal)


def estimate_all_k(rates: pd.DataFrame, minutes: pd.Series) -> pd.Series:
    """k for every stat column (index = stat name)."""
    return pd.Series({c: estimate_k(rates[c], minutes) for c in rates.columns}, name="k")


# --- Applying the shrinkage ---

def shrink(rates: pd.DataFrame, minutes: pd.Series, k: pd.Series,
           means: pd.Series | None = None) -> pd.DataFrame:
    """Shrink every stat toward its group mean. `k` and `means` are indexed by stat name.

    Pass `means` to shrink toward a mean computed elsewhere; by default it is computed here.
    """
    if means is None:
        means = pd.Series({c: group_mean(rates[c], minutes) for c in rates.columns})
    out = pd.DataFrame(index=rates.index, columns=rates.columns, dtype=float)
    for stat in rates.columns:
        weight = minutes / (minutes + k[stat])
        out[stat] = weight * rates[stat] + (1 - weight) * means[stat]
    return out


def trust_weight(minutes: pd.Series, k: float) -> pd.Series:
    """The weight w given to the player's own number (handy for plotting)."""
    return minutes / (minutes + k)


# --- Quick self-check ---

def main() -> None:
    group = "Center Forwards"
    data = common.prepare_group(common.connect(), "female", group)

    # Fit k and the means on every player-season, not just those with a next season.
    k = estimate_all_k(data.all_rates, data.terms["minutes"])
    means = pd.Series({c: group_mean(data.all_rates[c], data.terms["minutes"]) for c in data.all_rates})

    print(f"{group}: k (minutes) per stat:\n", k.round(0).to_string(), "\n")
    print("Raw per-90:\n", common.evaluate(data.raw, data.target, data.minutes).round(3), "\n")
    shrunk = shrink(data.raw, data.minutes, k, means)
    print("Shrunk:\n", common.evaluate(shrunk, data.target, data.minutes).round(3))


if __name__ == "__main__":
    main()
