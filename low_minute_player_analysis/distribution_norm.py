"""Minutes-conditional normalisation of the low-minute per-90.

A running per-90 is noisier when few minutes have been played (the noise variance is about sigma^2 / minutes)
and it also has a different average, because the players who keep getting minutes are not a random sample.
So the same raw value means something different at 100 minutes than at 700 minutes.

The fix: for each stat, look at all training rows with a similar number of minutes and store their mean and
standard deviation. A value is then described as a z-score "for that many minutes":

    z = (value - mean for these minutes) / (sd for these minutes)

Use it in two steps (fit on TRAINING rows only):

    norm = fit_minutes_norm(low_per90_frame, minutes)
    z = minutes_z(norm, low_per90_array, minutes)
"""
import numpy as np

# Edges of the 12 minute bins the means and sds are stored for.
FINE_EDGES = [0, 50, 100, 150, 200, 250, 300, 375, 450, 525, 600, 675, 750.1]
MIN_ROWS_PER_BIN = 30   # a bin with fewer rows falls back to the overall mean and sd


def minute_bin(minutes):
    """Index (0..11) of the minute bin each value falls in."""
    index = np.digitize(np.asarray(minutes), FINE_EDGES) - 1
    return np.clip(index, 0, len(FINE_EDGES) - 2)


def fit_minutes_norm(low, minutes):
    """Mean and sd of every stat in every minute bin.

    low: DataFrame of low-minute per-90 values (one column per stat). minutes: minutes behind each row.
    Returns a dict with "mu" and "sd", both arrays of shape (number of bins, number of stats).
    """
    values = low.to_numpy()
    bins = minute_bin(minutes)
    n_bins = len(FINE_EDGES) - 1
    n_stats = values.shape[1]

    mu = np.zeros((n_bins, n_stats))
    sd = np.ones((n_bins, n_stats))
    overall_mu = values.mean(axis=0)
    overall_sd = values.std(axis=0)

    for b in range(n_bins):
        rows = values[bins == b]
        if len(rows) >= MIN_ROWS_PER_BIN:
            mu[b] = rows.mean(axis=0)
            sd[b] = rows.std(axis=0)
        else:
            mu[b] = overall_mu
            sd[b] = overall_sd

    sd[sd < 1e-9] = 1.0   # a stat that never varies in a bin: avoid dividing by zero
    return {"mu": mu, "sd": sd}


def minutes_z(norm, values, minutes):
    """z-scores of `values` (rows x stats) against the distribution at each row's number of minutes."""
    bins = minute_bin(minutes)
    return (values - norm["mu"][bins]) / norm["sd"][bins]
