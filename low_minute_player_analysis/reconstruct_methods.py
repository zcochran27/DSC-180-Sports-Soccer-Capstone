"""Methods that estimate a player's full-season (900+ minute) per-90 volume profile from a low-minute one.

ONE model per gender and position group, trained on every per-game prefix (checkpoints.py) of the training
players, each with its full-season target. The same fitted model handles any number of minutes up to 750.

Every method has the same signature:  method(T, E, stems, cache) -> DataFrame (index = E.index, columns = stems)
    T      training rows. Columns: low_<stat>, full_<stat> for each stat in stems, actual_minutes, n_matches,
           player_id, year, full_minutes, weight. Whole player-seasons, sorted by season and match number.
           Never contains a player being predicted.
    E      rows to predict (same layout; full_<stat> is NOT used).
    stems  the stat names.
    cache  a dict shared by the calls of one fold, so a model is fitted once and reused.
`weight` = 1 / (rows of that player-season in the same minutes bin), so each season counts once per bin.

Methods in this file: raw, shrink, ridge, knn, ot_1d, ot_joint, ot_group.
(kalman, kalman_blend, ridge_kalman and gru are in temporal_methods.py; they read the sequence of matches.)

How each one handles different amounts of data:
  raw     the low per-90 as it is.
  shrink  weight w = m / (m + k) on the player's value and 1 - w on the group mean; m = minutes, k fitted once
          on all prefixes (weighted Normal maximum likelihood, one k per stat).
  ridge   features: the low per-90 as z-scores against the training distribution at that many minutes
          (distribution_norm), the same z times s(m) = sqrt(250 / m), log minutes, matches played and minutes
          per match. The weight on the noisy value can therefore fall as minutes grow.
          Weighted RidgeCV, one alpha per stat.
  knn, ot_*  cannot take weights or a minutes input, so training rows are bucketed by minutes (BUCKET_EDGES):
          for each bucket the method is fitted on one prefix per training player-season (the one closest to the
          bucket centre) and used for the test rows whose minutes fall in that bucket.
"""
import numpy as np
import ot
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from low_minute_player_analysis import distribution_norm, stat_groups

KEYS = ["player_id", "year"]

# Stats that can legitimately be negative (differences). Every other estimate is clipped at 0.
SIGNED = {"goals_minus_xg", "xa_minus_assists", "goals_prevented", "np_goals"}

KNN_K = 30                                    # neighbours used by the KNN method
RIDGE_ALPHAS = np.logspace(-2, 4, 13)         # ridge penalties to choose from
SINKHORN_REGS = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]   # entropy settings to choose from for optimal transport
OT_FIT_ROWS = 500                             # rows used to choose the entropy setting
OT_REF_ROWS = 1500                            # reference (truth) rows transported onto; caps the cost matrix size
REF_MINUTES = 250.0                           # s(m) = sqrt(REF_MINUTES / m)
BUCKET_EDGES = [0, 125, 250, 375, 500, 625, 750.1]   # minute buckets for knn / ot
BUCKET_PAD = 40                               # training rows may lie this many minutes outside a bucket


# ---------------------------------------------------------------- small helpers

def get_low(df, stems):
    """The low-minute per-90 columns of df, renamed to the plain stat names."""
    return df[[f"low_{s}" for s in stems]].set_axis(stems, axis=1)


def get_full(df, stems):
    """The full-season (truth) per-90 columns of df, renamed to the plain stat names."""
    return df[[f"full_{s}" for s in stems]].set_axis(stems, axis=1)


def clip_estimates(estimates):
    """Counts cannot be negative: clip every estimate at 0 except the stats that are differences."""
    columns = [c for c in estimates.columns if c not in SIGNED]
    estimates[columns] = estimates[columns].clip(lower=0)
    return estimates


# ---------------------------------------------------------------- baseline

def raw(T, E, stems, cache):
    """No adjustment: use the low-minute per-90 as the estimate."""
    return get_low(E, stems).copy()


# ---------------------------------------------------------------- minutes shrinkage

def shrinkage_negative_log_likelihood(log_params, deviation, inverse_minutes, weight):
    """Weighted Normal negative log-likelihood. Variance of a row = signal + noise / minutes."""
    signal, noise = np.exp(log_params)
    variance = signal + noise * inverse_minutes
    return 0.5 * np.sum(weight * (np.log(variance) + deviation ** 2 / variance))


def weighted_k(rate, minutes, weight):
    """(k, weighted mean) for one stat, from rate ~ Normal(mean, signal + noise / minutes), with k = noise / signal."""
    r, m, w = rate.to_numpy(), minutes.to_numpy(), weight.to_numpy()
    w = w / w.mean()
    mean = float((r * m * w).sum() / (m * w).sum())
    deviation = r - mean
    inverse_minutes = 1 / m

    start = np.log([rate.var() / 2 + 1e-9, rate.var() * np.median(m) / 2 + 1e-9])
    fit = minimize(shrinkage_negative_log_likelihood, start, args=(deviation, inverse_minutes, w),
                   method="Nelder-Mead")
    signal, noise = np.exp(fit.x)
    return float(noise / signal), mean


def shrink(T, E, stems, cache):
    """Pull the low per-90 towards the group mean: estimate = w * value + (1 - w) * mean, w = m / (m + k)."""
    if "k_w" not in cache:
        low = get_low(T, stems)
        fits = {s: weighted_k(low[s], T["actual_minutes"], T["weight"]) for s in stems}
        cache["k_w"] = pd.Series({s: fit[0] for s, fit in fits.items()})
        cache["mean_w"] = pd.Series({s: fit[1] for s, fit in fits.items()})

    minutes = E["actual_minutes"].to_numpy()[:, None]
    w = minutes / (minutes + cache["k_w"].to_numpy())
    estimate = w * get_low(E, stems).to_numpy() + (1 - w) * cache["mean_w"].to_numpy()
    return pd.DataFrame(estimate, index=E.index, columns=stems)


# ---------------------------------------------------------------- ridge regression

def ridge_features(df, stems, norm):
    """Feature matrix: [z, z * s(m), log minutes, matches, minutes per match].

    z is the minutes-conditional z-score of each low per-90 (distribution_norm).
    """
    minutes = df["actual_minutes"].to_numpy()
    s = np.sqrt(REF_MINUTES / minutes)[:, None]
    z = distribution_norm.minutes_z(norm, get_low(df, stems).to_numpy(), minutes)
    matches = df["n_matches"].to_numpy()
    return np.hstack([z, z * s, np.log(minutes)[:, None], matches[:, None], (minutes / matches)[:, None]])


def ridge_feature_names(stems):
    """Names of the columns of ridge_features, in order."""
    return ([f"z:{s}" for s in stems] + [f"z*s(m):{s}" for s in stems] +
            ["log_minutes", "matches", "minutes_per_match"])


def fit_ridge(T, stems):
    """Fit the final model. Returns (minutes normalizer, feature scaler, fitted ridge)."""
    norm = distribution_norm.fit_minutes_norm(get_low(T, stems), T["actual_minutes"])
    features = ridge_features(T, stems, norm)
    scaler = StandardScaler().fit(features)
    model = RidgeCV(alphas=RIDGE_ALPHAS, alpha_per_target=True)
    model.fit(scaler.transform(features), get_full(T, stems).to_numpy(), sample_weight=T["weight"].to_numpy())
    return norm, scaler, model


def ridge(T, E, stems, cache):
    """The final model: ridge regression on minutes-conditional z-scores."""
    if "ridge" not in cache:
        cache["ridge"] = fit_ridge(T, stems)
    norm, scaler, model = cache["ridge"]
    prediction = model.predict(scaler.transform(ridge_features(E, stems, norm)))
    return clip_estimates(pd.DataFrame(prediction, index=E.index, columns=stems))


# ---------------------------------------------------------------- minute buckets (for knn and ot)

def bucket_of(minutes):
    """Index of the minutes bucket each value falls in."""
    index = np.digitize(minutes, BUCKET_EDGES) - 1
    return np.clip(index, 0, len(BUCKET_EDGES) - 2)


def bucket_training_rows(T):
    """One row per training player-season per bucket (the prefix closest to the bucket centre).

    The `checkpoint` column holds the bucket number.
    """
    parts = []
    minutes = T["actual_minutes"]
    for b in range(len(BUCKET_EDGES) - 1):
        low_edge, high_edge = BUCKET_EDGES[b], BUCKET_EDGES[b + 1]
        in_bucket = T[(minutes >= low_edge - BUCKET_PAD) & (minutes <= high_edge + BUCKET_PAD)].copy()
        in_bucket["distance"] = (in_bucket["actual_minutes"] - (low_edge + high_edge) / 2).abs()
        in_bucket = in_bucket.sort_values("distance").drop_duplicates(KEYS).drop(columns="distance")
        in_bucket["checkpoint"] = b
        parts.append(in_bucket)
    return pd.concat(parts, ignore_index=True)


def predict_by_bucket(per_bucket_method, T, E, stems, cache):
    """Run per_bucket_method(Tc, Eb, bucket, stems, cache) once per minutes bucket and stitch the results.

    Tc = training rows of the bucket, Eb = the test rows whose minutes fall in the bucket.
    """
    if "bucketed_training_rows" not in cache:
        cache["bucketed_training_rows"] = bucket_training_rows(T)
    training_rows = cache["bucketed_training_rows"]

    estimates = pd.DataFrame(np.nan, index=E.index, columns=stems)
    test_buckets = bucket_of(E["actual_minutes"].to_numpy())
    for b in np.unique(test_buckets):
        test_rows = E[test_buckets == b]
        train_rows = training_rows[training_rows["checkpoint"] == b]
        result = per_bucket_method(train_rows, test_rows, int(b), stems, cache)
        estimates.loc[test_rows.index] = result.to_numpy()
    return estimates


# ---------------------------------------------------------------- similar-player KNN

def knn_one_bucket(Tc, E, bucket, stems, cache):
    """Average the full-season profile of the KNN_K training players whose low profile is closest."""
    scaler = StandardScaler().fit(get_low(Tc, stems))
    model = KNeighborsRegressor(n_neighbors=min(KNN_K, len(Tc)))
    model.fit(scaler.transform(get_low(Tc, stems)), get_full(Tc, stems).to_numpy())
    prediction = model.predict(scaler.transform(get_low(E, stems)))
    return pd.DataFrame(prediction, index=E.index, columns=stems)


def knn(T, E, stems, cache):
    return predict_by_bucket(knn_one_bucket, T, E, stems, cache)


# ---------------------------------------------------------------- optimal transport

def middle_rank(values, sample):
    """Empirical CDF of `values` within `sample` (0 to 1). Ties get the middle of their rank interval."""
    sorted_sample = np.sort(sample)
    below = np.searchsorted(sorted_sample, values, "left")
    up_to = np.searchsorted(sorted_sample, values, "right")
    return (below + up_to) / (2.0 * len(sorted_sample))


def ot_1d_one_bucket(Tc, E, bucket, stems, cache):
    """Per-stat quantile matching: map the low-minute distribution (training rows, same bucket) onto the truth."""
    low_train, full_train, low_test = get_low(Tc, stems), get_full(Tc, stems), get_low(E, stems)
    estimates = {}
    for s in stems:
        rank = middle_rank(low_test[s].to_numpy(), low_train[s].to_numpy())
        estimates[s] = np.quantile(full_train[s].to_numpy(), np.clip(rank, 0, 1))
    return clip_estimates(pd.DataFrame(estimates, index=E.index))


def sinkhorn_map(source, reference, reg):
    """Image of each source row under entropic optimal transport: the coupling-weighted mean of the reference rows."""
    cost = ot.dist(source, reference)
    cost = cost / cost.max()
    source_mass = np.full(len(source), 1 / len(source))
    reference_mass = np.full(len(reference), 1 / len(reference))
    plan = ot.sinkhorn(source_mass, reference_mass, cost, reg, method="sinkhorn", numItermax=300, stopThr=1e-5)
    if not np.isfinite(plan).all():   # numbers underflowed at a small reg: redo in the log domain
        plan = ot.sinkhorn(source_mass, reference_mass, cost, reg, method="sinkhorn_log",
                           numItermax=300, stopThr=1e-5)
    mass = plan.sum(axis=1, keepdims=True)
    return (plan @ reference) / np.where(mass > 0, mass, 1.0)


def ot_block(Tc, E, block, cache, key):
    """Sinkhorn transport of the test low-minute rows onto the training truth profiles, for one block of stats.

    The entropy setting `reg` is chosen on training rows only: transport a subsample of the training low
    profiles to the training truth distribution (unpaired) and keep the reg whose images are closest to
    those players' own truth. Returns the image in the original units.
    """
    # standardise with the truth's mean and sd
    full = get_full(Tc, block)
    mean = full.mean().to_numpy()
    sd = full.std().replace(0, 1).to_numpy()
    reference = (full.to_numpy() - mean) / sd

    rng = np.random.default_rng(0)
    reference_rows = rng.choice(len(reference), min(OT_REF_ROWS, len(reference)), replace=False)

    if key not in cache:
        fit_rows = rng.choice(len(Tc), min(OT_FIT_ROWS, len(Tc)), replace=False)
        low_train = (get_low(Tc, block).to_numpy() - mean) / sd
        source, own_truth = low_train[fit_rows], reference[fit_rows]
        errors = {}
        for reg in SINKHORN_REGS:
            image = sinkhorn_map(source, reference[reference_rows], reg)
            errors[reg] = np.sqrt(np.mean((image - own_truth) ** 2))
        cache[key] = min(errors, key=errors.get)

    low_test = (get_low(E, block).to_numpy() - mean) / sd
    image = sinkhorn_map(low_test, reference[reference_rows], cache[key])
    return pd.DataFrame(image * sd + mean, index=E.index, columns=block)


def ot_joint_one_bucket(Tc, E, bucket, stems, cache):
    """One Sinkhorn transport over the whole stat vector."""
    return clip_estimates(ot_block(Tc, E, stems, cache, ("reg_joint", bucket)))


def ot_group_one_bucket(Tc, E, bucket, stems, cache):
    """One Sinkhorn transport per stat group (wyscout/STAT_GROUPS.md), then recombine.

    Each stat belongs to exactly one group (stat_groups.PRIMARY_GROUP resolves the stats the file lists
    twice), so the blocks partition the stat vector. A stat in no group (none today) is transported
    together with the other such stats as its own block.
    """
    groups, leftover = stat_groups.stat_to_groups(stems)
    blocks = dict(groups)
    if leftover:
        blocks["_ungrouped"] = leftover

    total = pd.DataFrame(0.0, index=E.index, columns=stems)
    times_estimated = pd.Series(0, index=stems, dtype=float)
    for name, block in blocks.items():
        total[block] += ot_block(Tc, E, block, cache, (f"reg_{name}", bucket))
        times_estimated[block] += 1
    return clip_estimates(total / times_estimated)


def ot_1d(T, E, stems, cache):
    return predict_by_bucket(ot_1d_one_bucket, T, E, stems, cache)


def ot_joint(T, E, stems, cache):
    return predict_by_bucket(ot_joint_one_bucket, T, E, stems, cache)


def ot_group(T, E, stems, cache):
    return predict_by_bucket(ot_group_one_bucket, T, E, stems, cache)


METHODS = {"raw": raw, "shrink": shrink, "ridge": ridge, "knn": knn,
           "ot_1d": ot_1d, "ot_joint": ot_joint, "ot_group": ot_group}
