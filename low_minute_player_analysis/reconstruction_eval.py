"""Evaluation: run every method out-of-fold and score volumes and rebuilt rates, by minutes bin.

Setup (per gender and position group):
  * Training and test rows are per-game prefixes (checkpoints.py): every prefix, up to 750 minutes, of a
    900+ minute player-season, each with its full-season per-90 as the target.
  * 5-fold GroupKFold by player_id over the group's player-seasons: all seasons and prefixes of a player share
    a fold. Each method is fitted on the training folds and predicts every prefix of the held-out players.
  * Minutes bins of the prefix's actual minutes: [0,375) -> 250, [375,625) -> 500, [625,750] -> 750.
  * Weights: 1 / (rows of that player-season in the bin), for training (ridge, shrinkage, GRU) and for scoring,
    so each player-season counts once per bin.

Scoring (per gender, group, bin, method, stat):
  volume stats  weighted RMSE in per-90 units, nrmse = RMSE / (weighted std of the truth), weighted Spearman.
  rates         the estimate's volume profile is pushed through rate_rebuild.rebuild_rates and compared with the
                rates rebuilt from the truth profile (percent_* on 0-100). Rates are compared only where BOTH
                exist; n_truth_ok, n_est_missing and n_compared are reported separately.
  The raw low-minute baseline is a method like any other (method == "raw").
GK-only stats (gk_*, goal_kicks*, xg_save, goals_prevented) are used for the GK group only.

Also written: <tag>_paired.parquet, the mean per player-season difference in squared error (divided by the bin's
truth variance, averaged over stats) of each method versus the reference "ridge", with a bootstrap 95% interval
over player-seasons. Negative = better than the reference.

Run everything from a notebook (takes hours for all 14 gender/group sets):
    reconstruction_eval.run_evaluation(n_jobs=8)                       # all groups, genders and methods
    reconstruction_eval.run_evaluation(groups=["GK"], genders=["female"], methods=["raw", "ridge"], tag="test")
"""
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold

from low_minute_player_analysis import checkpoints, common, rate_rebuild, reconstruct_methods, temporal_methods

N_FOLDS = 5
KEYS = ["player_id", "year"]
GK_ONLY_PREFIXES = ("gk_", "goal_kicks", "successful_goal_kicks")
GK_ONLY = {"xg_save", "goals_prevented"}
BIN_EDGES = [0, 375, 625, 750.1]
BIN_LABELS = [250, 500, 750]
REFERENCE = "ridge"      # the method every other method is compared with
N_BOOT = 300             # bootstrap resamples for the paired comparison
METHODS = {**reconstruct_methods.METHODS, **temporal_methods.METHODS}


# ---------------------------------------------------------------- loading the data of one group

def load_truth(gender):
    """(truth table, list of all stat names)."""
    truth = pd.read_parquet(checkpoints.RESULTS_DIR / f"truth_{gender}.parquet")
    all_stems = [c[len("full_"):] for c in truth.columns
                 if c.startswith("full_") and c not in ("full_minutes", "full_matches")]
    return truth, all_stems


def group_stems(all_stems, group, truth):
    """Stats used for one group: GK-only stats only for goalkeepers, and no stat that never varies."""
    stems = []
    for stat in all_stems:
        is_gk_stat = stat.startswith(GK_ONLY_PREFIXES) or stat in GK_ONLY
        if group != "GK" and is_gk_stat:
            continue
        if truth[f"full_{stat}"].std() > 0:
            stems.append(stat)
    return stems


def group_setup(gender, group):
    """(stats, folds) for one group. Each fold is the set of held-out player ids."""
    truth, all_stems = load_truth(gender)
    data = truth[truth["group"] == group].reset_index(drop=True)
    stems = group_stems(all_stems, group, data)

    folds = []
    splitter = GroupKFold(n_splits=N_FOLDS)
    for _, test_rows in splitter.split(data, groups=data["player_id"]):
        folds.append(set(data.iloc[test_rows]["player_id"]))
    return stems, folds


def load_pool(gender, group, stems):
    """All prefixes of the group's player-seasons with their truth, minutes bin and weight."""
    prefixes = pd.read_parquet(checkpoints.RESULTS_DIR / f"prefixes_{gender}.parquet",
                               filters=[("group", "==", group)])
    truth = pd.read_parquet(checkpoints.RESULTS_DIR / f"truth_{gender}.parquet")
    truth_columns = KEYS + ["full_minutes"] + [f"full_{s}" for s in stems]
    pool = prefixes.merge(truth[truth_columns], on=KEYS)

    # minutes bin of every prefix; prefixes beyond 750 minutes are dropped
    pool["bin"] = pd.cut(pool["actual_minutes"], BIN_EDGES, right=False, labels=BIN_LABELS).astype(float)
    pool = pool.dropna(subset=["bin"]).copy()
    pool["bin"] = pool["bin"].astype(int)

    # each player-season counts once per bin
    rows_in_bin = pool.groupby(KEYS + ["bin"])["actual_minutes"].transform("size")
    pool["weight"] = 1.0 / rows_in_bin
    return pool.sort_values(KEYS + ["n_matches"]).reset_index(drop=True)


def oof_predictions(pool, folds, stems, methods):
    """Out-of-fold estimates: {method name: DataFrame with the same index as pool}."""
    estimates = {m: pd.DataFrame(np.nan, index=pool.index, columns=stems) for m in methods}
    for test_players in folds:
        is_test = pool["player_id"].isin(test_players)
        T, E, cache = pool[~is_test], pool[is_test], {}
        assert not set(T["player_id"]) & set(E["player_id"]), "player leaked across folds"
        for m in methods:
            estimates[m].loc[E.index, stems] = METHODS[m](T, E, stems, cache).to_numpy()
    return estimates


# ---------------------------------------------------------------- scoring

def weighted_mean(x, w):
    """Weighted mean of a 1-D array, or of each column of a 2-D array."""
    if x.ndim == 2:
        return (x * w[:, None]).sum(axis=0) / w.sum()
    return (x * w).sum() / w.sum()


def weighted_correlation(a, b, w):
    """Weighted Pearson correlation of the columns of a and b (used on ranks, so it is a Spearman correlation)."""
    da = a - weighted_mean(a, w)
    db = b - weighted_mean(b, w)
    covariance = (da * db * w[:, None]).sum(axis=0)
    variance_a = (da ** 2 * w[:, None]).sum(axis=0)
    variance_b = (db ** 2 * w[:, None]).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return covariance / np.sqrt(variance_a * variance_b)


def score_volumes(truth_p90, estimates, weights, stems, info):
    """One result row per method and volume stat for one minutes bin. `info` holds the columns shared by all rows."""
    rows = []
    truth = truth_p90.to_numpy()
    truth_sd = np.sqrt(weighted_mean((truth - weighted_mean(truth, weights)) ** 2, weights))
    truth_rank = truth_p90.rank().to_numpy()

    for method, estimate in estimates.items():
        error = estimate.to_numpy() - truth
        rmse = np.sqrt(weighted_mean(error ** 2, weights))
        spearman = weighted_correlation(estimate.rank().to_numpy(), truth_rank, weights)
        for i, stat in enumerate(stems):
            rows.append(dict(info, method=method, stat=stat, stat_type="volume",
                             n_truth_ok=info["n"], n_est_missing=0, n_compared=info["n"],
                             rmse=rmse[i], nrmse=rmse[i] / truth_sd[i], spearman=spearman[i]))
    return rows


def score_rates(truth_rates, estimates, specs, minutes, weights, info):
    """One result row per method and rebuilt rate for one minutes bin. Only rows where both rates exist are compared."""
    rows = []
    for method, estimate in estimates.items():
        estimated_rates = rate_rebuild.rebuild_rates(estimate, specs, minutes)
        for spec in specs:
            x = estimated_rates[spec["name"]]    # rate rebuilt from the estimate
            y = truth_rates[spec["name"]]        # rate rebuilt from the truth
            both_exist = (x.notna() & y.notna()).to_numpy()

            if both_exist.any():
                w = weights[both_exist]
                difference = (x[both_exist] - y[both_exist]).to_numpy()
                truth_values = y[both_exist].to_numpy()
                truth_sd = np.sqrt(weighted_mean((truth_values - weighted_mean(truth_values, w)) ** 2, w))
                rmse = float(np.sqrt(weighted_mean(difference ** 2, w)))
                if both_exist.sum() > 2:
                    x_rank = x[both_exist].rank().to_numpy()[:, None]
                    y_rank = y[both_exist].rank().to_numpy()[:, None]
                    spearman = weighted_correlation(x_rank, y_rank, w)[0]
                else:
                    spearman = np.nan
            else:
                rmse, truth_sd, spearman = np.nan, np.nan, np.nan

            nrmse = rmse / truth_sd if truth_sd and truth_sd > 0 else np.nan
            rows.append(dict(info, method=method, stat=spec["name"], stat_type=f"rate_{spec['source']}",
                             n_truth_ok=int(y.notna().sum()), n_est_missing=int((y.notna() & x.isna()).sum()),
                             n_compared=int(both_exist.sum()), rmse=rmse, nrmse=nrmse, spearman=spearman))
    return rows


def score(pool, stems, estimates, specs, gender, group):
    """All result rows (volumes and rates) for one gender and group, for every minutes bin."""
    rows = []
    truth_p90 = get_truth_p90(pool, stems)
    specs = [s for s in specs if rate_rebuild.stats_used(s["num"], s["den"]) <= set(stems)]
    truth_rates = rate_rebuild.rebuild_rates(truth_p90, specs, pool["full_minutes"])

    for b in BIN_LABELS:
        in_bin = (pool["bin"] == b).to_numpy()
        weights = pool["weight"].to_numpy()[in_bin]
        info = dict(gender=gender, group=group, checkpoint=b, n=int(in_bin.sum()), n_seasons=int(round(weights.sum())))
        estimates_in_bin = {method: e[in_bin] for method, e in estimates.items()}

        rows += score_volumes(truth_p90[in_bin], estimates_in_bin, weights, stems, info)
        rows += score_rates(truth_rates[in_bin], estimates_in_bin, specs, pool["full_minutes"][in_bin], weights, info)
    return rows


def get_truth_p90(pool, stems):
    """The full-season per-90 columns of the pool under the plain stat names."""
    return pool[[f"full_{s}" for s in stems]].set_axis(stems, axis=1)


def paired_vs_reference(pool, stems, estimates, gender, group, seed=0):
    """For each method and bin: mean paired difference in scaled squared error vs REFERENCE, with a bootstrap interval."""
    rng = np.random.default_rng(seed)
    rows = []
    truth_all = pool[[f"full_{s}" for s in stems]].to_numpy()

    for b in BIN_LABELS:
        in_bin = (pool["bin"] == b).to_numpy()
        weights, truth = pool["weight"].to_numpy()[in_bin], truth_all[in_bin]
        mean = (truth * weights[:, None]).sum(0) / weights.sum()
        variance = ((truth - mean) ** 2 * weights[:, None]).sum(0) / weights.sum()
        has_variance = variance > 0
        safe_variance = np.where(has_variance, variance, 1)

        reference_error = ((estimates[REFERENCE][in_bin].to_numpy() - truth) ** 2 / safe_variance)[:, has_variance].mean(1)
        season_labels = pool.loc[in_bin, KEYS].astype(str).agg("_".join, axis=1).to_numpy()

        for method, estimate in estimates.items():
            if method == REFERENCE:
                continue
            error = ((estimate[in_bin].to_numpy() - truth) ** 2 / safe_variance)[:, has_variance].mean(1)
            # one number per player-season: the weights of a season's rows sum to 1
            per_season = pd.Series((error - reference_error) * weights).groupby(season_labels).sum().to_numpy()
            resampled_means = [per_season[rng.integers(0, len(per_season), len(per_season))].mean()
                               for _ in range(N_BOOT)]
            rows.append(dict(gender=gender, group=group, checkpoint=b, reference=REFERENCE, method=method,
                             n_seasons=len(per_season), mean_diff=per_season.mean(),
                             lo=np.quantile(resampled_means, 0.025), hi=np.quantile(resampled_means, 0.975)))
    return rows


# ---------------------------------------------------------------- run everything

def run_one(gender, group, methods, specs):
    """Out-of-fold predictions and scores for one gender and group. Returns (scores, paired comparison)."""
    warnings.filterwarnings("ignore")
    start_time = time.time()

    stems, folds = group_setup(gender, group)
    pool = load_pool(gender, group, stems)
    estimates = oof_predictions(pool, folds, stems, methods)

    scores = pd.DataFrame(score(pool, stems, estimates, specs, gender, group))
    if REFERENCE in estimates:
        paired = pd.DataFrame(paired_vs_reference(pool, stems, estimates, gender, group))
    else:
        paired = pd.DataFrame()

    print(f"{gender}/{group}: {len(pool)} prefix rows, {len(stems)} stats, {time.time() - start_time:.0f}s", flush=True)
    return scores, paired


def run_evaluation(groups=None, genders=("female", "male"), methods=None, n_jobs=4, tag="results"):
    """Run the evaluation for every gender/group combination and save <tag>.parquet and <tag>_paired.parquet.

    groups   position groups to run (default: all 7).      methods  method names (default: all 11).
    n_jobs   number of gender/group sets run in parallel.  tag      name of the output files in results/.
    Returns (scores, paired).
    """
    groups = list(common.GROUPS) if groups is None else list(groups)
    methods = list(METHODS) if methods is None else list(methods)

    # the rebuildable rates (needs the database for the API percent definitions)
    con = common.connect()
    _, all_stems = load_truth("female")
    specs, _ = rate_rebuild.load_specs(con, set(all_stems))

    jobs = [(gender, group) for gender in genders for group in groups]
    results = Parallel(n_jobs=n_jobs)(delayed(run_one)(gender, group, methods, specs) for gender, group in jobs)

    scores = pd.concat([r[0] for r in results], ignore_index=True)
    paired = pd.concat([r[1] for r in results], ignore_index=True)
    scores.to_parquet(checkpoints.RESULTS_DIR / f"{tag}.parquet")
    paired.to_parquet(checkpoints.RESULTS_DIR / f"{tag}_paired.parquet")
    print("saved", tag, scores.shape)
    return scores, paired
