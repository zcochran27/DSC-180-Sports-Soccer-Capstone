"""Method 3: prediction from similar players ("what would they look like with more minutes?").

Idea: players who played a lot give us the answer key. For each of them we know their full-season
per-90 rate. We can also replay their season as if they had only played, say, 200 minutes: pick
a random handful of their matches and compute the stats from just those. That gives many
(partial-season stats -> full-season stats) examples for every minute level.

For a real low-minute player we find the simulated examples that look most like them (same
amount of minutes, similar stats) and average what those examples turned into. That average is
our estimate of the player's rate had they played a full season.

Steps:
    1. simulate_low_minutes  - build the training examples from high-minute players
    2. fit_models            - one nearest-neighbour model per minute bin
    3. predict               - estimate the full-season rates of low-minute players
    cross_fitted_predict wraps all three so no player ever helps predict themselves.

Run a quick self-check (needs .env with MotherDuck credentials):
    python -m low_minute_player_analysis.similar_player_prediction
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from . import common

FULL_SEASON_MINUTES = 900  # players with at least this many minutes are the "answer key"
SIM_LEVELS = [75, 150, 225, 300, 400, 500, 650, 800]  # minutes to replay each season down to
BIN_EDGES = [0, 125, 250, 375, 500, 700, FULL_SEASON_MINUTES]  # one model per bin of minutes
N_DRAWS = 8  # random replays per player-season per level
N_NEIGHBORS = 30


# ── 1. Build the training examples ────────────────────────────────────────────

def simulate_low_minutes(matches: pd.DataFrame, role: str, levels: list[int] = SIM_LEVELS,
                         n_draws: int = N_DRAWS, seed: int = 0) -> pd.DataFrame:
    """Replay high-minute player-seasons with only a random subset of their matches.

    Returns one row per replay with: player_id, year, minutes (of the replay),
    part_<stat> (per-90 from the replayed matches) and full_<stat> (per-90 over the whole season).
    """
    rng = np.random.default_rng(seed)
    stats = common.STAT_COLS[role]
    columns = [f"total_{s}" for s in stats]
    rows = []

    for (player_id, year), season in matches.groupby(["player_id", "year"]):
        season_minutes = season["minutes"].to_numpy()
        if season_minutes.sum() < FULL_SEASON_MINUTES:
            continue
        season_totals = season[columns].to_numpy()
        full_rate = season_totals.sum(axis=0) / season_minutes.sum() * 90

        for level in levels:
            for _ in range(n_draws):
                order = rng.permutation(len(season))  # a random order to play the matches in
                played = np.cumsum(season_minutes[order])
                last = np.searchsorted(played, level)  # first match that reaches the target
                chosen = order[: last + 1]
                minutes = season_minutes[chosen].sum()
                part_rate = season_totals[chosen].sum(axis=0) / minutes * 90
                rows.append([player_id, year, minutes, *part_rate, *full_rate])

    names = ["player_id", "year", "minutes"] + [f"part_{s}" for s in stats] + [f"full_{s}" for s in stats]
    return pd.DataFrame(rows, columns=names)


# ── 2. Fit one model per minute bin ───────────────────────────────────────────

def fit_models(sim: pd.DataFrame, role: str) -> dict:
    """Fit a nearest-neighbour model for each minute bin.

    Models are separate per bin so a 150-minute player is only compared with other
    ~150-minute replays (the amount of luck in the numbers is similar).
    """
    stats = common.STAT_COLS[role]
    models = {}
    bins = pd.cut(sim["minutes"], BIN_EDGES, right=False)
    for bin_, rows in sim.groupby(bins, observed=True):
        features = rows[[f"part_{s}" for s in stats]].to_numpy()
        scaler = StandardScaler().fit(features)
        knn = KNeighborsRegressor(n_neighbors=N_NEIGHBORS)
        knn.fit(scaler.transform(features), rows[[f"full_{s}" for s in stats]].to_numpy())
        models[bin_] = (scaler, knn)
    return models


# ── 3. Predict ────────────────────────────────────────────────────────────────

def predict(models: dict, rates: pd.DataFrame, minutes: pd.Series) -> pd.DataFrame:
    """Estimate full-season per-90 for players below FULL_SEASON_MINUTES; keep the rest as is."""
    out = rates.copy()
    bins = pd.cut(minutes, BIN_EDGES, right=False)
    for bin_, (scaler, knn) in models.items():
        rows = rates.index[bins == bin_]
        if len(rows):
            out.loc[rows] = knn.predict(scaler.transform(rates.loc[rows].to_numpy()))
    return out


def cross_fitted_predict(sim: pd.DataFrame, rates: pd.DataFrame, minutes: pd.Series,
                         player_ids: pd.Series, role: str, n_folds: int = 5) -> pd.DataFrame:
    """Predict every player using models trained only on *other* players.

    Players are split into folds. For each fold the models are trained on replays from the
    players outside that fold, so a player's own seasons never help predict them.
    """
    out = rates.copy()
    for _, test in GroupKFold(n_splits=n_folds).split(rates, groups=player_ids):
        test_rows = rates.index[test]
        train = sim[~sim["player_id"].isin(player_ids.iloc[test])]
        models = fit_models(train, role)
        out.loc[test_rows] = predict(models, rates.loc[test_rows], minutes.loc[test_rows])
    return out


# ── Quick self-check ──────────────────────────────────────────────────────────

def main() -> None:
    role = "FW"
    con = common.connect()
    terms = common.load_player_terms(con, gender="female", role=role)
    matches = common.load_player_matches(con, gender="female", role=role)
    pairs = common.make_next_year_pairs(terms, role)
    raw, target, minutes = common.split_pairs(pairs, role)

    sim = simulate_low_minutes(matches, role)
    print(f"{len(sim):,} simulated examples from {sim['player_id'].nunique():,} players\n")

    estimate = cross_fitted_predict(sim, raw, minutes, pairs["player_id"], role)
    print("Raw per-90:\n", common.evaluate(raw, target, minutes).round(3), "\n")
    print("Similar-player prediction:\n", common.evaluate(estimate, target, minutes).round(3))


if __name__ == "__main__":
    main()
