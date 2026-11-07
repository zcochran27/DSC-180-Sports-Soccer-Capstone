"""Random-search hyperparameter tuning for every rating model.

How a setting is judged (the test season is never touched):
    1. Rate all matches up to and including the validation season, one at a time in date order.
    2. Fit the gap -> outcome mapping on the seasons before the validation season
       (the first season is skipped as burn-in).
    3. Score the validation season by log loss.

The best settings per gender are saved to ratings/best_params.json.

From a notebook:
    tune.run_tuning(["male", "female"])          # all models (takes a while); saves best_params.json
    tune.load_best("male")                       # read the saved settings
"""
import json
import random
from pathlib import Path

import pandas as pd

from ratings import base
from ratings.data import load_matches
from ratings.specs import BATCH, SPECS, build

PARAMS_PATH = Path(__file__).with_name("best_params.json")


def split_seasons(matches, test_seasons):
    """Returns (fit_years, validation_year, matches up to and including the validation year).

    The newest `test_seasons` seasons are held out and never used. The one before them is the
    validation season, and the seasons before that (skipping the first, a burn-in) are used to
    fit the gap-to-outcome mapping.
    """
    years = sorted(matches["year"].unique())
    years_before_test = years[:-test_seasons]
    validation_year = int(years_before_test[-1])
    fit_years = [int(y) for y in years_before_test[1:-1]]
    return fit_years, validation_year, matches[matches["year"] <= validation_year]


def validation_log_loss(name, params, matches, fit_years, validation_year):
    """Log loss on the validation season for one model with one set of parameters."""
    model = build(name, params)
    if name in BATCH:
        model["refit_days"] = 28   # refit less often so the search stays fast
    predictions = base.walk_forward(model, matches)
    fit_rows = predictions[predictions["year"].isin(fit_years)]
    validation_rows = predictions[predictions["year"] == validation_year]
    mapper = base.fit_outcome_mapper(fit_rows["gap"], fit_rows["hg"], fit_rows["ag"])
    return base.score(validation_rows, mapper)["log_loss"]


def sample_params(space, n, rng):
    """Pick up to n different random parameter combinations from the search space."""
    total_combinations = 1
    for values in space.values():
        total_combinations *= len(values)

    seen = set()
    chosen = []
    while len(chosen) < min(n, total_combinations):
        params = {name: rng.choice(values) for name, values in space.items()}
        key = tuple(sorted(params.items()))
        if key not in seen:
            seen.add(key)
            chosen.append(params)
    return chosen


def tune(gender, test_seasons=1, n_online=40, n_batch=8, seed=0, only=None):
    """Random search for every model. Returns {model: {"params": ..., "val_log_loss": ...}}."""
    all_matches = load_matches(gender)
    fit_years, validation_year, matches = split_seasons(all_matches, test_seasons)
    rng = random.Random(seed)
    best = {}
    log = []
    for name, (_, _, space) in SPECS.items():
        if only and name not in only:
            continue
        n_tries = n_batch if name in BATCH else n_online
        results = []
        for params in sample_params(space, n_tries, rng):
            loss = validation_log_loss(name, params, matches, fit_years, validation_year)
            results.append((loss, params))
        results.sort(key=lambda result: result[0])

        best_loss, best_params = results[0]
        best[name] = {"params": best_params, "val_log_loss": best_loss}
        log += [{"gender": gender, "model": name, "val_log_loss": loss, **params} for loss, params in results]
        print(f"[{gender}] {name}: {best_loss:.4f} {best_params}", flush=True)

    pd.DataFrame(log).to_csv(PARAMS_PATH.with_name(f"tuning_log_{gender}.csv"), index=False)
    return best


def load_best(gender):
    """Tuned parameters saved by run_tuning for this gender ({} if tuning has not been run)."""
    if PARAMS_PATH.exists():
        return json.loads(PARAMS_PATH.read_text()).get(gender, {})
    return {}


def run_tuning(genders=("male", "female"), test_seasons=1, n_online=40, n_batch=8, only=None):
    """Tune every model for each gender and save the best settings to best_params.json.

    n_online  settings to try for each Elo / Pi / Glicko model.   n_batch  settings for each batch model.
    only      list of model names to tune (default: all; other models keep their saved settings).
    """
    saved = json.loads(PARAMS_PATH.read_text()) if PARAMS_PATH.exists() else {}
    for gender in genders:
        new_results = tune(gender, test_seasons, n_online, n_batch, only=only)
        saved[gender] = {**saved.get(gender, {}), **new_results}
        PARAMS_PATH.write_text(json.dumps(saved, indent=2))
    return saved
