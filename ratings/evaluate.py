"""Compare all the rating models on held-out matches.

Each model goes through the matches in date order, predicting each match before it learns the
result. The gap -> outcome mapping is fit on the training seasons (the first season is skipped
as burn-in) and every model is scored on the last `test_seasons` seasons. The table also
includes a constant "home rate" baseline and an ensemble that combines all the models.

From a notebook:
    matches = load_matches("male")
    models = default_models("male")
    table = evaluate(matches, models)
"""
import numpy as np
import pandas as pd

from ratings import base
from ratings.models import constant
from ratings.specs import SPECS, build
from ratings.tune import load_best

BASELINE = "baseline_home_rate"


def default_models(gender=None, tuned=True):
    """The baseline plus every model in SPECS, using the tuned parameters when available."""
    best = load_best(gender) if (tuned and gender) else {}
    models = [constant.new_model(BASELINE)]
    for name in SPECS:
        models.append(build(name, best.get(name, {}).get("params")))
    return models


def split_years(matches, test_seasons):
    """Returns (training years, test years). The first season is dropped as burn-in."""
    years = sorted(matches["year"].unique())
    if len(years) < 3:
        raise ValueError(f"Need at least 3 seasons, found {years}")
    test_years = years[-test_seasons:]
    train_years = [y for y in years[1:] if y not in test_years]
    return train_years, test_years


def make_ensemble(predictions, train_years):
    """Combine the models' gaps with a linear regression on goal difference.

    Each model's gap is standardised first. The weights are fit on training seasons only and are
    saved in the result's `.attrs["coef"]`.
    """
    names = [name for name in predictions if name != BASELINE]
    gaps = np.c_[[predictions[name]["gap"].values for name in names]].T   # one column per model

    first = next(iter(predictions.values()))
    goal_diff = (first["hg"] - first["ag"]).values
    is_train = first["year"].isin(train_years).values

    z_gaps = (gaps - gaps[is_train].mean(axis=0)) / (gaps[is_train].std(axis=0) + 1e-9)
    design = np.c_[z_gaps, np.ones(len(z_gaps))]
    coef = np.linalg.lstsq(design[is_train], goal_diff[is_train], rcond=None)[0]

    ensemble = first.copy()
    ensemble["gap"] = design @ coef
    ensemble.attrs["coef"] = dict(zip(names, coef[:-1]))
    return ensemble


def evaluate(matches, models=None, test_seasons=1, return_preds=False):
    """Score every model on the test seasons.

    Returns a table (one row per model, best log loss first). With return_preds=True it also
    returns the predictions, the fitted outcome mappers and the test years.
    """
    models = models or default_models()
    train_years, test_years = split_years(matches, test_seasons)

    predictions = {model["name"]: base.walk_forward(model, matches) for model in models}
    predictions["ensemble_ols"] = make_ensemble(predictions, train_years)

    rows = []
    mappers = {}
    for name, df in predictions.items():
        train = df[df["year"].isin(train_years)]
        test = df[df["year"].isin(test_years)]
        mappers[name] = base.fit_outcome_mapper(train["gap"], train["hg"], train["ag"])
        rows.append({"model": name, **base.score(test, mappers[name])})
    table = pd.DataFrame(rows).set_index("model").sort_values("log_loss")

    if return_preds:
        return table, predictions, mappers, test_years
    return table
