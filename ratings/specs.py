"""The list of rating models: for each name, the class, fixed settings, and the values tried when tuning."""
from __future__ import annotations

from ratings.base import RatingModel
from ratings.models.batch import DixonColes, RidgeRating
from ratings.models.elo import Elo
from ratings.models.glicko import Glicko
from ratings.models.pi import PiRatings

ELO_SPACE = {"k": [10, 20, 30, 40, 60, 80], "hfa": [30, 50, 70, 90], "carry": [0.5, 0.65, 0.8, 0.9, 1.0]}
BATCH_SPACE = {"half_life_days": [180, 365, 730, 1500]}

# name -> (model class, settings that never change, parameter values to search when tuning)
SPECS = {
    "elo_none": (Elo, {"mov": "none"}, ELO_SPACE),
    "elo_sqrt": (Elo, {"mov": "sqrt"}, ELO_SPACE),
    "elo_wfe_autocorr": (Elo, {"mov": "wfe", "autocorr": True}, ELO_SPACE),
    "elo_xg": (Elo, {}, {**ELO_SPACE, "xg_weight": [0.25, 0.5, 0.75, 1.0], "xg_c": [0.6, 1.0, 1.5]}),
    "pi": (PiRatings, {}, {"lam": [0.02, 0.04, 0.06, 0.1], "gamma": [0.2, 0.5, 0.8],
                           "c": [1.5, 3.0, 5.0], "carry": [0.6, 0.8, 0.95]}),
    "glicko": (Glicko, {}, {"hfa": [40, 60, 80], "c_week": [2, 6, 12], "season_c": [60, 120, 180],
                            "carry": [0.6, 0.75, 0.9], "rd_max": [200, 300]}),
    "dixon_coles": (DixonColes, {}, {**BATCH_SPACE, "ridge": [0.5, 2, 8]}),
    "dixon_coles_xg": (DixonColes, {"target": "xg"},
                       {**BATCH_SPACE, "ridge": [0.5, 2, 8], "xg_weight": [0.5, 1.0]}),
    "ridge_gd": (RidgeRating, {}, {**BATCH_SPACE, "ridge": [2, 5, 15, 40]}),
    "ridge_gd_xg": (RidgeRating, {"target": "xg"},
                    {**BATCH_SPACE, "ridge": [2, 5, 15, 40], "xg_weight": [0.5, 1.0]}),
}

# Models that refit in batches (slower, so tuning tries fewer settings for them).
BATCH = {"dixon_coles", "dixon_coles_xg", "ridge_gd", "ridge_gd_xg"}


def build(name: str, params: dict | None = None) -> RatingModel:
    """Create a model by name. `params` are tuned values, e.g. {"k": 60, "hfa": 70}."""
    model_class, fixed_settings, _ = SPECS[name]
    model = model_class(**fixed_settings, **(params or {}))
    model.name = name
    return model
