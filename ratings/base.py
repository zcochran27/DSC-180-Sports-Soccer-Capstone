"""Shared pieces for every rating model: the model interface, the walk-forward runner,
the mapping from rating gap to match outcomes, and the scoring metrics.

Every model has two methods:
    gap(match)    -> home strength minus away strength before the match (home advantage included)
    update(match) -> learn from the result of the match

Because all models produce a "gap", we can turn any of them into win/draw/loss probabilities
with the same OutcomeMapper and score them all the same way.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit


class RatingModel:
    name = "base"

    def start_season(self, year: int) -> None:
        """Called when the first match of a new season is reached. Most models regress ratings here."""

    def gap(self, match) -> float:
        """`match` is one row from matches.itertuples()."""
        raise NotImplementedError

    def update(self, match) -> None:
        raise NotImplementedError


def walk_forward(model: RatingModel, matches: pd.DataFrame) -> pd.DataFrame:
    """Go through matches in date order, record the model's gap *before* each match, then update.

    Recording the gap before updating means the model never sees a result it is predicting.
    """
    gaps = []
    last_year = None
    for match in matches.itertuples(index=False):
        if match.year != last_year:
            model.start_season(match.year)
            last_year = match.year
        gaps.append(model.gap(match))
        model.update(match)

    out = matches[["match_id", "year", "date", "home", "away", "hg", "ag", "hxg", "axg"]].copy()
    out["gap"] = gaps
    return out


@dataclass
class OutcomeMapper:
    """Turns a rating gap into outcome probabilities and an expected goal difference.

    Probabilities use an ordered logit: a larger gap makes a home win more likely, and a draw
    happens when the latent result falls inside a band of half-width `cut` around zero.
    The expected goal difference is a straight line: slope * gap + intercept.
    """
    beta: float = 1.0
    cut: float = 0.5
    slope: float = 0.0
    intercept: float = 0.0

    def fit(self, gap, home_goals, away_goals) -> "OutcomeMapper":
        gap = np.asarray(gap, float)
        goal_diff = np.asarray(home_goals, float) - np.asarray(away_goals, float)
        result = np.sign(goal_diff)  # 1 = home win, 0 = draw, -1 = away win

        # Work with a standardised gap so the optimiser starts from sensible numbers.
        spread = gap.std() or 1.0
        z = gap / spread

        def negative_log_likelihood(params):
            beta, cut = params[0], np.exp(params[1])  # exp keeps the cut positive
            p_home = expit(beta * z - cut)
            p_away = expit(-beta * z - cut)
            p_draw = np.clip(1 - p_home - p_away, 1e-9, 1)
            p_actual = np.where(result > 0, p_home, np.where(result < 0, p_away, p_draw))
            return -np.log(np.clip(p_actual, 1e-12, 1)).sum()

        best = minimize(negative_log_likelihood, [1.0, np.log(0.5)], method="Nelder-Mead")
        self.beta = best.x[0] / spread
        self.cut = float(np.exp(best.x[1]))

        # Expected goal difference: least-squares line through (gap, goal difference).
        design = np.c_[gap, np.ones_like(gap)]
        self.slope, self.intercept = np.linalg.lstsq(design, goal_diff, rcond=None)[0]
        return self

    def probs(self, gap) -> np.ndarray:
        """One row per gap: [P(home win), P(draw), P(away win)]."""
        gap = np.asarray(gap, float)
        p_home = expit(self.beta * gap - self.cut)
        p_away = expit(-self.beta * gap - self.cut)
        p_draw = np.clip(1 - p_home - p_away, 0, 1)
        return np.c_[p_home, p_draw, p_away]

    def exp_gd(self, gap) -> np.ndarray:
        """Expected goal difference (home minus away)."""
        return self.slope * np.asarray(gap, float) + self.intercept


def score(predictions: pd.DataFrame, mapper: OutcomeMapper) -> dict:
    """Score predictions (a frame from walk_forward). Lower is better for everything except accuracy."""
    probs = mapper.probs(predictions["gap"].values)
    goal_diff = (predictions["hg"] - predictions["ag"]).values

    # Actual result as a column index: 0 = home win, 1 = draw, 2 = away win.
    actual = np.where(goal_diff > 0, 0, np.where(goal_diff == 0, 1, 2))
    actual_onehot = np.eye(3)[actual]

    log_loss = -np.log(np.clip(probs[np.arange(len(actual)), actual], 1e-12, 1)).mean()
    brier = ((probs - actual_onehot) ** 2).sum(axis=1).mean()
    # Ranked probability score: compares cumulative probabilities (home, home+draw).
    cumulative_gap = np.cumsum(probs, axis=1) - np.cumsum(actual_onehot, axis=1)
    rps = (cumulative_gap[:, :2] ** 2).sum(axis=1).mean() / 2

    expected_gd = mapper.exp_gd(predictions["gap"].values)
    results = {
        "n": len(predictions),
        "log_loss": log_loss,
        "rps": rps,
        "brier": brier,
        "accuracy": (probs.argmax(axis=1) == actual).mean(),
        "gd_mae": np.abs(expected_gd - goal_diff).mean(),
        "gd_rmse": np.sqrt(((expected_gd - goal_diff) ** 2).mean()),
    }

    # Same comparison against xG difference, only for matches that have xG.
    xg_diff = (predictions["hxg"] - predictions["axg"]).values
    has_xg = ~np.isnan(xg_diff)
    results["xgd_mae"] = np.abs(expected_gd[has_xg] - xg_diff[has_xg]).mean() if has_xg.any() else np.nan
    return results
