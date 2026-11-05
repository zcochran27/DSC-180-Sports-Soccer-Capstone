"""Tests on made-up matches where we know each team's true strength.

Run with:  python -m pytest ratings
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ratings.base import OutcomeMapper, walk_forward
from ratings.models.batch import DixonColes, RidgeRating
from ratings.models.elo import Elo
from ratings.models.glicko import Glicko
from ratings.models.pi import PiRatings


def make_fake_matches(n_teams=24, seasons=3, seed=0):
    """Play a random schedule where team `t` has true strength skill[t]. Returns (matches, skill)."""
    rng = np.random.default_rng(seed)
    skill = rng.normal(0, 0.45, n_teams)

    rows = []
    match_id = 0
    for year in range(2020, 2020 + seasons):
        season_start = pd.Timestamp(f"{year}-08-20")
        for week in range(14):
            order = rng.permutation(n_teams)
            for home, away in zip(order[::2], order[1::2]):
                # Stronger teams score more and concede less; the home team gets a small boost.
                home_rate = np.exp(0.5 + skill[home] - 0.5 * skill[away])
                away_rate = np.exp(0.25 + skill[away] - 0.5 * skill[home])
                rows.append((match_id, year, season_start + pd.Timedelta(days=7 * week), home, away,
                             rng.poisson(home_rate), rng.poisson(away_rate), np.nan, np.nan))
                match_id += 1

    columns = ["match_id", "year", "date", "home", "away", "hg", "ag", "hxg", "axg"]
    return pd.DataFrame(rows, columns=columns), skill


class FakeMatch:
    """Just enough of a match to ask a model for a gap."""
    def __init__(self, date, home, away):
        self.date, self.home, self.away = date, home, away


def correlation_with_true_skill(model, matches, skill):
    """Run the model, then see how closely its team ratings follow the true skill."""
    walk_forward(model, matches)
    teams = range(len(skill))

    if hasattr(model, "r"):  # Elo and Glicko store one rating per team
        estimated = [model.r[team] for team in teams]
    elif hasattr(model, "home_rating"):  # Pi-ratings store a home and an away rating
        estimated = [model.home_rating[team] + model.away_rating[team] for team in teams]
    else:  # batch models: ask for the gap against a fixed opponent (team 0)
        last_date = matches["date"].max()
        estimated = [model.gap(FakeMatch(last_date, team, 0)) for team in teams]

    return np.corrcoef(estimated, skill)[0, 1]


def test_models_recover_strength():
    matches, skill = make_fake_matches()
    models = [Elo(), Elo(mov="wfe"), PiRatings(), Glicko(),
              RidgeRating(min_matches=50), DixonColes(min_matches=50)]
    for model in models:
        assert correlation_with_true_skill(model, matches, skill) > 0.6, model.name


def test_first_gap_uses_only_home_advantage():
    """Before any match is played, the gap should just be the home advantage (no look-ahead)."""
    matches, _ = make_fake_matches(seasons=1)
    predictions = walk_forward(Elo(), matches)
    assert predictions["gap"].iloc[0] == Elo().hfa


def test_probabilities_sum_to_one_and_rise_with_gap():
    matches, _ = make_fake_matches(seasons=2)
    predictions = walk_forward(Elo(), matches)
    mapper = OutcomeMapper().fit(predictions["gap"], predictions["hg"], predictions["ag"])

    probs = mapper.probs(np.array([-200.0, 0.0, 200.0]))
    assert np.allclose(probs.sum(axis=1), 1)
    assert probs[0, 0] < probs[1, 0] < probs[2, 0]  # P(home win) rises with the gap
    assert mapper.slope > 0
