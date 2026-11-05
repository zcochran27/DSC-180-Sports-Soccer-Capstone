"""Synthetic-data sanity tests: models should recover known team strengths. Run: py -m pytest ratings"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ratings.base import OutcomeMapper, walk_forward
from ratings.models.batch import DixonColes, RidgeRating
from ratings.models.elo import Elo
from ratings.models.glicko import Glicko
from ratings.models.pi import PiRatings


def synth(n_teams=24, seasons=3, seed=0):
    rng = np.random.default_rng(seed)
    skill = rng.normal(0, 0.45, n_teams)  # log goal-rate strength
    rows, mid = [], 0
    for yr in range(2020, 2020 + seasons):
        day = pd.Timestamp(f"{yr}-08-20")
        for rnd in range(14):
            order = rng.permutation(n_teams)
            for a, b in zip(order[::2], order[1::2]):
                lh = np.exp(0.2 + 0.25 + skill[a] - skill[b] * 0.0 - 0.0)
                lh, la = np.exp(0.25 + 0.25 + skill[a] - 0.5 * skill[b]), np.exp(0.25 + skill[b] - 0.5 * skill[a])
                rows.append((mid, yr, day + pd.Timedelta(days=7 * rnd), a, b,
                             rng.poisson(lh), rng.poisson(la), np.nan, np.nan))
                mid += 1
    df = pd.DataFrame(rows, columns=["match_id", "year", "date", "home", "away", "hg", "ag", "hxg", "axg"])
    return df, skill


def _corr(model, df, skill):
    walk_forward(model, df)
    if hasattr(model, "r"):
        est = np.array([model.r[t] for t in range(len(skill))])
    elif hasattr(model, "rh"):
        est = np.array([model.rh[t] + model.ra[t] for t in range(len(skill))])
    else:  # batch: probe gaps vs a fixed opponent
        class M:  # minimal row
            pass
        est = []
        for t in range(len(skill)):
            m = M()
            m.date, m.home, m.away = df["date"].max(), t, 0
            est.append(model.gap(m))
        est = np.array(est)
    return np.corrcoef(est, skill)[0, 1]


def test_models_recover_strength():
    df, skill = synth()
    for model in [Elo(), Elo(mov="wfe"), PiRatings(), Glicko(), RidgeRating(min_matches=50),
                  DixonColes(min_matches=50)]:
        assert _corr(model, df, skill) > 0.6, model.name


def test_no_lookahead_first_gap_is_neutral():
    df, _ = synth(seasons=1)
    out = walk_forward(Elo(), df)
    assert out["gap"].iloc[0] == Elo().hfa


def test_mapper_probabilities_sum_to_one_and_are_monotone():
    df, _ = synth(seasons=2)
    out = walk_forward(Elo(), df)
    mp = OutcomeMapper().fit(out["gap"], out["hg"], out["ag"])
    p = mp.probs(np.array([-200.0, 0.0, 200.0]))
    assert np.allclose(p.sum(1), 1)
    assert p[0, 0] < p[1, 0] < p[2, 0]
    assert mp.slope > 0
