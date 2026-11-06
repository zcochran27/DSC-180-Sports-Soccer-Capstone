"""Offline checks for learned adjustments and chronological evaluation."""
import numpy as np
import pandas as pd
import pytest

from . import methods as m
from . import normalize as n


@pytest.fixture(scope="module")
def history():
    rng = np.random.default_rng(180)
    rows = []
    for year in [2023, 2024, 2025]:
        for match in range(90):
            home, away = match % 8 + 1, (match + 3) % 8 + 1
            p = float(rng.uniform(20, 80))
            hg, ag = rng.poisson(1.5, size=2)
            for team, opponent, is_home, share, goals, against in [
                (home, away, True, p, hg, ag), (away, home, False, 100-p, ag, hg)
            ]:
                row = dict(match_id=year * 1000 + match, year=year, date=f"{year}-08-01",
                           gender="female", term="Fall", team_id=team, opponent_id=opponent,
                           team_name=f"Team {team}", opponent_name=f"Team {opponent}",
                           is_home=is_home, competition_id=1, match_duration="Regular",
                           team_goals=goals, opponent_goals=against, stats_available=True,
                           opponent_stats_available=True, reported_possession_pct=share,
                           opponent_reported_possession_pct=100-share)
                row["date"] = str(pd.Timestamp(row["date"]) + pd.Timedelta(days=match))
                for stat, base in zip(n.STATS, [20, 45, 10]):
                    row[f"total_{stat}"] = rng.poisson(base * ((1-share/100)/.5)**.35)
                rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def result(history):
    return m.compare_methods(history, [2023, 2024], 2025)


def test_power_recovers_partial_exposure_and_keeps_zeros():
    rng = np.random.default_rng(81)
    p = pd.Series(rng.uniform(10, 90, 20000))
    y = pd.Series(rng.poisson(6 * ((1-p/100)/.5)**.35))
    fit = m.fit_power(p, y)
    assert fit.alpha == pytest.approx(.35, abs=.03)
    assert fit.mean_at_50 == pytest.approx(6, abs=.1)
    assert fit.n_train == len(y) and (y == 0).any()
    assert fit.predict(pd.Series([50]))[0] == pytest.approx(fit.mean_at_50)


def test_power_boundaries_and_invalid_inputs():
    p = pd.Series(np.linspace(10, 90, 100))
    assert m.fit_power(p, pd.Series(np.repeat(10, 100))).alpha == 0
    y = pd.Series(np.round(100 * ((1-p/100)/.5)**2))
    assert m.fit_power(p, y).alpha == 1
    for bad_p, bad_y in [(p, pd.Series(np.zeros(100))), (p*0+50, y),
                         (p, y*0-1), (p, y+.1), (p.where(p.index != 0), y)]:
        with pytest.raises(ValueError):
            m.fit_power(bad_p, bad_y)


def test_elo_is_before_results_and_ignores_future_scores(history):
    rated = m.add_prematch_elo(history)
    changed = history.copy()
    first_match = changed.loc[changed.year.eq(2025), "match_id"].min()
    idx = changed.match_id.eq(first_match)
    changed.loc[idx, "team_goals"] = np.where(changed.loc[idx, "is_home"], 9, 0)
    changed.loc[idx, "opponent_goals"] = np.where(changed.loc[idx, "is_home"], 0, 9)
    rerated = m.add_prematch_elo(changed)
    before = history.match_id.le(first_match)
    np.testing.assert_allclose(rated.loc[before, ["team_elo", "opponent_elo"]],
                               rerated.loc[before, ["team_elo", "opponent_elo"]])
    assert (rated.loc[history.match_id.eq(history.match_id.min()), "team_elo"] == 1500).all()
    # Supplied database ratings cannot change this experiment's fixed Elo features.
    injected = history.assign(team_elo=99999, opponent_elo=-99999)
    np.testing.assert_allclose(m.add_prematch_elo(injected)[["team_elo", "opponent_elo"]],
                               rated[["team_elo", "opponent_elo"]])


def test_elo_rejects_incomplete_or_inconsistent_match_pairs(history):
    with pytest.raises(ValueError, match="home and away"):
        m.add_prematch_elo(history.iloc[1:])
    bad = history.copy()
    bad.loc[0, "team_goals"] = 999
    with pytest.raises(ValueError, match="disagrees"):
        m.add_prematch_elo(bad)


def test_simultaneous_matches_use_prior_ratings(history):
    same_time = history.loc[history.year.eq(2023)].iloc[:4].copy()
    same_time["date"] = "2023-08-01"
    # The second match shares a team with the first; neither can see its result.
    home = same_time["is_home"].eq(True)
    second = same_time["match_id"].eq(same_time["match_id"].max())
    same_time.loc[second & home, "team_id"] = 1
    same_time.loc[second & ~home, "opponent_id"] = 1
    rated = m.add_prematch_elo(same_time)
    assert rated[["team_elo", "opponent_elo"]].eq(1500).all().all()


def test_multi_year_loading_requires_all_fit_seasons(history, tmp_path):
    sql, params = n.history_query("female", [2023, 2024, 2025])
    assert params == ["female", 2023, 2024, 2025, "Fall"]
    assert sql.count("?") == 5 and "t.year IN (?,?,?)" in sql
    path = tmp_path / "history.csv"
    history.to_csv(path, index=False)
    assert len(n.load_history_csv(path, "female", [2023, 2024, 2025])) == len(history)
    with pytest.raises(ValueError, match="2022"):
        n.load_history_csv(path, "female", [2022, 2025])


def test_split_keeps_matches_together_and_rejects_leakage(history):
    train, test = m.split_history(history, [2023, 2024], 2025)
    assert not set(train.match_id) & set(test.match_id)
    assert train.groupby("match_id").size().eq(2).all()
    for years in [[2025], [], [2023, 2023]]:
        with pytest.raises(ValueError):
            m.split_history(history, years, 2025)
    bad = history.copy()
    bad.loc[bad.year.eq(2024), "date"] = "2026-01-01"
    with pytest.raises(ValueError, match="precede"):
        m.split_history(bad, [2023, 2024], 2025)


def test_adjustments_preserve_raw_counts_and_use_common_cohort(history, result):
    test = result.test
    original = history.loc[history.year.eq(2025)]
    for stat in n.STATS:
        pd.testing.assert_series_equal(test[f"total_{stat}"], original[f"total_{stat}"])
        valid = test[f"common_{stat}"]
        assert np.isfinite(test.loc[valid, f"context50_{stat}"]).all()
        raw, fixed, power = (test.loc[valid, prefix+stat] for prefix in ["total_", "p50_", "power50_"])
        assert (power >= np.minimum(raw, fixed)-1e-8).all()
        assert (power <= np.maximum(raw, fixed)+1e-8).all()
        scores = result.scores.query("stat == @stat")
        assert scores.team_matches.eq(valid.sum()).all()
        assert (scores.poisson_deviance >= 0).all()


def test_test_counts_do_not_influence_fits_or_factors(history, result):
    changed = history.copy()
    for stat in n.STATS:
        changed.loc[changed.year.eq(2025), f"total_{stat}"] += 40
    other = m.compare_methods(changed, [2023, 2024], 2025)
    pd.testing.assert_frame_equal(result.parameters, other.parameters)
    for stat in n.STATS:
        np.testing.assert_allclose(result.context[stat].model.coef_, other.context[stat].model.coef_)
        np.testing.assert_allclose(result.test[f"context_factor_{stat}"], other.test[f"context_factor_{stat}"])


def test_neutral_possession_and_zero_counts(history):
    changed = history.copy()
    idx = changed.year.eq(2025)
    changed.loc[idx, ["reported_possession_pct", "opponent_reported_possession_pct"]] = 50
    for stat in n.STATS:
        changed.loc[idx, f"total_{stat}"] = 0
    result = m.compare_methods(changed, [2023, 2024], 2025)
    for stat in n.STATS:
        assert result.test[f"context_factor_{stat}"].eq(1).all()
        assert result.test[f"power_factor_{stat}"].eq(1).all()
        for prefix in m.PREFIXES.values():
            assert result.test[prefix+stat].eq(0).all()


def test_missing_context_does_not_discard_other_methods(history):
    changed = history.copy()
    idx = changed.index[changed.year.eq(2025)][0]
    changed.loc[idx, "match_duration"] = None
    changed.loc[idx, "total_interceptions"] = -1
    result = m.compare_methods(changed, [2023, 2024], 2025)
    row = result.test.loc[idx]
    assert not row.context_available
    assert np.isnan(row.context50_defensive_duels)
    assert np.isfinite(row.power50_defensive_duels)
    assert np.isnan(row.power50_interceptions)
    assert not row.common_defensive_duels


def test_context_uses_training_preprocessing_and_handles_unseen_team(history, result):
    fit = result.context["interceptions"]
    row = result.test.iloc[[0]].copy()
    row["team_id"] = 987654
    row["possession_pct"] = 50
    mean_before, knots_before = fit.scaler.mean_.copy(), fit.spline.bsplines_[0].t.copy()
    pred = fit.predict(row)
    assert np.isfinite(pred).all() and (pred > 0).all()
    np.testing.assert_array_equal(mean_before, fit.scaler.mean_)
    np.testing.assert_array_equal(knots_before, fit.spline.bsplines_[0].t)
    with pytest.raises(ValueError, match="before prediction"):
        m.ContextFit().predict(row)
