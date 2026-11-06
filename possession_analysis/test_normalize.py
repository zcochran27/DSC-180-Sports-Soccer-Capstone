"""Offline tests for possession adjustment and same-cohort comparisons.

Run: python -m pytest possession_analysis
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from possession_analysis import normalize as n


def example(own=30.0, opponent=70.0, count=14.0):
    return pd.DataFrame([{
        "match_id": 1, "team_id": 10, "opponent_id": 20, "team_name": "A",
        "stats_available": True, "opponent_stats_available": True,
        "reported_possession_pct": own, "opponent_reported_possession_pct": opponent,
        "passes": 100.0, "opponent_passes": 300.0,
        **{f"total_{s}": count for s in n.STATS},
    }])


def test_known_adjustment_and_raw_preservation():
    raw = example()
    before = raw.copy(deep=True)
    out = n.normalize(raw)
    assert out.loc[0, "p50_interceptions"] == pytest.approx(10)
    assert out.loc[0, "normalization_factor"] == pytest.approx(5 / 7)
    pd.testing.assert_frame_equal(raw, before)
    assert out.loc[0, "total_interceptions"] == 14


def test_neutral_possession_is_identity_and_zero_is_valid():
    out = n.normalize(example(50, 50, 0))
    assert out.loc[0, "normalization_factor"] == 1
    assert out.loc[0, "p50_interceptions"] == 0
    assert out.loc[0, "valid_interceptions"]


def test_numeric_text_counts_and_success_ratios():
    raw = example(70, 30, "14")
    raw["total_defensive_duels_won"] = "7"
    raw["percent_defensive_duels_won"] = 50
    out = n.normalize(raw)
    assert out.loc[0, "p50_defensive_duels_won"] / out.loc[0, "p50_defensive_duels"] == pytest.approx(.5)
    assert out.loc[0, "percent_defensive_duels_won"] == 50
    assert n.by_possession(out).iloc[0].raw_median == 14
    assert n.team_summary(out, min_matches=1).iloc[0].raw_per_match == 14


def test_cached_data_does_not_silently_change_scope(tmp_path):
    raw = example().assign(gender="female", year=2025, term="Fall")
    path = tmp_path / "matches.csv"
    raw.to_csv(path, index=False)
    assert len(n.load_csv(path, "female", 2025)) == 1
    with pytest.raises(ValueError, match="requested"):
        n.load_csv(path, "male", 2025)


def test_high_possession_increases_defensive_equivalent():
    out = n.normalize(example(70, 30, 12))
    assert out.loc[0, "p50_interceptions"] == pytest.approx(20)


@pytest.mark.parametrize("own,opp,status", [
    (np.nan, 70, "missing_possession"), (np.inf, 70, "invalid_possession"),
    (0, 100, "invalid_possession"), (-1, 101, "invalid_possession"),
    (100, 0, "invalid_possession"), (50, 40, "pair_sum_mismatch"),
    (99, 1, "extreme_possession"),
])
def test_invalid_exposure_never_falls_back_to_neutral(own, opp, status):
    out = n.normalize(example(own, opp))
    assert out.loc[0, "normalization_status"] == status
    assert pd.isna(out.loc[0, "p50_interceptions"])
    assert pd.isna(out.loc[0, "normalization_factor"])


def test_rounding_is_rescaled_and_threshold_boundary_is_inclusive():
    out = n.normalize(example(60, 39.9))
    assert out.loc[0, "possession_pct"] == pytest.approx(100 * 60 / 99.9)
    assert out.loc[0, "possession_pct"] + out.loc[0, "opponent_possession_pct"] == pytest.approx(100)
    assert n.normalize(example(95, 5)).loc[0, "normalization_status"] == "ok"


def test_missing_stats_are_retained_in_coverage():
    raw = example()
    raw["stats_available"] = False
    out = n.normalize(raw)
    assert len(out) == 1
    assert out.loc[0, "normalization_status"] == "missing_team_stats"
    assert n.coverage(out).iloc[0].team_rows == 1


def test_invalid_metric_does_not_discard_other_metrics():
    raw = example()
    raw["total_interceptions"] = -1
    raw["total_defensive_duels"] = np.nan
    out = n.normalize(raw)
    assert out.loc[0, "normalization_status"] == "ok"
    assert pd.isna(out.loc[0, "p50_interceptions"])
    assert pd.isna(out.loc[0, "p50_defensive_duels"])
    assert out.loc[0, "valid_defensive_duels_won"]
    result = n.comparison(out).set_index("stat")
    assert result.loc["interceptions", "team_matches"] == 0
    assert result.loc["defensive_duels_won", "team_matches"] == 1


@pytest.mark.parametrize("count", [np.inf, -1, .5, "bad"])
def test_unusable_counts_are_not_zero_filled(count):
    out = n.normalize(example(count=count))
    assert pd.isna(out.loc[0, "p50_interceptions"])


def test_duplicate_and_missing_keys_fail():
    raw = example()
    with pytest.raises(ValueError, match="Duplicate"):
        n.normalize(pd.concat([raw, raw], ignore_index=True))
    raw["team_id"] = np.nan
    with pytest.raises(ValueError, match="identifiers"):
        n.normalize(raw)


def test_pass_share_is_only_used_when_requested():
    raw = example()
    direct = n.normalize(raw)
    proxy = n.normalize(raw, source="pass_share")
    assert direct.loc[0, "possession_pct"] == 30
    assert proxy.loc[0, "possession_pct"] == 25
    raw["reported_possession_pct"] = np.nan
    assert pd.isna(n.normalize(raw).loc[0, "p50_interceptions"])


def test_band_and_team_summaries_use_same_valid_rows():
    a, b = example(30, 70, 14), example(50, 50, 8)
    b["match_id"] = 2
    out = n.normalize(pd.concat([a, b], ignore_index=True))
    summary = n.team_summary(out, min_matches=2).set_index("stat")
    assert summary.loc["interceptions", "raw_per_match"] == 11
    assert summary.loc["interceptions", "p50_per_match"] == 9
    bands = n.by_possession(out)
    assert bands.query("stat == 'interceptions'").team_matches.sum() == 2
    assert n.comparison(out).team_matches.eq(2).all()


def test_empty_input_has_stable_summary_columns():
    out = n.normalize(example().iloc[:0])
    assert n.comparison(out).team_matches.eq(0).all()
    assert n.by_possession(out).empty
    assert n.team_summary(out).empty


def test_query_binds_scope_values_and_restricts_status():
    sql, params = n.match_query("female", 2025)
    assert params == ["female", 2025, "Fall"]
    assert sql.count("?") == 3
    assert "m.status='Played'" in sql
    with pytest.raises(ValueError):
        n.match_query("unknown", 2025)
    with pytest.raises(ValueError):
        n.match_query("female", True)


def test_query_and_loader_on_local_database():
    duckdb = pytest.importorskip("duckdb")
    with duckdb.connect() as con:
        con.execute("ATTACH ':memory:' AS wyscout_dev")
        con.execute("CREATE SCHEMA wyscout_dev.wyscout")
        tables = {
            "matches": pd.DataFrame([{"match_id": 1, "date_utc": pd.Timestamp("2025-09-01"),
                "home_team_id": 10, "away_team_id": 20, "duration": "Regular", "season_id": 3, "status": "Played", "home_score": 1, "away_score": 0},
                {"match_id": 2, "date_utc": pd.Timestamp("2025-09-02"), "home_team_id": 10,
                 "away_team_id": 20, "duration": "Regular", "season_id": 3, "status": "Played", "home_score": 1, "away_score": 0}]),
            "seasons": pd.DataFrame([{"season_id": 3, "comp_id": 4}]),
            "competitions": pd.DataFrame([{"comp_id": 4, "gender": "female"}]),
            "season_terms": pd.DataFrame([{"season_id": 3, "year": 2025, "term": "Fall", "kind": "regular"}]),
            "teams": pd.DataFrame([{"team_id": 10, "name": "A"}, {"team_id": 20, "name": "B"}]),
            "team_match_stats": pd.DataFrame([{"match_id": 1, "team_id": tid,
                "average_possession_percent": possession, "total_passes": 100,
                **{f"total_{s}": 14 for s in n.STATS}} for tid, possession in [(10, 30), (20, 70)]]),
        }
        for name, frame in tables.items():
            con.register("fixture", frame)
            con.execute(f"CREATE TABLE wyscout_dev.wyscout.{name} AS SELECT * FROM fixture")
            con.unregister("fixture")
        loaded = n.load_matches(con, "female", 2025)
        assert len(loaded) == 4
        result = n.normalize(loaded)
        assert result.normalization_status.eq("ok").sum() == 2
        assert result.normalization_status.eq("missing_team_stats").sum() == 2
        row = result.query("match_id == 1 and team_id == 10").iloc[0]
        assert row.p50_interceptions == pytest.approx(10)
