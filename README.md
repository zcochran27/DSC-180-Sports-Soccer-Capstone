# Wyscout NCAA D1 soccer -> MotherDuck

Loads Wyscout v3 data for NCAA D1 men's and women's soccer into a MotherDuck (DuckDB) database
and rolls season stats up across competitions by year and term (Fall / Spring).

## Layout
- `wyscout/api.py` - Wyscout API client (returns pandas DataFrames)
- `wyscout/schema.sql` - idempotent DDL for schema `wyscout`
- `wyscout/load.py` - chunked, parallel, resumable loader
- `wyscout/rollup.py` - cross-competition season rollup into `team_term_stats` / `player_term_stats`
- `low_minute_player_analysis/` - everything for the low-minutes player work: three ways to estimate player rates from small minute samples, scored on next-season stability
  - `common.py` (data loading, per-90, scoring), `minutes_shrinkage.py`, `optimal_transport.py`, `similar_player_prediction.py`
  - `notebooks/` - one notebook per method (`01`-`03`) plus `04_compare_methods.ipynb`
- `ratings/` - team rating systems (Elo variants, pi-ratings, Glicko, Dixon-Coles, ridge) with tuning, evaluation and validation against published rankings; see `ratings/README.md`

## Setup
    pip install -r requirements.txt
    cp .env.example .env    # fill in credentials

## Run (from the repo root)
    python -m wyscout.load --stages dims rosters matches match-data season-stats check
    python -m wyscout.rollup --scope both     # after season-stats; add --include-friendlies if wanted

Loads are resumable via the `ingest_log` table; `--force` re-fetches, `--season-ids` / `--team-id` limit scope.

## Small-sample analysis
    python -m low_minute_player_analysis.minutes_shrinkage            # quick score table for each method
    python -m low_minute_player_analysis.optimal_transport
    python -m low_minute_player_analysis.similar_player_prediction
Then open the notebooks in `low_minute_player_analysis/notebooks/` (run from that folder). Keep `duckdb<2`: MotherDuck does not support DuckDB 2.x yet.

## Team ratings
    python -m ratings.evaluate --gender male        # held-out comparison of all rating models
    python -m ratings.matchup --gender male --model elo_sqrt --home "Georgetown Hoyas" --away "UCLA Bruins"
