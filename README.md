# Wyscout NCAA D1 soccer -> MotherDuck

Loads Wyscout v3 data for NCAA D1 men's and women's soccer into a MotherDuck (DuckDB) database
and rolls season stats up across competitions by year and term (Fall / Spring).

## Layout
- `wyscout/api.py` - Wyscout API client (returns pandas DataFrames)
- `wyscout/schema.sql` - idempotent DDL for schema `wyscout`
- `wyscout/load.py` - chunked, parallel, resumable loader
- `wyscout/rollup.py` - cross-competition season rollup into `team_term_stats` / `player_term_stats`

## Setup
    pip install -r requirements.txt
    cp .env.example .env    # fill in credentials

## Run (from the repo root)
    python -m wyscout.load --stages dims rosters matches match-data season-stats check
    python -m wyscout.rollup --scope both     # after season-stats; add --include-friendlies if wanted

Loads are resumable via the `ingest_log` table; `--force` re-fetches, `--season-ids` / `--team-id` limit scope.
