# Wyscout NCAA D1 soccer -> MotherDuck

Loads Wyscout v3 data for NCAA D1 men's and women's soccer into a MotherDuck (DuckDB) database
and rolls season stats up across competitions by year and term (Fall / Spring).

## Layout
- `wyscout/api.py` - Wyscout API client (returns pandas DataFrames)
- `wyscout/schema.sql` - idempotent DDL for schema `wyscout`
- `wyscout/load.py` - loader that fetches in parallel and can be re-run without duplicating data
- `wyscout/rollup.py` - cross-competition season rollup into `team_term_stats` / `player_term_stats`
- `wyscout/derived.py` - engineered stats (np_xg, turnover/style shares, team-vs-opponent shares) added during the rollup; documented in `wyscout/DERIVED_STATS.md` and in `stat_formulas` (`origin = 'derived'`); `wyscout/STAT_GROUPS.md` lists every raw and engineered stat by functional group
- `low_minute_player_analysis/` - everything for the low-minutes player work: rebuild a 900+ minute per-90 profile from a low-minute sample of any size up to 750 minutes. Everything is run from the notebooks; the `.py` files only hold functions
  - data: `common.py` (database connection, position groups), `audit_counts.py`, `checkpoints.py` (truth + one training prefix per match played), `stat_groups.py`, `distribution_norm.py`
  - methods: `reconstruct_methods.py` (raw, shrink, ridge, knn, OT), `temporal_methods.py` (Kalman, GRU); the final model is the ridge regression
  - scoring and use: `rate_rebuild.py`, `reconstruction_eval.py`, `apply_model.py`, `notebook_helpers.py` (plots and tables); scores are written to `results/`
  - `notebooks/` - `00_data_and_setup`, one notebook per method family (`01_minutes_shrinkage` ... `06_gru_sequence`), `07_method_comparison`, `08_use_final_model`
  - `archive/` - earlier versions and their results, reference only (git-ignored)
- `ratings/` - team rating systems (Elo variants, pi-ratings, Glicko, Dixon-Coles, ridge) with tuning, evaluation and comparison to published rankings; see `ratings/README.md`
- `possession_analysis/` - fixed 50/50, learned-exponent and context-model possession adjustments, with chronological evaluation and a notebook; see `possession_analysis/README.md`

## Setup
    pip install -r requirements.txt
    cp .env.example .env    # then fill in your credentials

## Run (from the repo root)
    python -m wyscout.load --stages dims rosters matches match-data season-stats check
    python -m wyscout.rollup --scope both     # after season-stats; add --include-friendlies if wanted

Loads are resumable via the `ingest_log` table; `--force` re-fetches, `--season-ids` / `--team-id` limit scope.

## Small-sample analysis
Open the notebooks in `low_minute_player_analysis/notebooks/` and run them top to bottom. They read the saved files in `results/`.
- `00_data_and_setup` - set `BUILD_DATA = True` to rebuild the truth and prefix files from the database (`checkpoints.build_and_save_data`).
- `07_method_comparison` - set `RUN_EVALUATION = True` to re-run every method on every group (`reconstruction_eval.run_evaluation`, takes hours).
- `08_use_final_model` - fit the final model and apply it to real low-minute players.
Keep `duckdb<2`: MotherDuck does not support DuckDB 2.x yet.

## Team ratings
Open the notebooks in `ratings/notebooks/` (see `ratings/README.md`): tuning, held-out evaluation, team comparison, validation against published rankings, and the Elo export.

## Possession adjustment
    python -m possession_analysis.normalize --gender female --year 2025
Then open `possession_analysis/notebooks/01_possession_normalization.ipynb`.
