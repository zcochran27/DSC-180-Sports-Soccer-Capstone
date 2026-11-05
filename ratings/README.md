# Team rating systems (NCAA D1 soccer)

Goal: estimate the skill gap between two teams. Every model here produces a **rating gap**
(home strength minus away strength, home advantage included). A shared mapping turns that gap into
**win/draw/loss probabilities** (ordered logit) and an **expected goal difference** (linear), so all
models are scored on the same footing.

**Chosen model: Elo with a square-root margin-of-victory multiplier (`elo_sqrt`).**

## How the pieces fit

| File | Role |
|---|---|
| `data.py` | Loads regular-season matches (scores, team xG) from MotherDuck `wyscout.*`. Men and women are rated separately. |
| `base.py` | `RatingModel` interface, walk-forward runner, `OutcomeMapper` (gap -> W/D/L and goal diff), scoring metrics. |
| `models/elo.py`, `pi.py`, `glicko.py`, `batch.py` | The models. |
| `specs.py` | Registry of model names, constructors and tuning search spaces. |
| `tune.py` | Random-search hyperparameter tuning -> `best_params.json`, `tuning_log_<gender>.csv`. |
| `evaluate.py` | Held-out comparison of all models (`python -m ratings.evaluate --gender male`). |
| `matchup.py` | Fit a model on all data, rank teams, compare two opponents. |
| `compare_polls.py`, `compare_rpi.py`, `history.py` | Validation against published rankings (single snapshot / multi-year). |
| `reference/` | Saved external rankings (coaches polls, NCAA RPI, Shortlistr). |
| `team_ratings.ipynb` | Notebook with all tables and plots. |

Use the chosen model:

```
python -m ratings.matchup --gender male --model elo_sqrt --home "Georgetown Hoyas" --away "UCLA Bruins"
python -m ratings.matchup --gender male --model elo_sqrt --home "Georgetown Hoyas" --away "UCLA Bruins" --neutral
```

or in code: `fit_final(gender, "elo_sqrt")`, then `predict_matchup(model, mapper, home_id, away_id, neutral=False)`.

## Evaluation protocol

* **Walk-forward, no look-ahead:** matches are processed in time order; each match is predicted from the
  state *before* it, then the model updates.
* First season is burn-in. The gap -> outcome mapping (and the ensemble weights) are fit on out-of-sample
  gaps from the training seasons. The **last season (2026, a partial season) is the held-out test.**
* **Tuning** uses the last pre-test season as validation (random search, log loss). The test season is
  never used for tuning.
* Metrics: log loss and ranked probability score (RPS) for W/D/L; MAE/RMSE of expected goal difference
  vs actual goal difference and vs xG difference. Lower is better for all.

## The models

Throughout, a team's rating moves after each match by how much the result differs from what the rating
gap predicted.

### Elo family (`models/elo.py`)
Each team has one rating (start 1500). Pre-match gap = `r_home - r_away + hfa`. Expected score
`E = 1 / (1 + 10^(-gap/400))`. After the match: `delta = K * mult * (S - E)`, added to the home team and
subtracted from the away team, where `S` = 1 / 0.5 / 0 for win / draw / loss. At each new season,
ratings are pulled toward 1500 by a factor `carry` (1.0 = no regression).

* **`elo_none`** – plain Elo, `mult = 1`.
* **`elo_sqrt`** *(chosen)* – `mult = sqrt(goal margin)`, so a 3-goal win moves ratings ~1.7x more than a
  1-goal win. Rewards convincing results without letting blowouts dominate.
* **`elo_wfe_autocorr`** – World-Football-Elo margin schedule (1, 1.5, (11+N)/8) with an autocorrelation
  damper that shrinks the boost when the favourite wins (so strong teams do not drift up forever).
* **`elo_xg`** – replaces the W/D/L result by `sigmoid(c * (w*xG_diff + (1-w)*goal_diff))`, so updates use
  chance quality as well as goals (falls back to goals when xG is missing, ~13% of matches).

### `pi` – Pi-ratings (`models/pi.py`, Constantinou & Fenton)
Each team has a **home** and an **away** rating in goal units. Expected goal difference = home team's home
rating minus away team's away rating. The error (actual minus expected) is passed through a
diminishing-returns function `c * log10(1 + |e|)`, scaled by learning rate `lam`, and applied to the
team's rating for that venue; a fraction `gamma` also moves its other-venue rating.

### `glicko` – Glicko-style rating with uncertainty (`models/glicko.py`)
Like Elo, but each team also carries a rating deviation (RD), a 1-D Kalman filter. RD shrinks as the team
plays, grows with idle time (`c_week`) and jumps at season boundaries (`season_c`). Teams with high RD
move more per match, which suits sparse schedules and roster turnover. `gap_sd(match)` gives the
uncertainty of a matchup.

### `dixon_coles` / `dixon_coles_xg` – Poisson attack/defence (`models/batch.py`)
Batch model, refit every 14 days on all earlier matches. Each team has an attack and a defence parameter;
home goals ~ Poisson(exp(mu + hfa + attack_home - defence_away)), away goals likewise. Matches are
down-weighted by age with a half-life, and parameters have a ridge penalty. Gap = expected home goals
minus expected away goals. The `_xg` version fits on a goals/xG blend. (The low-score correlation
parameter of the original paper is omitted; it barely changes the gap.)

### `ridge_gd` / `ridge_gd_xg` – ridge goal-difference ratings (`models/batch.py`)
Batch weighted ridge regression of goal difference on team indicators (+1 home, -1 away) plus a home
intercept, with the same time-decay weighting. A regularised Bradley-Terry-style batch comparator. `_xg`
fits the goals/xG blend.

### Ensembles (`evaluate.py`, `history.py`)
* **`ensemble_ols`** – OLS stack of all models' z-scored gaps, weights fit on training seasons.
* **`ensemble_mean`** – equal-weight average of z-scored strengths (only in the multi-year validation).

## Tuned hyperparameters (`best_params.json`)

Men / women, chosen by validation log loss:

| Model | Men | Women |
|---|---|---|
| **elo_sqrt** | K=60, hfa=70, carry=0.9 | K=80, hfa=70, carry=1.0 |
| elo_none | K=60, hfa=70, carry=1.0 | K=80, hfa=50, carry=1.0 |
| elo_wfe_autocorr | K=40, hfa=70, carry=1.0 | K=60, hfa=50, carry=1.0 |
| elo_xg | K=60, hfa=70, carry=1.0, w=0.5, c=1.5 | K=60, hfa=30, carry=0.9, w=0.25, c=0.6 |
| pi | lam=0.1, gamma=0.8, c=3, carry=0.8 | lam=0.1, gamma=0.5, c=5, carry=0.95 |
| glicko | hfa=60, c_week=2, season_c=120, carry=0.9, rd_max=200 | hfa=40, c_week=2, season_c=120, carry=0.9, rd_max=300 |
| dixon_coles | half-life 180d, ridge 0.5 | half-life 365d, ridge 0.5 |
| dixon_coles_xg | half-life 180d, ridge 0.5, xG w=0.5 | half-life 180d, ridge 0.5, xG w=0.5 |
| ridge_gd | half-life 180d, ridge 2 | half-life 365d, ridge 2 |
| ridge_gd_xg | half-life 365d, ridge 2, xG w=0.5 | half-life 365d, ridge 2, xG w=1.0 |

The best Elo settings use a large K and little or no regression between seasons: NCAA schedules are short
and rosters turn over, so ratings need to adapt quickly. These sat near the edge of the search grid, so
there may be a little more to gain from a wider search.

## Results

### 1. Held-out match prediction (2026 season, partial)

Men (1,579 matches) and women (2,780 matches). Log loss / RPS; lower is better. `baseline` = always predict
the training-set home/draw/away rates.

| Model | Men log loss | Men RPS | Women log loss | Women RPS |
|---|---|---|---|---|
| ensemble_ols | **0.9375** | 0.1902 | **0.8879** | 0.1780 |
| dixon_coles_xg | 0.9420 | 0.1915 | 0.8893 | 0.1783 |
| elo_xg | 0.9460 | 0.1926 | 0.8939 | 0.1800 |
| ridge_gd_xg | 0.9465 | 0.1933 | 0.9001 | 0.1822 |
| dixon_coles | 0.9547 | 0.1954 | 0.8952 | 0.1801 |
| ridge_gd | 0.9609 | 0.1975 | 0.8940 | 0.1802 |
| elo_wfe_autocorr | 0.9619 | 0.1977 | 0.9005 | 0.1819 |
| glicko | 0.9630 | 0.1983 | 0.8982 | 0.1814 |
| **elo_sqrt** | 0.9651 | 0.1987 | 0.8999 | 0.1817 |
| elo_none | 0.9653 | 0.1990 | 0.9052 | 0.1835 |
| pi | 0.9672 | 0.1994 | 0.9001 | 0.1820 |
| baseline | 1.0717 | 0.2372 | 1.0633 | 0.2387 |

Takeaways: every model beats the baseline by a wide margin. Models that use xG (`dixon_coles_xg`,
`elo_xg`) win on pure prediction. The square-root margin helps over plain Elo, but `elo_sqrt` is about
0.02 log loss behind the best men's model; that is a small gap in absolute terms (expected goal-difference
MAE 1.43 vs 1.38).

### 2. Agreement with published rankings (men's D1)

Rank correlation (Spearman) between our team strengths and each reference, averaged over the years that
reference covers. Our ratings use only matches up to the reference's date. References: NCAA RPI (2026,
~213 teams), Shortlistr Team Strength (2023-26, ~200 teams), United Soccer Coaches Week 11 poll (last
regular-season poll, 2021-26) and the post-tournament Final poll (2021-25). The polls are 25 teams only,
so their correlations are noisier and lower.

| Model | NCAA RPI | Shortlistr | Coaches Week 11 | Coaches Final |
|---|---|---|---|---|
| glicko | **0.943** | 0.889 | **0.643** | 0.539 |
| ensemble_mean | 0.935 | 0.909 | 0.601 | 0.574 |
| elo_none | 0.930 | 0.884 | 0.585 | 0.640 |
| **elo_sqrt** | 0.929 | 0.909 | 0.623 | **0.660** |
| ridge_gd | 0.924 | 0.903 | 0.470 | 0.483 |
| dixon_coles | 0.922 | 0.871 | 0.478 | 0.360 |
| ensemble_ols | 0.920 | 0.904 | 0.491 | 0.428 |
| pi | 0.917 | 0.910 | 0.511 | 0.546 |
| dixon_coles_xg | 0.916 | 0.890 | 0.465 | 0.356 |
| elo_xg | 0.911 | **0.920** | 0.511 | 0.495 |
| ridge_gd_xg | 0.899 | 0.848 | 0.406 | 0.377 |

Takeaways:
* Against the large references (RPI, Shortlistr) all models agree at 0.85-0.94, consistently across
  2023-2026.
* Against the human polls the margin-of-victory Elo models and Glicko track best; the Poisson and ridge
  models fall behind, sometimes badly in individual years (e.g. Dixon-Coles correlates 0.14 with the 2022
  Week 11 poll).
* **The best models for match prediction (xG, batch) are not the best at matching human/RPI rankings.**
  `elo_sqrt` is the most balanced choice: top-2 against the Shortlistr and Final-poll references, near the
  top elsewhere, simple, explainable, and fast to update.
* The regression-weighted ensemble over-fits (unstable weights on highly correlated inputs, including
  negative ones); the equal-weight average is steadier and was mildly look-ahead-free.

Per-year numbers: `history_results.csv`. Coaches-poll sources are saved in `reference/`.

## Caveats

* The test season (2026) is partial, and the database has no 2018 season.
* Team xG is missing for ~13% of matches (the xG models fall back to goals).
* No neutral-site flag in the data, so home advantage is applied to every match. Use
  `predict_matchup(..., neutral=True)` for neutral venues.
* The conferences are mostly self-contained, so cross-conference ratings depend on a limited number of
  non-conference games.
* A few schools cannot be name-matched to the external rankings (e.g. Sacred Heart, Western Illinois) and
  are excluded from the comparisons.
* The regression ensemble weights were fit on seasons through 2025, a mild look-ahead on 2021-2024
  snapshots.

## Reproduce

```
python -m ratings.tune --gender male --gender female        # tune; writes best_params.json
python -m ratings.evaluate --gender male                    # held-out table
python -m ratings.history                                   # multi-year validation -> history_results.csv
python -m ratings.compare_rpi --model elo_sqrt              # NCAA RPI / Shortlistr snapshot
py -m pytest ratings -q                                      # synthetic-data sanity tests
```
