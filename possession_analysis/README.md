# Possession normalization

**Question.** Two teams each make 14 interceptions. One has 30% possession and the
other has 70%. Did they have the same opportunity to defend?

Express each game's count at 50% possession, keeping its observed count alongside
it. With the fixed formula, those 14 interceptions become 10 and 23.33 equivalents.
The other two methods learn how much of that correction the data support.

## Three methods

Let `D` be the count and `q` the opponent possession share (0–1).

| Method | Formula | Strength | Main assumption or limitation |
|---|---|---|---|
| Fixed 50/50 | `D × (0.5 / q)` | Simple and transparent | Twice the opponent possession means twice the actions. |
| Learned exponent | `D × (0.5 / q)^alpha` | Smaller, stat-specific correction | One possession relationship is shared across teams. |
| Context model | `D × mu(50%, context) / mu(actual, context)` | Accounts for recorded team and match differences when fitting the possession curve | More assumptions; its flexible curve can raise some low-possession counts. |

The exponent is fitted in [0, 1]: zero keeps the observed count; one gives the fixed
correction. It always stays between those two values. All three methods preserve
zeros and leave counts unchanged at 50%.

## Notebook and results

`notebooks/01_possession_normalization.ipynb` follows the style of the low-minutes
notebooks: work through one statistic, then compare the full set.

- **Part A:** opportunity, the two illustrative teams, three methods, and a real
  UCSD game showing how the context ratio works.
- **Part B:** all three statistics, prediction error on later games, and two UCSD
  examples followed by season averages.
- **Takeaways:** what the methods show and what still needs testing for player use.

Saved outputs use real MotherDuck data captured on 2026-11-05: women's Fall
2023–24 for fitting, 2025 for evaluation. The examples labelled A and B are
illustrations, separate from the observed games.

All methods are compared on the same 6,260 team-game rows (3,130 games). Average
absolute count prediction error, in actions per game; lower is better:

| Statistic | Unadjusted mean reference | Fixed | Learned exponent | Context |
|---|---:|---:|---:|---:|
| Interceptions | 10.05 | 10.61 | 9.63 | 9.45 |
| Defensive duels | 17.89 | 16.44 | 15.49 | 14.90 |
| Duels won | 5.38 | 5.90 | 5.31 | 5.16 |

The context model predicts observed counts best. The learned exponent is a clear
candidate for an integration test because its adjustment is simple, bounded and
stat-specific. Neither result establishes accuracy against hypothetical 50/50
counts, which are unobserved. A smaller possession correlation is not proof of
better normalization.

## Run

From the repository root, install `requirements.txt` and set the local
`MOTHERDUCK_TOKEN` (optional `MOTHERDUCK_DB`, default `wyscout_dev`).

```sh
python -m possession_analysis.normalize --gender female --year 2025 --train-years 2023 2024
python -m possession_analysis.normalize --gender female --year 2025 --output-dir results/possession
python -m possession_analysis.normalize --methods fixed --gender female --year 2025
python -m pytest possession_analysis
```

The default runs all three methods using the two years before `--year` for fitting.
For offline use, pass `--input-csv <history.csv>`. The notebook accepts the same
multi-season file through `POSSESSION_MATCHES_CSV`; otherwise it queries MotherDuck.
A single-season export can run with `--methods fixed`. Saved notebook outputs are
readable without credentials.

The CLI prints coverage, fitted exponents, average prediction error and possession
correlations. `--output-dir` also saves the detailed CSVs and figures. `method_scores.csv`
includes MAE, RMSE, Poisson deviance, total calibration and adjusted-count summaries.
`history.csv` contains the source rows; `team_matches.csv` contains the evaluation
rows with every adjustment. The original `comparison.csv`, `possession_bands.csv`
and `teams.csv` describe raw versus fixed counts. Keep exports local unless sharing
them intentionally. No database objects are modified.

## Code

- `normalize.py`: loading, input checks, fixed formula, summaries and CLI.
- `methods.py`: chronological split, pre-match strength, learned fits, application
  and scoring. `score_table` gives the same readable comparison in the notebook and CLI.
- `plots.py`: shared plots, including single-stat views and multiplier curves.
- `test_normalize.py`, `test_methods.py`: formula, input, chronology and fitting checks.

## Data and model details

- **Scope:** one gender and term at a time; played matches in regular season entries,
  including stored nonconference and tournament games. Friendly entries are excluded.
  Both sides remain in the input when statistics are missing.
- **Possession:** both teams' `average_possession_percent`. Rescale by their sum for
  rounding, reject pairs more than one percentage point from 100, and require both
  shares to be at least 5%. Excluded rows get a status and NaN adjustments. A bad
  count invalidates only its statistic. Missing data are never filled with 50%.
- **Alternative source:** `--source pass_share` explicitly uses own passes divided
  by both sides' passes throughout fitting and evaluation. It is a sensitivity
  check, not a fallback for missing possession.
- **Split:** all fitting dates precede all evaluation dates; both sides of a match
  stay together. Preprocessing uses fitting rows only. All mean models fit the same
  complete-context rows; all method comparisons use the same held-out rows per stat.
- **Exponent:** Poisson maximum likelihood, retaining zeros, constrained to [0, 1].
  The saved estimates are 0.365 (interceptions), 0.608 (duels), and 0.167 (duels won),
  fitted on 12,692 earlier rows. Boundary estimates are flagged.
- **Context:** penalized Poisson log mean with a four-knot quadratic possession
  spline, standardized team/opponent Elo and home/away status, and categorical team,
  competition and provider duration. Penalty 0.01 is fixed in advance. Team effects
  shrink toward zero; unseen categories get zero categorical effect. The additive
  context terms cancel in the adjustment ratio, giving one multiplier per possession
  value for each statistic. The curve is not constrained to be monotonic.
- **Strength:** reuse `ratings.models.elo.Elo` with fixed `k=20`, `hfa=60`, `mov="none"`,
  `carry=0.75`; start at 1500 in the first supplied season. Record ratings before
  results update them; same-time matches share prior ratings. Earlier evaluation
  results may update later ratings. Current scores are never regression inputs.
  Stored ratings and tuning files are not used; invalid scores do not update Elo.
- **Flags:** `context_available`, `unseen_team`, and `outside_training_range_<stat>`.
  Missing context leaves fixed/exponent results available but excludes the row from
  the context adjustment and common comparison. No context ratios are clipped.
- **Summaries:** equal-weight averages of match-level equivalents, with five valid
  games required for a team summary. Do not adjust a season total using one average
  possession value.

This is retrospective prediction because possession and duration are known after
the game. Context-model gains can come from its additional predictors. The Poisson
loss estimates a mean; no Poisson confidence intervals or independent-row
significance tests are reported for these dispersed, paired observations.

The adjustments retain game duration. They are per-game equivalents, not per 90 or
[Wyscout Opp30](https://dataglossary.wyscout.com/p_adj/), whose reference is 30 minutes
of opponent possession. Success percentages, goals and Elo are not scaled. Player
use needs validation and ideally possession during actual playing stints. Score
state, territory and tactical roles remain unmeasured.
