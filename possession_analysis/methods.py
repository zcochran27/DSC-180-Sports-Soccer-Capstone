"""Two alternatives to the fixed 50/50 possession adjustment.

Question: does twice as much opponent possession really mean twice as many
defensive actions? The fixed method assumes it does. These methods relax that:

* Learned exponent: learn the strength of the possession effect for each stat.
  alpha=0 keeps the count; alpha=1 gives the full fixed correction.
* Context model: estimate expected counts at observed and 50% possession, holding
  team, opponent and match context constant. Multiply by the ratio of those means.

Fit on earlier seasons, then compare on a later season. Predicting actual counts
is testable; the count a team would have recorded at 50% possession is unobserved.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.sparse import hstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from . import normalize as n

METHODS = {"raw": "No adjustment", "fixed": "Fixed 50/50", "power": "Learned exponent", "context": "Context model"}
PREFIXES = {"raw": "total_", "fixed": "p50_", "power": "power50_", "context": "context50_"}
NUMERIC = ["team_elo", "opponent_elo", "is_home"]
CATEGORICAL = ["team_id", "competition_id", "match_duration"]
CONTEXT_PENALTY = 0.01             # fixed in advance; not tuned on the evaluation season
MIN_TRAIN_ROWS = 50


# --- Strength before each match ---

def add_prematch_elo(df: pd.DataFrame) -> pd.DataFrame:
    """Reuse the team's Elo class with fixed defaults, recording ratings before results.

    Ratings start at 1500 in the first supplied season. Same-time matches all use
    ratings from before that timestamp. Scores update later ratings only; they are
    never direct regression inputs. Stored, potentially tuned Elo is overwritten.
    """
    from ratings.models.elo import Elo

    required = {"match_id", "team_id", "opponent_id", "is_home", "team_goals",
                "opponent_goals", "date", "year", "gender", "term"}
    if required - set(df):
        raise ValueError(f"Missing Elo history columns: {sorted(required - set(df))}")
    if df["gender"].nunique(dropna=False) != 1 or df["term"].nunique(dropna=False) != 1:
        raise ValueError("Build Elo for one gender and term at a time.")
    if df.duplicated(["match_id", "team_id"]).any():
        raise ValueError("Duplicate match/team keys in Elo history.")
    out = df.copy()
    if not out["is_home"].isin([True, False]).all():
        raise ValueError("is_home must be boolean or 0/1.")
    home = out.loc[out["is_home"].eq(True)].copy()
    away = out.loc[out["is_home"].eq(False)]
    if home["match_id"].duplicated().any() or away["match_id"].duplicated().any() or set(home["match_id"]) != set(away["match_id"]):
        raise ValueError("Elo history needs exactly one home and away row per match.")
    pair = home.merge(away, on="match_id", suffixes=("_h", "_a"), validate="one_to_one")
    if not (pair["team_id_h"].eq(pair["opponent_id_a"]) & pair["team_id_a"].eq(pair["opponent_id_h"])).all():
        raise ValueError("Home and away identifiers do not match.")
    for left, right in [("team_goals_h", "opponent_goals_a"), ("opponent_goals_h", "team_goals_a"),
                        ("year_h", "year_a"), ("date_h", "date_a")]:
        if not (pair[left].eq(pair[right]) | (pair[left].isna() & pair[right].isna())).all():
            raise ValueError("Home and away match context disagrees.")
    home["_date"] = pd.to_datetime(home["date"], errors="coerce", utc=True)
    if home["_date"].isna().any():
        raise ValueError("Dates are required for pre-match Elo.")
    # Explicit constants make this experiment independent of ratings tuning files.
    model = Elo(k=20.0, hfa=60.0, mov="none", carry=.75)
    ratings, last_year = {}, None
    for _, group in home.sort_values(["_date", "match_id"]).groupby("_date", sort=True):
        if group["year"].nunique() != 1:
            raise ValueError("One timestamp belongs to multiple season years.")
        year = group["year"].iloc[0]
        if year != last_year:
            model.start_season(year)
            last_year = year
        for row in group.itertuples(index=False):
            ratings[(row.match_id, row.team_id)] = (model.r[row.team_id], model.r[row.opponent_id])
            ratings[(row.match_id, row.opponent_id)] = (model.r[row.opponent_id], model.r[row.team_id])
        for row in group.itertuples(index=False):
            goals = pd.to_numeric(pd.Series([row.team_goals, row.opponent_goals]), errors="coerce").to_numpy(float)
            if np.isfinite(goals).all() and (goals >= 0).all() and np.equal(goals, goals.round()).all():
                model.update(SimpleNamespace(home=row.team_id, away=row.opponent_id, hg=goals[0], ag=goals[1],
                                             hxg=np.nan, axg=np.nan))
    values = [ratings[(row.match_id, row.team_id)] for row in out.itertuples(index=False)]
    out[["team_elo", "opponent_elo"]] = values
    return out


# --- Training and evaluation rows ---

def context_valid(df: pd.DataFrame) -> pd.Series:
    """Missing context is flagged rather than imputed from the evaluation set."""
    missing = set(NUMERIC + CATEGORICAL) - set(df)
    if missing:
        raise ValueError(f"Missing context columns: {sorted(missing)}. Reload the team-match export.")
    numeric = df[NUMERIC].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(numeric).all(axis=1) & numeric["is_home"].isin([0, 1])
    valid &= df[CATEGORICAL].notna().all(axis=1)
    return valid


def split_history(df: pd.DataFrame, train_years: list[int], test_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep both sides of a match together and require all fit dates before test dates."""
    if not train_years or len(set(train_years)) != len(train_years) or max(train_years) >= test_year:
        raise ValueError("Training years must be distinct, nonempty and earlier than the test year.")
    if df["gender"].nunique(dropna=False) != 1 or df["term"].nunique(dropna=False) != 1:
        raise ValueError("Fit one gender and one term at a time.")
    if df.duplicated(["match_id", "team_id"]).any():
        raise ValueError("Duplicate match/team keys in history.")
    missing = set(train_years + [test_year]) - set(df["year"])
    if missing:
        raise ValueError(f"No data for requested years: {sorted(missing)}")
    train, test = df[df["year"].isin(train_years)].copy(), df[df["year"] == test_year].copy()
    train_date = pd.to_datetime(train["date"], errors="coerce", utc=True)
    test_date = pd.to_datetime(test["date"], errors="coerce", utc=True)
    if train_date.isna().any() or test_date.isna().any() or train_date.max() >= test_date.min():
        raise ValueError("Every training date must precede every evaluation date.")
    if set(train["match_id"]) & set(test["match_id"]):
        raise ValueError("Training and evaluation matches overlap.")
    return train, test


# --- Method 2: learn how strongly counts depend on possession ---

@dataclass(frozen=True)
class PowerFit:
    alpha: float
    mean_at_50: float
    n_train: int
    at_boundary: bool

    def predict(self, possession_pct: pd.Series) -> np.ndarray:
        q = 1 - possession_pct.to_numpy(float) / 100
        if not np.isfinite(q).all() or ((q <= 0) | (q >= 1)).any():
            raise ValueError("Prediction possession must be finite and between 0 and 100.")
        return self.mean_at_50 * (q / .5) ** self.alpha


def fit_power(possession_pct: pd.Series, counts: pd.Series, alpha: float | None = None) -> PowerFit:
    """Poisson maximum likelihood with intercept profiled out; zeros are retained.

    alpha=None estimates alpha in [0, 1]. A fixed alpha is used for the full
    correction (1) and unadjusted reference (0). This bounds the fitted correction
    between those two references; a boundary estimate is reported explicitly.
    """
    p, y = np.asarray(possession_pct, float), np.asarray(counts, float)
    if len(p) != len(y) or len(p) < MIN_TRAIN_ROWS:
        raise ValueError(f"Need at least {MIN_TRAIN_ROWS} paired training rows.")
    count_ok = np.isfinite(y).all() and (y >= 0).all() and np.isclose(y, y.round(), rtol=0, atol=1e-9).all()
    possession_ok = np.isfinite(p).all() and (p > 0).all() and (p < 100).all()
    if not count_ok or not possession_ok:
        raise ValueError("Training possession and counts must be finite and in range.")
    if y.sum() <= 0 or np.unique(p).size < 3 or not p.min() <= 50 <= p.max():
        raise ValueError("Training data need positive events and possession variation spanning 50%.")
    x = np.log((1 - p / 100) / .5)

    def objective(a):
        return y.sum() * logsumexp(a * x) - a * np.dot(y, x)

    if alpha is None:
        result = minimize_scalar(objective, bounds=(0, 1), method="bounded", options={"xatol": 1e-9})
        if not result.success:
            raise RuntimeError(f"Exponent fit failed: {result.message}")
        # Include exact endpoints so boundary optima are not merely near 0 or 1.
        alpha = min([0.0, float(result.x), 1.0], key=objective)
    elif not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")
    mean_at_50 = np.exp(np.log(y.sum()) - logsumexp(alpha * x))
    return PowerFit(float(alpha), float(mean_at_50), len(y), alpha < 1e-5 or alpha > 1 - 1e-5)


# --- Method 3: estimate the possession effect alongside context ---

class ContextFit:
    """Penalized Poisson mean with a smooth possession curve and shrunk team effects.

    Splines, scaling and categories are fitted on training rows only. Unknown
    categories contribute zero categorical effect. No missing Elo is invented.
    The Poisson objective estimates the conditional mean; no Poisson-based
    confidence intervals or significance tests are reported.
    """
    def __init__(self, penalty: float = CONTEXT_PENALTY):
        if not np.isfinite(penalty) or penalty <= 0:
            raise ValueError("penalty must be positive.")
        self.penalty = penalty
        self.spline = SplineTransformer(n_knots=4, degree=2, knots="quantile",
                                        include_bias=False, extrapolation="linear")
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        self.model = PoissonRegressor(alpha=penalty, max_iter=2000, tol=1e-7)
        self.fitted = False

    def _matrix(self, df: pd.DataFrame, fit: bool = False):
        p = df[["possession_pct"]].to_numpy(float) / 100
        numeric = df[NUMERIC].apply(pd.to_numeric).to_numpy(float)
        categories = df[CATEGORICAL].astype(str).to_numpy()
        if fit:
            curve = self.spline.fit_transform(p)
            numeric = self.scaler.fit_transform(numeric)
            categories = self.encoder.fit_transform(categories)
        else:
            curve = self.spline.transform(p)
            numeric = self.scaler.transform(numeric)
            categories = self.encoder.transform(categories)
        return hstack([curve, numeric, categories], format="csr")

    def fit(self, df: pd.DataFrame, stat: str):
        if len(df) < MIN_TRAIN_ROWS or not context_valid(df).all():
            raise ValueError("Insufficient complete-context training rows.")
        p = df["possession_pct"]
        if p.nunique() < 5 or not p.min() <= 50 <= p.max():
            raise ValueError("Possession variation spanning 50% is needed for the context model.")
        y = pd.to_numeric(df[f"total_{stat}"]).to_numpy(float)
        if not np.isfinite(y).all() or (y < 0).any() or y.sum() == 0:
            raise ValueError("Context-model counts must be nonnegative with some positive events.")
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            self.model.fit(self._matrix(df, fit=True), y)
        self.n_train = len(df)
        self.train_possession_range = (float(p.min()), float(p.max()))
        self.known_teams = frozenset(df["team_id"].astype(str))
        self.fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Fit the context model before prediction.")
        if not context_valid(df).all():
            raise ValueError("Prediction context is incomplete.")
        p = pd.to_numeric(df["possession_pct"], errors="coerce")
        if not np.isfinite(p).all() or not p.between(0, 100, inclusive="neither").all():
            raise ValueError("Prediction possession must be finite and between 0 and 100.")
        prediction = self.model.predict(self._matrix(df))
        if not np.isfinite(prediction).all() or (prediction <= 0).any():
            raise RuntimeError("Context model produced nonfinite or nonpositive expected counts.")
        return prediction


# --- Apply the methods to the same later season ---

@dataclass
class Experiment:
    test: pd.DataFrame
    parameters: pd.DataFrame
    scores: pd.DataFrame
    split: pd.DataFrame
    power: dict[str, PowerFit]
    context: dict[str, ContextFit]


def compare_methods(raw: pd.DataFrame, train_years: list[int], test_year: int,
                    source: str = "reported", min_share: float = n.MIN_SHARE,
                    penalty: float = CONTEXT_PENALTY) -> Experiment:
    """Fit on earlier seasons and apply all three methods to the held-out season."""
    # Validate the split before deriving any chronological strength features.
    split_history(raw, train_years, test_year)
    history = raw.loc[raw["year"].isin(train_years + [test_year])]
    normalized = n.normalize(add_prematch_elo(history), source=source, min_share=min_share)
    train, test = split_history(normalized, train_years, test_year)
    train_context, test_context = context_valid(train), context_valid(test)
    test["context_available"] = test_context
    test["unseen_team"] = ~test["team_id"].isin(train["team_id"])
    power, context, params = {}, {}, []
    for stat in n.STATS:
        fit_rows = train.loc[train[f"valid_{stat}"] & train_context]
        p, y = fit_rows["possession_pct"], pd.to_numeric(fit_rows[f"total_{stat}"])
        # All mean models use the same training rows, including the two references.
        references = {"raw": fit_power(p, y, alpha=0), "fixed": fit_power(p, y, alpha=1)}
        power[stat] = fit_power(p, y)
        context[stat] = ContextFit(penalty).fit(fit_rows, stat)
        lo, hi = context[stat].train_possession_range
        test[f"outside_training_range_{stat}"] = test["possession_pct"].notna() & ~test["possession_pct"].between(lo, hi)
        # Exponent correction: weaken the fixed factor by the fitted alpha.
        valid = test[f"valid_{stat}"]
        q = 1 - test.loc[valid, "possession_pct"] / 100
        test[f"power_factor_{stat}"] = np.nan
        test.loc[valid, f"power_factor_{stat}"] = (.5 / q) ** power[stat].alpha
        test[f"power50_{stat}"] = pd.to_numeric(test[f"total_{stat}"], errors="coerce") * test[f"power_factor_{stat}"]
        for method, fit in {**references, "power": power[stat]}.items():
            test[f"pred_{method}_{stat}"] = np.nan
            test.loc[valid, f"pred_{method}_{stat}"] = fit.predict(test.loc[valid, "possession_pct"])
        # Context correction: change possession alone in the reference prediction.
        usable = valid & test_context
        observed = test.loc[usable].copy()
        reference = observed.copy()
        reference["possession_pct"] = 50.0
        test[f"pred_context_{stat}"] = np.nan
        test[f"context_factor_{stat}"] = np.nan
        if len(observed):
            expected = context[stat].predict(observed)
            expected_50 = context[stat].predict(reference)
            test.loc[usable, f"pred_context_{stat}"] = expected
            test.loc[usable, f"context_factor_{stat}"] = expected_50 / expected
        test[f"context50_{stat}"] = pd.to_numeric(test[f"total_{stat}"], errors="coerce") * test[f"context_factor_{stat}"]
        test[f"common_{stat}"] = valid & test_context
        params.append({"stat": stat, "training_rows": len(fit_rows), "alpha": power[stat].alpha,
                       "alpha_at_boundary": power[stat].at_boundary,
                       "mean_at_50": power[stat].mean_at_50,
                       "training_possession_min": lo, "training_possession_max": hi,
                       "context_penalty": penalty, "context_iterations": int(context[stat].model.n_iter_)})
    split = pd.DataFrame([
        {"split": name, "years": ", ".join(map(str, sorted(df["year"].unique()))),
         "matches": df["match_id"].nunique(), "team_rows": len(df),
         "valid_possession_rows": int(df["normalization_status"].eq("ok").sum()),
         "missing_context_rows": int((~context_valid(df)).sum()),
         "first_date": str(pd.to_datetime(df["date"], utc=True).min().date()),
         "last_date": str(pd.to_datetime(df["date"], utc=True).max().date())}
        for name, df in [("Training", train), ("Evaluation", test)]])
    return Experiment(test, pd.DataFrame(params), score_methods(test), split, power, context)


# --- Compare prediction error and the resulting adjusted counts ---

def score_methods(df: pd.DataFrame) -> pd.DataFrame:
    """All forecasts and adjusted-count diagnostics use identical held-out rows."""
    rows = []
    for stat in n.STATS:
        d = df.loc[df[f"common_{stat}"]]
        if d.empty:
            raise ValueError(f"No common evaluation rows for {stat}.")
        y = pd.to_numeric(d[f"total_{stat}"]).to_numpy(float)
        for method in METHODS:
            prediction = d[f"pred_{method}_{stat}"].to_numpy(float)
            adjusted = d[PREFIXES[method] + stat]
            rows.append({"stat": stat, "method": method, "team_matches": len(d),
                         "matches": d["match_id"].nunique(),
                         "poisson_deviance": mean_poisson_deviance(y, prediction),
                         "rmse": float(np.sqrt(np.mean((y - prediction) ** 2))),
                         "mae": float(np.mean(np.abs(y - prediction))),
                         "observed_over_predicted": y.sum() / prediction.sum(),
                         "adjusted_mean": adjusted.mean(), "adjusted_sd": adjusted.std(),
                         "adjusted_possession_rho": n._corr(d["possession_pct"], adjusted)})
    return pd.DataFrame(rows)


def score_table(scores: pd.DataFrame, metric: str = "mae") -> pd.DataFrame:
    """Readable stat-by-method table, shared by the notebook and CLI."""
    table = scores.pivot(index="stat", columns="method", values=metric)
    table = table.reindex(index=n.STATS, columns=list(METHODS))
    return table.rename(index=n.LABELS, columns=METHODS).rename_axis(index="Statistic", columns=None)


def team_comparison(df: pd.DataFrame, team_name: str, min_matches: int = 5) -> pd.DataFrame:
    """Per-game means for one team, using the common cohort for all methods."""
    rows = []
    for stat in n.STATS:
        d = df.loc[df[f"common_{stat}"] & df["team_name"].eq(team_name)]
        if len(d) < min_matches:
            continue
        rows.append({"stat": stat, "matches": len(d), **{
            method: pd.to_numeric(d[prefix + stat]).mean() for method, prefix in PREFIXES.items()}})
    return pd.DataFrame(rows)


def possession_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Same held-out cohort and band definitions for every method."""
    rows = []
    for stat in n.STATS:
        d = df.loc[df[f"common_{stat}"]].copy()
        d["band"] = pd.cut(d["possession_pct"], n.BIN_EDGES, labels=n.BIN_LABELS,
                           right=False, include_lowest=True)
        for band, group in d.groupby("band", observed=False):
            for method, prefix in PREFIXES.items():
                y = pd.to_numeric(group[prefix + stat])
                rows.append({"stat": stat, "band": band, "method": method, "team_matches": len(group),
                             "median": y.median(), "q25": y.quantile(.25), "q75": y.quantile(.75)})
    return pd.DataFrame(rows)
