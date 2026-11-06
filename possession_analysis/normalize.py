"""Possession normalization: compare defensive activity at a common reference.

Question: two teams make 14 interceptions, but one has 30% possession and the
other has 70%. Did they have the same opportunity to make those interceptions?

Idea: express each game's count at 50% possession and keep the observed count
alongside it. The fixed correction gives 10 and 23.3 interception equivalents:

    adjusted = count * 50 / opponent_possession_percent

This assumes defensive counts grow proportionally with opponent possession.
The learned exponent weakens that assumption; the context model estimates the
relationship while accounting for team and match differences. Both fit on earlier
seasons and are compared on a later season. No database tables are modified.

Run:
    python -m possession_analysis.normalize --gender female --year 2025
    python -m possession_analysis.normalize --input-csv history.csv --output-dir results/possession
    python -m possession_analysis.normalize --methods fixed --gender female --year 2025
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

STATS = ("interceptions", "defensive_duels", "defensive_duels_won")
LABELS = {"interceptions": "Interceptions", "defensive_duels": "Defensive duels",
          "defensive_duels_won": "Defensive duels won"}
BIN_EDGES = [0, 35, 45, 55, 65, 100]
BIN_LABELS = ["<35%", "35–45%", "45–55%", "55–65%", "65%+"]
MIN_SHARE = 0.05                  # exclude extreme exposure; do not clip possession
PAIR_TOLERANCE = 1.0              # percentage points, to allow provider rounding


# --- Loading ---

def connect(database: str | None = None):
    """Connect using the same local environment variables as the group pipeline."""
    import duckdb
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    token = os.getenv("MOTHERDUCK_TOKEN")
    if not token:
        raise RuntimeError("Set MOTHERDUCK_TOKEN locally before loading live data.")
    db = database or os.getenv("MOTHERDUCK_DB", "wyscout_dev")
    return duckdb.connect(f"md:{db}", config={"motherduck_token": token})


def match_query(gender: str, year: int, term: str = "Fall", database: str = "wyscout_dev") -> tuple[str, list]:
    """Both sides of every selected match, including missing-stat rows for coverage."""
    if gender not in {"male", "female"} or term not in {"Fall", "Spring"}:
        raise ValueError("Use gender male/female and term Fall/Spring.")
    if isinstance(year, bool) or not isinstance(year, (int, np.integer)) or not 1900 <= year <= 2100:
        raise ValueError("year must be an integer between 1900 and 2100.")
    if not database:
        raise ValueError("database must not be empty.")
    db = '"' + database.replace('"', '""') + '"'
    prefix = f"{db}.wyscout"
    stats = ", ".join(f"a.total_{s}" for s in STATS)
    sql = f"""
        WITH selected AS (
            SELECT m.match_id, m.date_utc AS date, m.home_team_id, m.away_team_id,
                   m.duration AS match_duration, t.year, t.term, c.gender,
                   c.comp_id AS competition_id, m.home_score, m.away_score
            FROM {prefix}.matches m
            JOIN {prefix}.seasons s ON s.season_id=m.season_id
            JOIN {prefix}.competitions c ON c.comp_id=s.comp_id
            JOIN {prefix}.season_terms t ON t.season_id=s.season_id
            WHERE c.gender=? AND t.year=? AND t.term=? AND t.kind='regular' AND m.status='Played'
        ), sides AS (
            SELECT match_id, date, year, term, gender, match_duration, competition_id,
                   home_team_id AS team_id, away_team_id AS opponent_id, true AS is_home,
                   home_score AS team_goals, away_score AS opponent_goals FROM selected
            UNION ALL
            SELECT match_id, date, year, term, gender, match_duration, competition_id,
                   away_team_id AS team_id, home_team_id AS opponent_id, false AS is_home,
                   away_score AS team_goals, home_score AS opponent_goals FROM selected
        )
        SELECT s.*, t.name AS team_name, o.name AS opponent_name,
               a.match_id IS NOT NULL AS stats_available,
               b.match_id IS NOT NULL AS opponent_stats_available,
               a.average_possession_percent AS reported_possession_pct,
               b.average_possession_percent AS opponent_reported_possession_pct,
               a.total_passes AS passes, b.total_passes AS opponent_passes, {stats}
        FROM sides s
        LEFT JOIN {prefix}.team_match_stats a ON a.match_id=s.match_id AND a.team_id=s.team_id
        LEFT JOIN {prefix}.team_match_stats b ON b.match_id=s.match_id AND b.team_id=s.opponent_id
        LEFT JOIN {prefix}.teams t ON t.team_id=s.team_id
        LEFT JOIN {prefix}.teams o ON o.team_id=s.opponent_id
        ORDER BY s.date, s.match_id, s.team_id
    """
    return sql, [gender, int(year), term]


def load_matches(con, gender: str, year: int, term: str = "Fall", database: str = "wyscout_dev") -> pd.DataFrame:
    """Read team-match rows. Missing possession fields are an error, never a proxy fallback."""
    columns = con.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_catalog=? AND table_schema='wyscout' AND table_name='team_match_stats'
    """, [database]).df()["column_name"]
    required = {"average_possession_percent", "total_passes", *[f"total_{s}" for s in STATS]}
    missing = required - set(columns)
    if missing:
        raise ValueError(f"Required team-match columns are absent: {sorted(missing)}")
    sql, params = match_query(gender, year, term, database)
    df = con.execute(sql, params).df()
    if df.empty:
        raise ValueError("No played matches match this gender/year/term.")
    return df


def load_csv(path: Path | str, gender: str, year: int, term: str = "Fall") -> pd.DataFrame:
    """Read a prior team-match export and select the requested scope explicitly."""
    match_query(gender, year, term)  # validate the same scope as live loading
    df = pd.read_csv(path)
    if not {"gender", "year", "term"}.issubset(df):
        raise ValueError("CSV must include gender, year and term from load_matches.")
    mask = df["gender"].eq(gender) & pd.to_numeric(df["year"], errors="coerce").eq(year) & df["term"].eq(term)
    if not mask.any():
        raise ValueError("CSV has no matches in the requested gender/year/term.")
    return df.loc[mask].reset_index(drop=True)


def history_query(gender: str, years: list[int], term: str = "Fall", database: str = "wyscout_dev") -> tuple[str, list]:
    """Use one SELECT for multiple seasons without changing the single-year API."""
    if not years or len(set(years)) != len(years):
        raise ValueError("years must be nonempty and distinct.")
    for year in years:
        match_query(gender, year, term, database)
    sql, _ = match_query(gender, years[0], term, database)
    sql = sql.replace("t.year=?", "t.year IN (" + ",".join("?" for _ in years) + ")")
    return sql, [gender, *map(int, years), term]


def load_history(con, gender: str, years: list[int], term: str = "Fall", database: str = "wyscout_dev") -> pd.DataFrame:
    sql, params = history_query(gender, years, term, database)
    return con.execute(sql, params).df()


def load_history_csv(path: Path | str, gender: str, years: list[int], term: str = "Fall") -> pd.DataFrame:
    history_query(gender, years, term)
    df = pd.read_csv(path)
    if not {"gender", "year", "term"}.issubset(df):
        raise ValueError("CSV must contain gender, year and term.")
    mask = df["gender"].eq(gender) & pd.to_numeric(df["year"], errors="coerce").isin(years) & df["term"].eq(term)
    out = df.loc[mask].reset_index(drop=True)
    missing = set(years) - set(out["year"])
    if missing:
        raise ValueError(f"CSV has no data for years {sorted(missing)}; export multi-season history or use --methods fixed.")
    return out


# --- Possession and 50/50 equivalents ---

def normalize(df: pd.DataFrame, source: str = "reported", min_share: float = MIN_SHARE,
              pair_tolerance: float = PAIR_TOLERANCE) -> pd.DataFrame:
    """Append p50_<stat>, validity flags and the adjustment factor; keep all input rows.

    Reported percentages are rescaled by their pair's sum (normally 99.9–100.1).
    Source 'pass_share' is an explicit sensitivity analysis, not measured possession.
    Invalid/missing possession produces NaN, never a 50% substitution. A bad count
    invalidates only that metric. Duplicate keys and invalid identifiers are errors.
    """
    if source not in {"reported", "pass_share"}:
        raise ValueError("source must be reported or pass_share.")
    if not np.isfinite(min_share) or not 0 < min_share <= 0.5:
        raise ValueError("min_share must be in (0, 0.5].")
    if not np.isfinite(pair_tolerance) or pair_tolerance < 0:
        raise ValueError("pair_tolerance must be nonnegative.")
    own_col, opp_col = (("reported_possession_pct", "opponent_reported_possession_pct") if source == "reported"
                        else ("passes", "opponent_passes"))
    required = {"match_id", "team_id", "opponent_id", "stats_available", "opponent_stats_available",
                own_col, opp_col, *[f"total_{s}" for s in STATS]}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Required input columns are absent: {sorted(missing)}")
    if df[["match_id", "team_id", "opponent_id"]].isna().any().any():
        raise ValueError("Match and team identifiers must be populated.")
    if (df["team_id"] == df["opponent_id"]).any():
        raise ValueError("A team cannot be its own opponent.")
    if df.duplicated(["match_id", "team_id"]).any():
        raise ValueError("Duplicate (match_id, team_id) rows would double-count matches.")
    for col in ["stats_available", "opponent_stats_available"]:
        if not df[col].dropna().isin([True, False]).all():
            raise ValueError(f"{col} must contain booleans, not text.")
    out = df.copy()
    own = pd.to_numeric(out[own_col], errors="coerce").astype(float)
    opp = pd.to_numeric(out[opp_col], errors="coerce").astype(float)
    total = own + opp
    missing_source = own.isna() | opp.isna()
    invalid_source = ~np.isfinite(own) | ~np.isfinite(opp) | (own <= 0) | (opp <= 0)
    mismatch = pd.Series(False, index=out.index)
    if source == "reported":
        invalid_source |= (own >= 100) | (opp >= 100)
        mismatch = (total - 100).abs() > pair_tolerance + 1e-9
    share = own / total.where(total > 0)
    extreme = ~share.between(min_share, 1 - min_share)
    out["normalization_status"] = np.select(
        [~out["stats_available"].fillna(False).astype(bool),
         ~out["opponent_stats_available"].fillna(False).astype(bool),
         missing_source, invalid_source, mismatch, extreme],
        ["missing_team_stats", "missing_opponent_stats", "missing_possession",
         "invalid_possession", "pair_sum_mismatch", "extreme_possession"], default="ok")
    valid = out["normalization_status"].eq("ok")
    out["possession_source"] = source
    out["possession_pct"] = (share * 100).where(valid)
    out["opponent_possession_pct"] = ((1 - share) * 100).where(valid)
    out["normalization_factor"] = (0.5 / (1 - share).where(valid)).where(valid)
    for stat in STATS:
        value = pd.to_numeric(out[f"total_{stat}"], errors="coerce").astype(float)
        count_ok = np.isfinite(value) & (value >= 0) & np.isclose(value, value.round(), rtol=0, atol=1e-9)
        out[f"valid_{stat}"] = valid & count_ok
        out[f"p50_{stat}"] = (value * out["normalization_factor"]).where(out[f"valid_{stat}"])
    return out


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Row counts by exclusion reason. Match counts across reasons can overlap."""
    return (df.groupby("normalization_status", dropna=False)
            .agg(team_rows=("team_id", "size"), matches=("match_id", "nunique"))
            .sort_index().reset_index())


def _corr(x: pd.Series, y: pd.Series) -> float:
    return x.corr(y, method="spearman") if len(x) >= 3 and x.nunique() > 1 and y.nunique() > 1 else np.nan


def comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Compare counts on identical valid rows. Correlations are descriptive, not validation."""
    rows = []
    for stat in STATS:
        d = df.loc[df[f"valid_{stat}"]]
        raw = pd.to_numeric(d[f"total_{stat}"])
        adjusted = d[f"p50_{stat}"]
        rows.append({"stat": stat, "team_matches": len(d), "matches": d["match_id"].nunique(),
                     "raw_mean": raw.mean(), "p50_mean": adjusted.mean(),
                     "raw_sd": raw.std(), "p50_sd": adjusted.std(),
                     "raw_possession_rho": _corr(d["possession_pct"], raw),
                     "p50_possession_rho": _corr(d["possession_pct"], adjusted),
                     "mean_absolute_change": (adjusted - raw).abs().mean()})
    return pd.DataFrame(rows)


def by_possession(df: pd.DataFrame) -> pd.DataFrame:
    """Median and interquartile range in fixed possession bands, on the same rows."""
    rows = []
    for stat in STATS:
        d = df.loc[df[f"valid_{stat}"]].copy()
        d[f"total_{stat}"] = pd.to_numeric(d[f"total_{stat}"])
        d["band"] = pd.cut(d["possession_pct"], BIN_EDGES, labels=BIN_LABELS, right=False)
        for band, group in d.groupby("band", observed=True):
            raw, adjusted = group[f"total_{stat}"], group[f"p50_{stat}"]
            rows.append({"stat": stat, "band": str(band), "team_matches": len(group),
                         "raw_median": raw.median(), "raw_q25": raw.quantile(.25), "raw_q75": raw.quantile(.75),
                         "p50_median": adjusted.median(), "p50_q25": adjusted.quantile(.25),
                         "p50_q75": adjusted.quantile(.75)})
    return pd.DataFrame(rows, columns=["stat", "band", "team_matches", "raw_median", "raw_q25", "raw_q75",
                                      "p50_median", "p50_q25", "p50_q75"])


def team_summary(df: pd.DataFrame, min_matches: int = 5) -> pd.DataFrame:
    """Mean observed / 50%-equivalent count per game; each valid game has equal weight."""
    if min_matches < 1:
        raise ValueError("min_matches must be positive.")
    rows = []
    for stat in STATS:
        d = df.loc[df[f"valid_{stat}"]]
        for (team_id, name), group in d.groupby(["team_id", "team_name"], dropna=False):
            if len(group) < min_matches:
                continue
            rows.append({"stat": stat, "team_id": team_id, "team_name": name, "matches": len(group),
                         "mean_possession_pct": group["possession_pct"].mean(),
                         "raw_per_match": pd.to_numeric(group[f"total_{stat}"]).mean(),
                         "p50_per_match": group[f"p50_{stat}"].mean()})
    out = pd.DataFrame(rows, columns=["stat", "team_id", "team_name", "matches", "mean_possession_pct",
                                      "raw_per_match", "p50_per_match"])
    out["change_per_match"] = out["p50_per_match"] - out["raw_per_match"]
    return out


# --- Readable output and local exports ---

def print_report(df: pd.DataFrame, experiment=None) -> None:
    """Show coverage, then separate count prediction from adjustment diagnostics."""
    print("Coverage (one row is one team in one game)")
    print(coverage(df).to_string(index=False))
    if experiment is None:
        print("\nObserved counts and fixed 50/50 equivalents")
        print(comparison(df).round(3).to_string(index=False))
        return

    from .methods import score_table
    print("\nFit on earlier seasons; evaluate on the later season")
    print(experiment.split[["split", "years", "matches", "valid_possession_rows"]].to_string(index=False))
    print("\nLearned exponents (0 = no change, 1 = full fixed correction)")
    parameters = experiment.parameters.set_index("stat")[["training_rows", "alpha", "alpha_at_boundary"]]
    print(parameters.rename(index=LABELS).round(3).to_string())
    print("\nCount prediction: average absolute error in actions per game (lower is better)")
    print(score_table(experiment.scores, "mae").round(3).to_string())
    print("\nAdjusted counts: correlation with possession (descriptive, not an accuracy score)")
    print(score_table(experiment.scores, "adjusted_possession_rho").round(3).to_string())


def save_results(raw: pd.DataFrame, df: pd.DataFrame, output_dir: Path, experiment=None) -> None:
    """Keep detailed CSVs available while the notebook presents the main comparisons."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from . import plots

    output_dir.mkdir(parents=True, exist_ok=True)
    tables = [("team_matches", df), ("coverage", coverage(df)), ("comparison", comparison(df)),
              ("possession_bands", by_possession(df)), ("teams", team_summary(df))]
    if experiment is not None:
        from .methods import possession_bands
        tables += [("history", raw), ("parameters", experiment.parameters), ("split", experiment.split),
                   ("method_scores", experiment.scores), ("method_bands", possession_bands(df))]
    for name, table in tables:
        table.to_csv(output_dir / f"{name}.csv", index=False)

    figures = [("spread", plots.spread(df)), ("bands", plots.bands(df))]
    if experiment is not None:
        figures += [("learned_exponent", plots.method_bands(df, ("raw", "fixed", "power"))),
                    ("all_methods", plots.method_bands(df))]
    for name, fig in figures:
        fig.savefig(output_dir / f"{name}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


# --- CLI ---

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gender", choices=["male", "female"], default="female")
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--term", choices=["Fall", "Spring"], default="Fall")
    ap.add_argument("--database", default=None)
    ap.add_argument("--source", choices=["reported", "pass_share"], default="reported")
    ap.add_argument("--min-share", type=float, default=MIN_SHARE)
    ap.add_argument("--methods", choices=["all", "fixed"], default="all", help="all three methods, or the original single-season baseline")
    ap.add_argument("--train-years", type=int, nargs="+", help="earlier fit seasons; default: the two years preceding --year")
    ap.add_argument("--input-csv", type=Path, help="use previously exported team-match rows instead of connecting")
    ap.add_argument("--output-dir", type=Path, help="write CSV summaries and figures here")
    args = ap.parse_args()
    train_years = args.train_years or [args.year - 2, args.year - 1]
    years = train_years + [args.year]
    if args.input_csv:
        raw = (load_history_csv(args.input_csv, args.gender, years, args.term) if args.methods == "all"
               else load_csv(args.input_csv, args.gender, args.year, args.term))
    else:
        with connect(args.database) as con:
            db = args.database or os.getenv("MOTHERDUCK_DB", "wyscout_dev")
            raw = (load_history(con, args.gender, years, args.term, db) if args.methods == "all"
                   else load_matches(con, args.gender, args.year, args.term, db))
    experiment = None
    if args.methods == "all":
        from .methods import compare_methods
        experiment = compare_methods(raw, train_years, args.year, args.source, args.min_share)
        df = experiment.test
    else:
        df = normalize(raw, args.source, args.min_share)
    print_report(df, experiment)
    if args.output_dir:
        save_results(raw, df, args.output_dir, experiment)


if __name__ == "__main__":
    main()
