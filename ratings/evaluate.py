"""Walk-forward comparison of all team-rating models.

    python -m ratings.evaluate --gender male

Each model rates matches in time order (predict, then update). The gap->outcome mapping is fit
on out-of-sample gaps from the training seasons (first season is burn-in) and scored on the
last `--test-seasons` seasons. Also reports a gap-stacking ensemble and a constant baseline.
"""
from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import argparse

import numpy as np
import pandas as pd

from ratings.base import OutcomeMapper, RatingModel, score, walk_forward
from ratings.data import load_matches
from ratings.specs import SPECS, build
from ratings.tune import load_best


class Constant(RatingModel):
    name = "baseline_home_rate"

    def gap(self, m):
        return 0.0

    def update(self, m):
        pass


def default_models(gender: str | None = None, tuned: bool = True) -> list[RatingModel]:
    """All models; uses ratings/best_params.json (from ratings.tune) for `gender` when tuned."""
    best = load_best(gender) if (tuned and gender) else {}
    return [Constant()] + [build(n, best.get(n, {}).get("params")) for n in SPECS]


def split_years(matches: pd.DataFrame, test_seasons: int):
    years = sorted(matches["year"].unique())
    if len(years) < 3:
        raise SystemExit(f"Need >=3 seasons, found {years}")
    test = years[-test_seasons:]
    train = [y for y in years[1:] if y not in test]  # first season = burn-in
    return train, test


def evaluate(matches: pd.DataFrame, models=None, test_seasons: int = 1, return_preds: bool = False):
    """Score every model on the test seasons. With return_preds also returns
    ({model: walk-forward frame}, {model: fitted OutcomeMapper}, test_years)."""
    models = models or default_models()
    train_y, test_y = split_years(matches, test_seasons)
    preds = {m.name: walk_forward(m, matches) for m in models}

    # Ensemble: OLS of goal diff on all model gaps (fit on training seasons only).
    names = [n for n in preds if n != "baseline_home_rate"]
    G = np.c_[[preds[n]["gap"].values for n in names]].T
    base = next(iter(preds.values()))
    gd = (base["hg"] - base["ag"]).values
    tr = base["year"].isin(train_y).values
    Gz = (G - G[tr].mean(0)) / (G[tr].std(0) + 1e-9)
    A = np.c_[Gz, np.ones(len(Gz))]
    coef = np.linalg.lstsq(A[tr], gd[tr], rcond=None)[0]
    ens = base.copy()
    ens["gap"] = A @ coef
    ens.attrs["coef"] = dict(zip(names, coef[:-1]))  # weights per z-scored model gap
    preds["ensemble_ols"] = ens

    rows, mappers = [], {}
    for name, df in preds.items():
        trn, tst = df[df["year"].isin(train_y)], df[df["year"].isin(test_y)]
        mappers[name] = OutcomeMapper().fit(trn["gap"], trn["hg"], trn["ag"])
        rows.append({"model": name, **score(tst, mappers[name])})
    table = pd.DataFrame(rows).set_index("model").sort_values("log_loss")
    return (table, preds, mappers, test_y) if return_preds else table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gender", default="male", choices=["male", "female"])
    ap.add_argument("--test-seasons", type=int, default=1)
    ap.add_argument("--untuned", action="store_true", help="ignore ratings/best_params.json")
    args = ap.parse_args()
    matches = load_matches(args.gender)
    print(f"{len(matches)} matches, seasons {sorted(matches['year'].unique())}, "
          f"xG coverage {matches['hxg'].notna().mean():.0%}")
    pd.set_option("display.width", 200)
    models = default_models(args.gender, tuned=not args.untuned)
    print(evaluate(matches, models, test_seasons=args.test_seasons).round(4))


if __name__ == "__main__":
    main()
