"""Random-search hyperparameter tuning for every rating model.

    python -m ratings.tune --gender male --gender female

Validation protocol (the test season is never touched): matches up to and including the last
pre-test season are rated walk-forward; the gap->outcome mapper is fit on seasons before the
validation season (first season = burn-in); the config is scored by log loss on the validation
season. Best params per gender go to ratings/best_params.json.
"""
from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from ratings.base import OutcomeMapper, score, walk_forward
from ratings.data import load_matches
from ratings.specs import BATCH, SPECS, build

PARAMS_PATH = Path(__file__).with_name("best_params.json")


def val_split(matches: pd.DataFrame, test_seasons: int):
    years = sorted(matches["year"].unique())
    pre = years[:-test_seasons]
    return [int(y) for y in pre[1:-1]], int(pre[-1]), matches[matches["year"] <= pre[-1]]


def objective(name, params, data, fit_years, val_year) -> float:
    model = build(name, params)
    if name in BATCH:
        model.refit_days = 28  # coarser refits keep the search fast
    preds = walk_forward(model, data)
    trn, val = preds[preds["year"].isin(fit_years)], preds[preds["year"] == val_year]
    mapper = OutcomeMapper().fit(trn["gap"], trn["hg"], trn["ag"])
    return score(val, mapper)["log_loss"]


def sample(space: dict, n: int, rng: random.Random):
    seen, out = set(), []
    total = 1
    for v in space.values():
        total *= len(v)
    while len(out) < min(n, total):
        p = {k: rng.choice(v) for k, v in space.items()}
        key = tuple(sorted(p.items()))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def tune(gender: str, test_seasons=1, n_online=40, n_batch=8, seed=0, only=None) -> dict:
    matches = load_matches(gender)
    fit_years, val_year, data = val_split(matches, test_seasons)
    rng, best, log = random.Random(seed), {}, []
    for name, (_, space) in SPECS.items():
        if only and name not in only:
            continue
        results = []
        for p in sample(space, n_batch if name in BATCH else n_online, rng):
            results.append((objective(name, p, data, fit_years, val_year), p))
        results.sort(key=lambda t: t[0])
        best[name] = {"params": results[0][1], "val_log_loss": results[0][0]}
        log += [{"gender": gender, "model": name, "val_log_loss": s, **p} for s, p in results]
        print(f"[{gender}] {name}: {results[0][0]:.4f} {results[0][1]}", flush=True)
    pd.DataFrame(log).to_csv(PARAMS_PATH.with_name(f"tuning_log_{gender}.csv"), index=False)
    return best


def load_best(gender: str) -> dict:
    if PARAMS_PATH.exists():
        return json.loads(PARAMS_PATH.read_text()).get(gender, {})
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gender", action="append", choices=["male", "female"])
    ap.add_argument("--test-seasons", type=int, default=1)
    ap.add_argument("--n-online", type=int, default=40)
    ap.add_argument("--n-batch", type=int, default=8)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    allp = json.loads(PARAMS_PATH.read_text()) if PARAMS_PATH.exists() else {}
    for g in args.gender or ["male", "female"]:
        allp[g] = {**allp.get(g, {}), **tune(g, args.test_seasons, args.n_online, args.n_batch, only=args.only)}
        PARAMS_PATH.write_text(json.dumps(allp, indent=2))


if __name__ == "__main__":
    main()
