"""Plots shared by the notebook and command-line analysis."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import normalize as n

RAW_COLOR = "#3274a1"
ADJUSTED_COLOR = "#c47837"
METHOD_COLORS = {"raw": RAW_COLOR, "fixed": ADJUSTED_COLOR,
                 "power": "#27856a", "context": "#8762a8"}


def observed(df, stats=n.STATS):
    """Baseline spread of counts before applying the adjustment."""
    width = 8 if len(stats) == 1 else 4.3 * len(stats)
    fig, axes = plt.subplots(1, len(stats), figsize=(width, 3.8))
    for ax, stat in zip(np.atleast_1d(axes), stats):
        d = df.loc[df[f"valid_{stat}"]]
        ax.scatter(d["possession_pct"], pd.to_numeric(d[f"total_{stat}"]), s=7, alpha=.15, color=RAW_COLOR,
                   rasterized=True)
        ax.axvline(50, color="grey", lw=.8, ls="--")
        ax.set(xlim=(0, 100), ylim=(0, None), xlabel="Team possession (%)",
               ylabel="Observed count per game", title=n.LABELS[stat])
        ax.grid(alpha=.15)
    fig.tight_layout()
    return fig


def spread(df):
    """Show every valid team-match; raw and adjusted panels share the same axes."""
    fig, axes = plt.subplots(len(n.STATS), 2, figsize=(12, 9), sharex=True, sharey="row")
    for i, stat in enumerate(n.STATS):
        d = df.loc[df[f"valid_{stat}"]]
        for j, (column, title, color) in enumerate([
            (f"total_{stat}", "Observed", RAW_COLOR), (f"p50_{stat}", "50/50 equivalent", ADJUSTED_COLOR)
        ]):
            ax = axes[i, j]
            ax.scatter(d["possession_pct"], pd.to_numeric(d[column]), s=7, alpha=.12, color=color, rasterized=True)
            ax.axvline(50, color="grey", lw=.8, ls="--")
            ax.set_title(f"{n.LABELS[stat]}: {title}")
            ax.set_ylabel("Count per team-match" if j == 0 else "Equivalent count per team-match")
            ax.set_ylim(bottom=0)
            ax.set_xlim(0, 100)
            ax.grid(alpha=.15)
    for ax in axes[-1]:
        ax.set_xlabel("Team possession (%)")
    fig.tight_layout()
    return fig


def bands(df, stats=n.STATS):
    """Medians and middle 50% of counts in possession bands; identical cohorts."""
    summary = n.by_possession(df)
    width = 8 if len(stats) == 1 else 4.3 * len(stats)
    fig, axes = plt.subplots(1, len(stats), figsize=(width, 3.8), squeeze=False)
    for ax, stat in zip(axes[0], stats):
        d = summary.loc[summary["stat"] == stat].set_index("band").reindex(n.BIN_LABELS)
        x = np.arange(len(n.BIN_LABELS))
        for prefix, label, color, offset in [("raw", "Observed", RAW_COLOR, -.08),
                                             ("p50", "Fixed 50/50", ADJUSTED_COLOR, .08)]:
            ax.plot(x + offset, d[f"{prefix}_median"], "o-", color=color, label=label)
            ax.fill_between(x + offset, d[f"{prefix}_q25"].to_numpy(float),
                            d[f"{prefix}_q75"].to_numpy(float), color=color, alpha=.12)
        ax.set_xticks(x, n.BIN_LABELS, rotation=25)
        ax.set(xlabel="Team possession", ylabel="Count / equivalent count", title=n.LABELS[stat], ylim=(0, None))
        ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def source_comparison(reported, proxy):
    """Compare measured possession percentages with pass share on common valid rows."""
    a = reported.loc[reported["normalization_status"] == "ok", ["match_id", "team_id", "possession_pct"]]
    b = proxy.loc[proxy["normalization_status"] == "ok", ["match_id", "team_id", "possession_pct"]]
    pair = a.merge(b, on=["match_id", "team_id"], suffixes=("_reported", "_proxy"), validate="one_to_one")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(pair["possession_pct_reported"], pair["possession_pct_proxy"], s=7, alpha=.15, color=RAW_COLOR,
               rasterized=True)
    ax.plot([0, 100], [0, 100], "--", color="grey", lw=1)
    ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="Reported possession (%)", ylabel="Pass share (%)",
           title="Possession and pass share describe different quantities")
    fig.tight_layout()
    return fig


def method_bands(df, methods=("raw", "fixed", "power", "context"), stats=n.STATS):
    """Held-out band medians; counts use a common cohort across all methods."""
    from .methods import METHODS, possession_bands
    summary = possession_bands(df)
    width = 8 if len(stats) == 1 else 4.3 * len(stats)
    fig, axes = plt.subplots(1, len(stats), figsize=(width, 3.8))
    for ax, stat in zip(np.atleast_1d(axes), stats):
        for method in methods:
            d = summary.loc[summary["stat"].eq(stat) & summary["method"].eq(method)]
            d = d.set_index("band").reindex(n.BIN_LABELS)
            ax.plot(np.arange(len(d)), d["median"], "o-", color=METHOD_COLORS[method],
                    label=METHODS[method], lw=1.5, ms=4)
        ax.set_xticks(np.arange(len(n.BIN_LABELS)), n.BIN_LABELS, rotation=25)
        ax.set(xlabel="Team possession", ylabel="Median count / equivalent count", title=n.LABELS[stat], ylim=(0, None))
        ax.grid(alpha=.15)
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def factors(experiment, stat="interceptions"):
    """Compare fitted multipliers; one means that the count is unchanged."""
    from .methods import METHODS

    fit = experiment.context[stat]
    lo, hi = fit.train_possession_range
    grid = np.linspace(max(20, lo), min(80, hi), 121)
    row = experiment.test.loc[experiment.test[f"common_{stat}"]].iloc[[0]]
    context = pd.concat([row] * len(grid), ignore_index=True).assign(possession_pct=grid)
    at_50 = context.assign(possession_pct=50.)
    fixed = .5 / (1 - grid / 100)
    values = {"fixed": fixed, "power": fixed ** experiment.power[stat].alpha,
              "context": fit.predict(at_50) / fit.predict(context)}

    fig, ax = plt.subplots(figsize=(8, 4))
    for method, value in values.items():
        ax.plot(grid, value, color=METHOD_COLORS[method], label=METHODS[method])
    ax.axhline(1, color="grey", ls="--", lw=1)
    ax.axvline(50, color="grey", ls="--", lw=1)
    ax.set(xlabel="Team possession (%)", ylabel="Multiplier applied to observed count",
           title=f"{n.LABELS[stat]}: how much does each method change the count?")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig
