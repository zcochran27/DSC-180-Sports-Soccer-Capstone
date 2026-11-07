"""Shared plotting and summary helpers for the reconstruction-validation notebooks.

Colors follow the validated default categorical palette (dataviz skill): the raw low-minute baseline is a
neutral gray. There are 11 methods but only 6 hues, so a hue is shared only by methods that are never drawn in the
same chart (the comparison chart uses raw, shrink, ridge, ridge_kalman, knn, ot_group and gru, all distinct).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from low_minute_player_analysis import checkpoints, reconstruction_eval as ev

METHOD_ORDER = ["raw", "shrink", "ridge", "knn", "ot_1d", "ot_joint", "ot_group",
                "kalman", "kalman_blend", "ridge_kalman", "gru"]
CURVE_METHODS = ["raw", "shrink", "ridge_kalman", "ridge", "knn", "ot_group", "gru"]  # drawn in the comparison chart (final ridge on top of its near-identical variant)
LABELS = {"raw": "Raw (no adjustment)", "shrink": "Minutes shrinkage", "ridge": "Ridge regression (final)",
          "knn": "Similar-player KNN", "ot_1d": "OT: per-stat quantiles", "ot_joint": "OT: whole vector",
          "ot_group": "OT: per stat group", "kalman": "Kalman (state space)",
          "kalman_blend": "Kalman + observed blend", "ridge_kalman": "Ridge + Kalman", "gru": "GRU sequence model"}
COLORS = {"raw": "#898781", "shrink": "#2a78d6", "ridge": "#eb6834", "knn": "#1baf7a",
          "ot_1d": "#eda100", "ot_joint": "#e87ba4", "ot_group": "#008300",
          "kalman": "#1baf7a", "kalman_blend": "#eda100", "ridge_kalman": "#e87ba4", "gru": "#eda100"}
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "semibold", "axes.titlesize": 11,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 9,
        "font.family": "sans-serif", "font.size": 10, "figure.dpi": 110,
    })


# --- loading and slicing ---

def load_results(tag="results"):
    return pd.read_parquet(checkpoints.RESULTS_DIR / f"{tag}.parquet")


def volume(r):
    return r[r["stat_type"] == "volume"]


def rates(r):
    return r[r["stat_type"].str.startswith("rate")]


def describe(r):
    print(f"{len(r):,} scored rows | genders {sorted(r.gender.unique())} | {r.group.nunique()} groups | "
          f"checkpoints {sorted(r.checkpoint.unique())} | methods {[m for m in METHOD_ORDER if m in set(r.method)]}")
    print(f"volume stats scored: {volume(r).stat.nunique()} | rebuilt rates scored: {rates(r).stat.nunique()}")


# --- summaries ---

def median_nrmse(df, methods, by="checkpoint"):
    """Median normalized RMSE over stats (rows = `by`, columns = method). Lower is better."""
    t = df[df["method"].isin(methods)].groupby([by, "method"] if isinstance(by, str) else [*by, "method"])["nrmse"].median()
    return t.unstack()[[m for m in methods if m in set(df["method"])]]


def improvement(df):
    """Per-stat relative error reduction vs the raw baseline: imp = 1 - nrmse / nrmse_raw."""
    key = ["gender", "group", "checkpoint", "stat"]
    raw = df[df["method"] == "raw"][key + ["nrmse"]].rename(columns={"nrmse": "raw_nrmse"})
    out = df.merge(raw, on=key)
    out["imp"] = (1 - out["nrmse"] / out["raw_nrmse"]).replace([np.inf, -np.inf], np.nan)
    return out


def improvement_table(df, methods):
    imp = improvement(df)
    t = imp[imp["method"].isin(methods)].groupby(["checkpoint", "method"])["imp"].median().unstack()
    return t[[m for m in methods if m in t.columns]]


def improvement_by_group(df, method):
    imp = improvement(df)
    imp = imp[imp["method"] == method]
    return imp.groupby(["group", "checkpoint"])["imp"].median().unstack()


def per_stat(df, method, cp, n=8):
    imp = improvement(df)
    s = imp[(imp["method"] == method) & (imp["checkpoint"] == cp)].groupby("stat")["imp"].median().sort_values()
    return s.tail(n)[::-1], s.head(n)


def best_method_share(df, methods):
    d = df[df["method"].isin(methods)]
    idx = d.groupby(["gender", "group", "checkpoint", "stat"])["nrmse"].idxmin().dropna()
    b = d.loc[idx].groupby(["checkpoint", "method"]).size().unstack().fillna(0)
    b = b.div(b.sum(axis=1), axis=0)
    return b[[m for m in methods if m in b.columns]]


def rate_table(rt, methods, cp=None):
    d = rt if cp is None else rt[rt["checkpoint"] == cp]
    t = d[d["method"].isin(methods)].groupby(["stat_type", "method"])["nrmse"].median().unstack()
    return t[[m for m in methods if m in t.columns]]


def missing_table(rt, methods):
    """Share of rates that exist for the truth but cannot be computed from the estimate."""
    m = rt[rt["method"].isin(methods)].groupby("method")[["n_truth_ok", "n_est_missing", "n_compared"]].sum()
    m["est_missing_share"] = m["n_est_missing"] / m["n_truth_ok"]
    return m.loc[[x for x in methods if x in m.index]]


# --- per-stat error ---

def stat_table(df, methods, cp, group=None, gender=None):
    """Normalized RMSE per stat (rows) and method (columns). Median over groups/genders unless one is given."""
    d = df[(df["checkpoint"] == cp) & df["method"].isin(methods)]
    if group:
        d = d[d["group"] == group]
    if gender:
        d = d[d["gender"] == gender]
    t = d.groupby(["stat", "method"])["nrmse"].median().unstack()
    return t[[m for m in methods if m in t.columns]]


def stat_rmse_units(df, methods, cp, group, gender):
    """RMSE in the stat's own units (per 90, or % for percents) for one group, plus the truth's std."""
    d = df[(df["checkpoint"] == cp) & (df["group"] == group) & (df["gender"] == gender) & df["method"].isin(methods)]
    t = d.pivot(index="stat", columns="method", values="rmse")[[m for m in methods if m in set(d["method"])]]
    base = d[d["method"] == "raw"].set_index("stat")
    t.insert(0, "truth_sd", base["rmse"] / base["nrmse"])
    return t


def best_per_stat(table):
    """Lowest nrmse across methods for each stat, and which method achieves it."""
    out = pd.DataFrame({"best_nrmse": table.min(axis=1), "best_method": table.idxmin(axis=1)})
    return out.sort_values("best_nrmse")


def heatmap(table, title, vmax=1.4):
    """Stats x methods heatmap of nrmse. Darker = larger error; the lowest cell in each row is bold."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    t = table.dropna(how="all")
    t = t.loc[t.min(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(2.0 + 0.95 * t.shape[1], 0.3 * len(t) + 1.3))
    im = ax.imshow(t.to_numpy(), cmap=cmap, vmin=0.2, vmax=vmax, aspect="auto")
    ax.set_xticks(range(t.shape[1]), t.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(t)), t.index, fontsize=8.5)
    ax.xaxis.tick_top()
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    best = np.nanargmin(t.to_numpy(), axis=1)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.iat[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if (v - 0.2) / (vmax - 0.2) > 0.55 else INK,
                    fontweight="bold" if j == best[i] else "normal")
    cb = fig.colorbar(im, ax=ax, shrink=min(1.0, 8 / max(len(t) * 0.3, 1)), pad=0.02)
    cb.set_label("normalized RMSE (darker = worse)", fontsize=8.5)
    ax.set_title(title, loc="left", pad=28)
    return fig


# --- plots ---

def curve_plot(summary, methods, title, ylabel="Median normalized RMSE",
               ax=None):
    """Error vs minutes at checkpoint, one line per method (legend always on, end values if <=3 lines)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 3.8))
    for i, m in enumerate(methods):
        # earlier methods get bigger markers so a series hidden behind a near-identical one stays visible
        ax.plot(summary.index, summary[m], marker="o", ms=max(5, 10 - i), lw=2, color=COLORS[m],
                label=LABELS[m], markeredgecolor=SURFACE, markeredgewidth=2, zorder=3 + i)
        if len(methods) <= 3:
            for x, v in summary[m].items():
                ax.annotate(f"{v:.2f}", (x, v), xytext=(0, 9), textcoords="offset points", ha="center",
                            color=INK2, fontsize=8.5)
    ax.set_xticks(list(summary.index))
    ax.set_xlabel("Minutes bin (labelled by its centre: 0-375, 375-625, 625-750)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)
    return ax


def group_bars(imp, method, title):
    """Median error reduction vs raw by group; shade encodes the checkpoint (light = 250, dark = 750)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    groups = list(imp.index)
    h = 0.26
    for i, cp in enumerate(imp.columns):
        ax.barh(np.arange(len(groups)) + (i - 1) * h, imp[cp] * 100, height=h - 0.04,
                color=COLORS[method], alpha=[0.35, 0.65, 1.0][i], label=f"{cp} min")
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(np.arange(len(groups)), groups)
    ax.invert_yaxis()
    ax.set_xlabel("Median error reduction vs raw (%)")
    ax.set_title(title, loc="left")
    ax.legend(title="Minutes bin", loc="lower right")
    return ax


# --- paired comparison versus the final ridge ---

def load_paired(tag="results"):
    return pd.read_parquet(checkpoints.RESULTS_DIR / f"{tag}_paired.parquet")


def paired_summary(p, methods):
    """Per method and minutes bin: mean paired difference vs the final ridge (negative = better) and the number
    of the 14 gender/group sets where the 95% interval is entirely below (better) or above (worse) 0."""
    d = p[p["method"].isin(methods)].copy()
    d["better"], d["worse"] = d["hi"] < 0, d["lo"] > 0
    g = d.groupby(["method", "checkpoint"]).agg(mean_diff=("mean_diff", "mean"), better=("better", "sum"),
                                                worse=("worse", "sum"), sets=("better", "size"))
    return g.loc[[m for m in methods if m in set(d["method"])]]


# --- one worked example, run live ---

def demo_group(gender, group, methods):
    """Run the out-of-fold pipeline for one gender/group. Returns (pool, stems, estimates)."""
    stems, folds = ev.group_setup(gender, group)
    pool = ev.load_pool(gender, group, stems)
    est = ev.oof_predictions(pool, folds, stems, methods)
    return pool, stems, est


def nrmse_per_stat(pool, est, b):
    """Weighted nrmse per stat in minutes bin `b` (each player-season counts once)."""
    sel = (pool["bin"] == b).to_numpy()
    w = pool["weight"].to_numpy()[sel]
    t = pool.loc[sel, [f"full_{s}" for s in est.columns]].to_numpy()
    mean = (t * w[:, None]).sum(0) / w.sum()
    sd = np.sqrt((((t - mean) ** 2) * w[:, None]).sum(0) / w.sum())
    rmse = np.sqrt((((est[sel].to_numpy() - t) ** 2) * w[:, None]).sum(0) / w.sum())
    return pd.Series(rmse / sd, index=est.columns)


def scatter_grid(pool, est, stats, b, methods):
    """Estimate (y) vs true full-season per-90 (x), one panel per stat and method. One point per player-season
    (its longest prefix inside minutes bin `b`)."""
    rows = pool[pool["bin"] == b].groupby(["player_id", "year"]).tail(1).index
    fig, axes = plt.subplots(len(stats), len(methods), figsize=(3.3 * len(methods), 3.0 * len(stats)),
                             squeeze=False)
    for i, s in enumerate(stats):
        truth = pool.loc[rows, f"full_{s}"].to_numpy()
        hi = max(truth.max(), *(est[m].loc[rows, s].max() for m in methods)) * 1.02
        for j, m in enumerate(methods):
            ax = axes[i, j]
            y = est[m].loc[rows, s].to_numpy()
            ax.scatter(truth, y, s=12, color=COLORS[m], alpha=0.4, edgecolors="none")
            ax.plot([0, hi], [0, hi], color=INK2, lw=1, ls=(0, (4, 3)))
            ax.set_xlim(0, hi), ax.set_ylim(min(0, y.min()), hi)
            nr = np.sqrt(np.mean((y - truth) ** 2)) / truth.std()
            ax.set_title(f"{s} - {LABELS[m]}\nnrmse {nr:.2f}", loc="left", fontsize=8.5)
            if i == len(stats) - 1:
                ax.set_xlabel("True full-season per 90")
            if j == 0:
                ax.set_ylabel("Estimate per 90")
    fig.tight_layout()
    return fig


# --- distribution plots ---

HIGH_FILL = SEQ_BLUE[1]   # high-minute (truth) players


def dist_summary(series):
    """Mean, sd, share exactly 0 and a few quantiles for named samples (rows = sample)."""
    rows = {}
    for name, x in series.items():
        x = pd.Series(np.asarray(x, dtype=float)).dropna()
        rows[name] = {"n": len(x), "mean": x.mean(), "sd": x.std(), "share_zero": (x == 0).mean(),
                      "p10": x.quantile(0.1), "median": x.median(), "p90": x.quantile(0.9)}
    return pd.DataFrame(rows).T


def overlay_hist(ax, high, other, other_label, other_color, bins=30, title="",
                 high_label="High-minute players (900+ min, full season)"):
    """High-minute distribution as a filled histogram, `other` as a step outline, shared bins and density scale."""
    high, other = np.asarray(high, float), np.asarray(other, float)
    hi = np.nanpercentile(np.concatenate([high, other]), 99.5)
    lo = min(0.0, np.nanmin(np.concatenate([high, other])))
    edges = np.linspace(lo, hi, bins + 1)
    ax.hist(np.clip(high, lo, hi), bins=edges, density=True, color=HIGH_FILL, alpha=0.9, label=high_label)
    ax.hist(np.clip(other, lo, hi), bins=edges, density=True, histtype="step", lw=2, color=other_color,
            label=other_label)
    ax.set_title(title, loc="left", fontsize=9.5)
    ax.set_yticks([])
    ax.grid(False)
    return ax
