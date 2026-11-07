"""Methods that read the sequence of matches: state-space (Kalman) shrinkage and a GRU.

Same signature and data layout as reconstruct_methods.py:  method(T, E, stems, cache) -> DataFrame.
T and E are whole player-seasons of per-game prefixes, sorted by season and match number and contiguous from
match 1. Per-match totals are recovered by differencing the running totals (low per-90 * minutes / 90).

Kalman (kalman, kalman_blend, ridge_kalman)
    Per stat, a latent rate "theta" follows a random walk over matches (variance delta2 per match) starting
    from a prior N(mu, tau2). A match with m minutes is observed as y ~ N(theta, sigma2 / m) in per-90 units.
    (mu, tau2, delta2, sigma2) are fitted by Normal maximum likelihood on the training sequences.
    delta2 -> 0 gives back the shrinkage estimator, so a fitted delta2 > 0 measures drift in form within a season.
      kalman        estimate = filtered theta after the matches seen so far.
      kalman_blend  the full-season per-90 contains the m minutes already seen, so
                    estimate = (m / Mbar) * observed per-90 + (1 - m / Mbar) * theta, where Mbar = mean training
                    full-season minutes (a modelling choice; `kalman` is the version comparable to `shrink`).
      ridge_kalman  the ridge features plus the Kalman estimate (and its product with s(m)).

GRU (gru)
    A causal GRU over the matches of a season. Step input: minutes-normalised running per-90 z, per-match
    per-90 z, log match minutes, log running minutes, step / 10. Output at every step: the full-season
    per-90 as a z-score, plus a linear skip from the running z. Every prefix of every training season is a
    training example (weights 1 / rows per player-season per bin, as for ridge). Early stopping on 10% of
    the training players.
"""
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from torch import nn

from low_minute_player_analysis import distribution_norm
from low_minute_player_analysis import reconstruct_methods as rm
from low_minute_player_analysis.reconstruct_methods import clip_estimates, get_full, get_low

KEYS = ["player_id", "year"]


# ---------------------------------------------------------------- turn prefix rows into per-match sequences

def build_seq(df, stems):
    """Padded arrays with one row per season and one column per match, built from prefix rows.

    Rows of df must be sorted by season and match number. Returns a dict with
        sid, k        season number and match index of each row of df
        mask          [seasons, matches]  True where the season has that match
        cum_min       running minutes after each match
        match_min     minutes played in each match
        match_tot     per-match totals of each stat       [seasons, matches, stats]
        match_p90     per-match per-90 of each stat       [seasons, matches, stats]
        S, K, P       number of seasons, max matches, stats
    """
    season_id = df.groupby(KEYS, sort=False).ngroup().to_numpy()
    match_index = df["n_matches"].to_numpy() - 1
    S, K, P = int(season_id.max()) + 1, int(match_index.max()) + 1, len(stems)

    mask = np.zeros((S, K), bool)
    mask[season_id, match_index] = True
    assert (np.cumprod(mask, axis=1).sum(1) == mask.sum(1)).all(), "prefix rows must be contiguous from match 1"

    # running minutes and running totals after each match
    cum_min = np.zeros((S, K))
    cum_min[season_id, match_index] = df["actual_minutes"].to_numpy()
    cum_total = np.zeros((S, K, P))
    cum_total[season_id, match_index] = get_low(df, stems).to_numpy() * df["actual_minutes"].to_numpy()[:, None] / 90.0

    # difference the running values to get what happened in each single match
    match_min = np.where(mask, np.diff(cum_min, axis=1, prepend=0.0), 0.0)
    match_tot = np.where(mask[:, :, None], np.diff(cum_total, axis=1, prepend=0.0), 0.0)
    safe_minutes = np.where(match_min > 0, match_min, 1.0)
    match_p90 = np.where(mask[:, :, None], 90.0 * match_tot / safe_minutes[:, :, None], 0.0)

    return dict(sid=season_id, k=match_index, mask=mask, cum_min=cum_min, match_min=match_min,
                match_tot=match_tot, match_p90=match_p90, S=S, K=K, P=P)


# ---------------------------------------------------------------- Kalman filter

def kalman_filter(y, m, mask, mu, tau2, delta2, sigma2):
    """Run the filter over all seasons at once, one match at a time.

    y: per-match per-90 [S, K]; m: match minutes [S, K]; mask: which matches exist.
    Returns (theta [S, K] = filtered rate after each match, total log-likelihood).
    """
    S, K = y.shape
    estimate = np.full(S, mu)       # current estimate of the latent rate
    variance = np.full(S, tau2)     # uncertainty of that estimate
    theta = np.zeros((S, K))
    log_likelihood = 0.0

    for j in range(K):
        if j > 0:
            variance = variance + delta2           # the rate may drift between matches
        valid = mask[:, j]
        noise = sigma2 / np.where(valid, m[:, j], 1.0)   # a match with few minutes is a noisier observation
        total_variance = variance + noise
        residual = y[:, j] - estimate
        gain = variance / total_variance
        log_likelihood += np.where(valid, -0.5 * (np.log(2 * np.pi * total_variance)
                                                  + residual ** 2 / total_variance), 0.0).sum()
        estimate = np.where(valid, estimate + gain * residual, estimate)
        variance = np.where(valid, (1 - gain) * variance, variance)
        theta[:, j] = estimate

    return theta, log_likelihood


def kalman_negative_log_likelihood(log_params, y, m, mask, mu):
    """What the optimiser minimises: minus the log-likelihood as a function of log(tau2, delta2, sigma2)."""
    tau2, delta2, sigma2 = np.exp(np.clip(log_params, -30, 30))
    return -kalman_filter(y, m, mask, mu, tau2, delta2, sigma2)[1]


def fit_kalman(T, stems):
    """Fit the Kalman parameters of every stat. Returns an array (4, number of stats): mu, tau2, delta2, sigma2."""
    seq = build_seq(T, stems)
    match_minutes, mask = seq["match_min"], seq["mask"]
    total_minutes = match_minutes.sum()
    season_minutes = match_minutes.sum(1)

    params = np.zeros((4, len(stems)))
    for i in range(len(stems)):
        y = seq["match_p90"][:, :, i]
        mu = float(seq["match_tot"][:, :, i].sum() / total_minutes * 90.0)   # overall mean per 90
        season_p90 = 90.0 * seq["match_tot"][:, :, i].sum(1) / season_minutes
        tau2_start = max(float(np.var(season_p90)) / 2, 1e-8)
        start = np.log([tau2_start, tau2_start * 0.05, tau2_start * 300.0])

        fit = minimize(kalman_negative_log_likelihood, start, args=(y, match_minutes, mask, mu),
                       method="Nelder-Mead", options=dict(maxiter=250, xatol=1e-2, fatol=1e-2))
        tau2, delta2, sigma2 = np.exp(np.clip(fit.x, -30, 30))
        params[:, i] = [mu, tau2, delta2, sigma2]
    return params


def kalman_theta(df, stems, params):
    """Filtered rate after the matches seen so far, for each row of df -> array (rows, stats)."""
    seq = build_seq(df, stems)
    theta_per_row = np.zeros((len(df), len(stems)))
    for i in range(len(stems)):
        mu, tau2, delta2, sigma2 = params[:, i]
        theta, _ = kalman_filter(seq["match_p90"][:, :, i], seq["match_min"], seq["mask"], mu, tau2, delta2, sigma2)
        theta_per_row[:, i] = theta[seq["sid"], seq["k"]]
    return theta_per_row


def get_kalman_params(T, stems, cache):
    """Fit the Kalman parameters once per fold and reuse them."""
    if "kalman_params" not in cache:
        cache["kalman_params"] = fit_kalman(T, stems)
    return cache["kalman_params"]


def kalman(T, E, stems, cache):
    theta = kalman_theta(E, stems, get_kalman_params(T, stems, cache))
    return clip_estimates(pd.DataFrame(theta, index=E.index, columns=stems))


def kalman_blend(T, E, stems, cache):
    theta = kalman_theta(E, stems, get_kalman_params(T, stems, cache))
    mean_full_minutes = T.drop_duplicates(KEYS)["full_minutes"].mean()
    weight_on_observed = np.minimum(E["actual_minutes"].to_numpy() / mean_full_minutes, 1.0)[:, None]
    estimate = weight_on_observed * get_low(E, stems).to_numpy() + (1 - weight_on_observed) * theta
    return clip_estimates(pd.DataFrame(estimate, index=E.index, columns=stems))


# ---------------------------------------------------------------- ridge + Kalman

def ridge_kalman_features(df, stems, norm, theta, mu):
    """The ridge features plus the Kalman estimate (centred on the group mean) and its product with s(m)."""
    s = np.sqrt(rm.REF_MINUTES / df["actual_minutes"].to_numpy())[:, None]
    centred_theta = theta - mu
    return np.hstack([rm.ridge_features(df, stems, norm), centred_theta, centred_theta * s])


def ridge_kalman(T, E, stems, cache):
    params = get_kalman_params(T, stems, cache)
    mu = params[0]

    if "ridge_kalman" not in cache:
        norm = distribution_norm.fit_minutes_norm(get_low(T, stems), T["actual_minutes"])
        features = ridge_kalman_features(T, stems, norm, kalman_theta(T, stems, params), mu)
        scaler = StandardScaler().fit(features)
        model = RidgeCV(alphas=rm.RIDGE_ALPHAS, alpha_per_target=True)
        model.fit(scaler.transform(features), get_full(T, stems).to_numpy(), sample_weight=T["weight"].to_numpy())
        cache["ridge_kalman"] = (norm, scaler, model)

    norm, scaler, model = cache["ridge_kalman"]
    features = ridge_kalman_features(E, stems, norm, kalman_theta(E, stems, params), mu)
    prediction = model.predict(scaler.transform(features))
    return clip_estimates(pd.DataFrame(prediction, index=E.index, columns=stems))


# ---------------------------------------------------------------- GRU

GRU_HIDDEN = 96            # size of the hidden state
GRU_DROPOUT = 0.2
GRU_LR = 2e-3              # learning rate
GRU_WD = 1e-2              # weight decay
GRU_BATCH = 128            # seasons per training step
GRU_MAX_EPOCHS = 60
GRU_PATIENCE = 6           # stop after this many epochs without improvement on the validation players
GRU_VAL_SHARE = 0.1        # share of the training players held out for early stopping
Z_CLIP_CUM, Z_CLIP_MATCH = 6.0, 5.0   # z-scores are clipped to this range so outliers cannot blow up training


def make_gru_inputs(df, stems, prep):
    """Network input [seasons, matches, 2 * stats + 3] (float32) and the sequence dict, using the fitted `prep`."""
    seq = build_seq(df, stems)
    S, K, P = seq["S"], seq["K"], len(stems)

    # running per-90 after each match, as a z-score for the minutes played so far
    cum_p90 = np.zeros((S, K, P))
    cum_p90[seq["sid"], seq["k"]] = get_low(df, stems).to_numpy()
    cum_minutes = seq["cum_min"]
    z_running = distribution_norm.minutes_z(prep["norm"], cum_p90.reshape(-1, P),
                                            np.maximum(cum_minutes.reshape(-1), 1.0)).reshape(S, K, P)
    z_running = np.clip(z_running, -Z_CLIP_CUM, Z_CLIP_CUM)

    # per-match per-90 as a z-score
    z_match = (seq["match_p90"] - prep["match_mu"]) / prep["match_sd"]
    z_match = np.clip(z_match, -Z_CLIP_MATCH, Z_CLIP_MATCH)

    step = np.broadcast_to(np.arange(K)[None, :], (S, K)) / 10.0
    extras = np.stack([np.log1p(seq["match_min"]), np.log1p(cum_minutes), step], axis=2)

    x = np.concatenate([z_running, z_match, extras], axis=2) * seq["mask"][:, :, None]
    return x.astype(np.float32), seq


def fit_gru_prep(T, stems):
    """Everything the network needs to scale its inputs and outputs, computed from the training rows."""
    low = get_low(T, stems)
    seq = build_seq(T, stems)

    # mean and sd of the per-match per-90, using matches of at least 20 minutes
    long_enough = (seq["match_min"] >= 20)[:, :, None].astype(float)
    n_long = np.maximum(long_enough.sum((0, 1)), 1)
    match_mu = (seq["match_p90"] * long_enough).sum((0, 1)) / n_long
    match_sd = np.sqrt((((seq["match_p90"] - match_mu) ** 2) * long_enough).sum((0, 1)) / n_long)
    match_sd[match_sd < 1e-9] = 1.0

    # mean and sd of the full-season targets
    full = get_full(T.drop_duplicates(KEYS), stems)
    return dict(norm=distribution_norm.fit_minutes_norm(low, T["actual_minutes"]),
                match_mu=match_mu, match_sd=match_sd,
                y_mu=full.mean().to_numpy(), y_sd=full.std().replace(0, 1).to_numpy())


def build_gru_layers(n_stats):
    """The layers of the network, in a ModuleDict (no custom class needed)."""
    return nn.ModuleDict({
        "inp": nn.Linear(2 * n_stats + 3, GRU_HIDDEN),
        "gru": nn.GRU(GRU_HIDDEN, GRU_HIDDEN, batch_first=True),
        "drop": nn.Dropout(GRU_DROPOUT),
        "head": nn.Linear(GRU_HIDDEN, n_stats),
        "skip": nn.Linear(n_stats, n_stats),
    })


def gru_forward(layers, x, n_stats):
    """Network output [seasons, matches, stats]: GRU path plus a linear skip from the running z-scores."""
    hidden, _ = layers["gru"](torch.relu(layers["inp"](x)))
    return layers["head"](layers["drop"](hidden)) + layers["skip"](x[..., :n_stats])


def gru_loss(layers, x, y, weight, rows, n_stats):
    """Weighted mean squared error over the seasons in `rows`, averaged over matches and stats."""
    prediction = gru_forward(layers, x[rows], n_stats)
    squared_error = ((prediction - y[rows][:, None, :]) ** 2).mean(-1)
    return (squared_error * weight[rows]).sum() / weight[rows].sum()


def fit_gru(T, stems):
    """Train the GRU on the training rows. Returns (layers, prep)."""
    torch.set_num_threads(2)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n_stats = len(stems)

    prep = fit_gru_prep(T, stems)
    x, seq = make_gru_inputs(T, stems, prep)
    S, K = seq["S"], seq["K"]

    # one target per season (its full-season per-90 as a z-score) and one weight per prefix
    first_row_of_season = T.groupby(KEYS, sort=False).head(1)
    y_season = (get_full(first_row_of_season, stems).to_numpy() - prep["y_mu"]) / prep["y_sd"]
    weights = np.zeros((S, K))
    weights[seq["sid"], seq["k"]] = T["weight"].to_numpy()

    # hold out some players for early stopping
    player_of_season = np.zeros(S, dtype=object)
    player_of_season[seq["sid"]] = T["player_id"].to_numpy()
    players = np.unique(player_of_season.astype(str))
    n_validation = max(1, int(len(players) * GRU_VAL_SHARE))
    validation_players = set(rng.choice(players, n_validation, replace=False))
    is_validation = np.array([str(p) in validation_players for p in player_of_season])
    train_seasons = np.where(~is_validation)[0]
    validation_seasons = np.where(is_validation)[0]

    layers = build_gru_layers(n_stats)
    optimizer = torch.optim.AdamW(layers.parameters(), lr=GRU_LR, weight_decay=GRU_WD)
    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y_season.astype(np.float32))
    w_t = torch.from_numpy(weights.astype(np.float32))

    best_loss, best_state, epochs_without_gain = np.inf, None, 0
    for _ in range(GRU_MAX_EPOCHS):
        layers.train()
        shuffled = rng.permutation(train_seasons)
        for start in range(0, len(shuffled), GRU_BATCH):
            batch = torch.from_numpy(shuffled[start:start + GRU_BATCH])
            optimizer.zero_grad()
            gru_loss(layers, x_t, y_t, w_t, batch, n_stats).backward()
            optimizer.step()

        layers.eval()
        with torch.no_grad():
            validation_loss = float(gru_loss(layers, x_t, y_t, w_t, torch.from_numpy(validation_seasons), n_stats))
        if validation_loss < best_loss - 1e-5:
            best_loss, epochs_without_gain = validation_loss, 0
            best_state = {name: value.clone() for name, value in layers.state_dict().items()}
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= GRU_PATIENCE:
                break

    layers.load_state_dict(best_state)
    layers.eval()
    return layers, prep


def gru(T, E, stems, cache):
    if "gru" not in cache:
        cache["gru"] = fit_gru(T, stems)
    layers, prep = cache["gru"]

    x, seq = make_gru_inputs(E, stems, prep)
    with torch.no_grad():
        output = gru_forward(layers, torch.from_numpy(x), len(stems)).numpy()
    z = output[seq["sid"], seq["k"]]
    estimate = z * prep["y_sd"] + prep["y_mu"]
    return clip_estimates(pd.DataFrame(estimate, index=E.index, columns=stems))


METHODS = {"kalman": kalman, "kalman_blend": kalman_blend, "ridge_kalman": ridge_kalman, "gru": gru}
