"""Engineered ("derived") stats built from the rolled-up total_* columns.

Every stat here is computed in wyscout.rollup *after* totals are summed across competitions, so
ratios are ratios of sums (a 5-pass match never weighs the same as a 700-pass match). Nothing
is fetched from the API; each stat is a formula over columns that already exist.

Column naming (the prefix tells you what the number is):
  total_<name>     summed count          (only for count-type stats, e.g. total_np_xg)
  average_<name>   total_<name> / exposure - same basis as the API's average_* columns, which is
                   "per exposure" (roughly per match scaled by tagged time), NOT strictly per 90.
                   Relative (opponent) stats divide by n_rel_matches instead: plain per match.
  percent_<name>   100 * numerator / denominator   (shares and rates)
  ratio_<name>     numerator / denominator          (ratios that are not 0-100 percentages)

Ratios are NULL when the summed denominator is below the entry's min_den (default 10), so tiny
samples don't top leaderboards. Every stat's formula, group and note is also written to
wyscout.stat_formulas (origin = 'derived'), and `python -m wyscout.derived --write-docs`
regenerates wyscout/DERIVED_STATS.md from this registry.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PENALTY_XG = 0.76   # xg_shot includes penalties; a lone penalty is worth exactly 0.76 xG
MIN_DEN = 10        # default minimum summed denominator before a rate is reported

# Group names, in display order.
GROUPS = ["Shooting", "Chance Creation", "Passing", "Progression", "Dribbling", "Duels", "Aerial",
          "Defending", "Pressing & Recovery", "Ball Security", "Discipline", "Goalkeeping",
          "Results", "Territory & Dominance", "Playing Time"]

_TERM = r"\s*([+-]?)\s*(?:(\d+(?:\.\d+)?)\s*\*\s*)?([a-z][a-z0-9_]*)"


def parse_terms(expr: str) -> tuple[tuple[float, str], ...]:
    """'xg_shot - 0.76*penalties' -> ((1, 'xg_shot'), (-0.76, 'penalties')). Stems exclude total_."""
    if not re.fullmatch(rf"(?:{_TERM})+\s*", expr):
        raise ValueError(f"bad formula: {expr!r}")
    return tuple(((-1.0 if sign == "-" else 1.0) * (float(coef) if coef else 1.0), stem)
                 for sign, coef, stem in re.findall(_TERM, expr))


def render_terms(terms, prefix: str = "total_") -> str:
    out = []
    for i, (c, stem) in enumerate(terms):
        mag = abs(c)
        body = f"{prefix}{stem}" if mag == 1 else f"{mag:g}*{prefix}{stem}"
        out.append((("-" if c < 0 else "") if i == 0 else (" - " if c < 0 else " + ")) + body)
    return "".join(out)


@dataclass(frozen=True)
class Derived:
    name: str
    kind: str                 # count | percent | ratio
    groups: tuple[str, ...]
    scope: str                # team | player | both
    num: tuple
    den: tuple = ()
    min_den: float = MIN_DEN
    note: str = ""
    origin: str = "totals"    # totals (rollup totals) | match (team vs opponent, from team_match_stats)

    @property
    def per(self) -> str:
        return "n_rel_matches" if self.origin == "match" else "exposure"


def _d(name, kind, groups, scope, num, den="", min_den=MIN_DEN, note="", origin="totals"):
    g = (groups,) if isinstance(groups, str) else tuple(groups)
    return Derived(name, kind, g, scope, parse_terms(num), parse_terms(den) if den else (),
                   min_den, note, origin)


def count(name, groups, scope, num, note="", origin="totals"):
    return _d(name, "count", groups, scope, num, note=note, origin=origin)


def pct(name, groups, scope, num, den, min_den=MIN_DEN, note="", origin="totals"):
    return _d(name, "percent", groups, scope, num, den, min_den, note, origin)


def ratio(name, groups, scope, num, den, min_den=MIN_DEN, note="", origin="totals"):
    return _d(name, "ratio", groups, scope, num, den, min_den, note, origin)


DIR_NOTE = ("Denominator is forward + lateral + back (the ~86% of passes Wyscout gives a direction), "
            "so the three direction shares sum to 100%.")
OVERLAP = "Categories overlap (a pass can be several), so this is an index, not a strict share of unique passes."

REGISTRY: list[Derived] = [
    # ---- Shooting ----
    count("np_xg", "Shooting", "both", f"xg_shot - {PENALTY_XG}*penalties",
          "Non-penalty xG. xg_shot includes penalties and a lone penalty is exactly 0.76 xG (2,051 single-penalty rows)."),
    count("np_goals", "Shooting", "both", "goals - successful_penalties", "Goals excluding converted penalties."),
    count("goals_minus_xg", "Shooting", "both", "goals - xg_shot",
          "Finishing over (+) / under (-) expectation. Includes penalties on both sides."),
    ratio("xg_per_shot", "Shooting", "both", "xg_shot", "shots", note="Average shot quality."),
    ratio("goals_per_xg", "Shooting", "both", "goals", "xg_shot", min_den=1,
          note="Goals scored per xG; 1.0 = finishing exactly to expectation. Denominator is xG, so min_den is 1."),
    ratio("shots_per_box_touch", "Shooting", "both", "shots", "touch_in_box",
          note="Not a percentage: only ~80% of rows have shots <= box touches (shots come from outside the box)."),
    pct("head_shot_share", ["Shooting", "Aerial"], "both", "head_shots", "shots"),
    pct("penalty_goal_share", "Shooting", "both", "successful_penalties", "goals", note="Share of goals from penalties."),
    pct("shot_share_of_attack", "Shooting", "both", "shots", "attacking_actions"),
    # ---- Chance creation ----
    count("xa_minus_assists", "Chance Creation", "player", "xg_assist - assists",
          "Positive = teammates finished fewer of the player's chances than expected."),
    pct("key_pass_rate", "Chance Creation", "both", "key_passes", "passes"),
    pct("shot_assist_rate", "Chance Creation", "both", "shot_assists", "passes"),
    pct("assist_per_key_pass", "Chance Creation", "both", "assists", "key_passes",
          note="assists <= key_passes holds in ~98.6% of season rows; chain is roughly assists < key passes < shot assists."),
    pct("cross_share", "Chance Creation", "both", "crosses", "passes"),
    pct("risky_pass_index", ["Chance Creation", "Passing"], "both", "through_passes + smart_passes + key_passes", "passes",
        note=OVERLAP),
    ratio("box_touch_per_final_third_pass", ["Chance Creation", "Progression"], "both", "touch_in_box",
          "passes_to_final_third", note="Box touches per pass into the final third; can exceed 1."),
    # ---- Passing / progression ----
    pct("forward_pass_share", ["Passing", "Progression"], "both", "forward_passes",
        "forward_passes + lateral_passes + back_passes", note=DIR_NOTE),
    pct("lateral_pass_share", "Passing", "both", "lateral_passes",
        "forward_passes + lateral_passes + back_passes", note=DIR_NOTE),
    pct("back_pass_share", "Passing", "both", "back_passes",
        "forward_passes + lateral_passes + back_passes", note=DIR_NOTE),
    pct("long_pass_share", "Passing", "both", "long_passes", "passes"),
    pct("vertical_pass_share", ["Passing", "Progression"], "both", "vertical_passes", "passes",
        note="Vertical passes are NOT nested inside forward passes (only ~65% of rows), so don't read this as a subset of forward share."),
    pct("final_third_pass_share", "Progression", "both", "passes_to_final_third", "passes"),
    pct("directness_index", "Progression", "both", "forward_passes + long_passes + through_passes", "passes", note=OVERLAP),
    pct("progressive_carry_share", ["Progression", "Dribbling"], "player", "progressive_run",
        "progressive_run + progressive_passes", note="Carrying vs passing balance of progression. Team data has no progressive_passes."),
    # ---- Dribbling / duels ----
    pct("take_on_rate", "Dribbling", "both", "dribbles", "dribbles + passes",
        note="Dribbles as a share of dribbles + passes (dribbles and passes are separate action types)."),
    pct("dribble_share_of_off_duels", ["Dribbling", "Duels"], "both", "dribbles", "offensive_duels",
        note="Dribbles are a subset of offensive duels in 100% of rows."),
    count("failed_dribbles", "Dribbling", "both", "dribbles - successful_dribbles",
          "Already part of failed_offensive_duels - never add the two together."),
    pct("aerial_duel_share", ["Duels", "Aerial"], "both", "aerial_duels", "duels",
        note="aerial_duels includes goalkeeper aerials, which is what makes duels = aerial + defensive + offensive + loose-ball exact."),
    pct("defensive_duel_share", ["Duels", "Defending"], "both", "defensive_duels", "duels"),
    pct("offensive_duel_share", ["Duels", "Dribbling"], "both", "offensive_duels", "duels"),
    pct("loose_ball_duel_share", "Duels", "both", "loose_ball_duels", "duels"),
    # ---- Defending / pressing ----
    ratio("pressing_duels_per_def_duel", ["Defending", "Pressing & Recovery"], "both", "pressing_duels", "defensive_duels",
          note="A ratio, not a percent: pressing_duels <= defensive_duels in only ~97.5% of rows, and the summed ratio exceeds 1 for some players (max ~5)."),
    pct("interception_share", "Defending", "both", "interceptions", "defensive_actions"),
    pct("sliding_tackle_share", "Defending", "player", "sliding_tackles", "defensive_actions"),
    ratio("fouls_per_defensive_duel", ["Defending", "Discipline"], "both", "fouls", "defensive_duels",
          note="How reckless the defending is. A ratio, not a percent: fouls are not all committed inside defensive duels (player values exceed 1)."),
    pct("high_recovery_share", "Pressing & Recovery", "both", "opponent_half_recoveries", "recoveries",
        note="How high up the pitch the ball is won."),
    pct("dangerous_recovery_share", "Pressing & Recovery", "both", "dangerous_opponent_half_recoveries", "recoveries"),
    ratio("fouls_won_vs_committed", ["Discipline", "Defending"], "player", "fouls_suffered", "fouls"),
    ratio("shot_quality_conceded", "Defending", "team", "xg_shot_against", "shots_against", note="xG per shot faced."),
    ratio("shots_against_per_def_action", "Defending", "team", "shots_against", "defensive_actions"),
    ratio("block_clear_ratio", "Defending", "player", "shots_blocked", "clearances"),
    count("goals_conceded_minus_xga", ["Defending", "Goalkeeping"], "team", "conceded_goals - xg_shot_against",
          "Positive = conceding more than the shots faced suggest (defence/goalkeeping underperforming)."),
    # ---- Ball security ----
    count("failed_passes", ["Passing", "Ball Security"], "both", "passes - successful_passes"),
    count("failed_offensive_duels", ["Dribbling", "Ball Security"], "both", "offensive_duels - offensive_duels_won",
          "Includes failed dribbles. failed_passes + failed_offensive_duels is within ~2% of the API's losses column, "
          "so no separate 'giveaways' stat is built - use losses."),
    pct("loss_rate", "Ball Security", "both", "losses", "passes + offensive_duels", min_den=30,
        note="Turnovers per pass + offensive duel. losses ~ failed passes + failed offensive duels in aggregate, but not per player, "
             "so very low-volume players can exceed 100 - hence min_den 30."),
    pct("retention_rate", "Ball Security", "both", "passes + offensive_duels - losses", "passes + offensive_duels", min_den=30,
        note="100 - loss_rate (same caveat, can go negative for very low-volume players)."),
    pct("loss_rate_per_touch", "Ball Security", "both", "losses", "passes + received_pass + offensive_duels", min_den=30),
    pct("own_half_loss_share", "Ball Security", "both", "own_half_losses", "losses"),
    pct("dangerous_loss_share", "Ball Security", "both", "dangerous_own_half_losses", "own_half_losses",
        note="dangerous_own_half_losses is a subset of own_half_losses in 100% of rows."),
    # ---- Discipline ----
    count("card_weight", "Discipline", "both", "yellow_cards + 2*red_cards",
          "Yellow = 1, red = 2. Unknown whether yellow_cards already includes the first yellow of second-yellow dismissals."),
    pct("cards_per_foul", "Discipline", "both", "yellow_cards + red_cards", "fouls"),
    # ---- Goalkeeping ----
    pct("gk_long_kick_share", "Goalkeeping", "team", "long_goal_kicks", "long_goal_kicks + short_goal_kicks",
        note="Team long + short does not equal gk_goal_kicks (107%), so long + short is the denominator."),
    pct("gk_long_kick_share", "Goalkeeping", "player", "goal_kicks_long", "goal_kicks",
        note="Player goal_kicks_long + goal_kicks_short matches goal_kicks to within 0.1%."),
    count("goals_prevented", "Goalkeeping", "player", "xg_save - gk_conceded_goals",
          "Reads xg_save as post-shot xG faced (27.3k vs 27.9k conceded league-wide) - an inference, not a documented definition."),
    ratio("gk_exits_per_shot_on_target_faced", "Goalkeeping", "player", "gk_exits", "gk_shots_against",
          note="gk_shots_against is shots ON TARGET faced (gk_saves + gk_conceded_goals = gk_shots_against in 100% of rows)."),
    # ---- Playing time ----
    pct("starter_rate", "Playing Time", "player", "matches_in_start", "matches", min_den=1),
    pct("sub_off_rate", "Playing Time", "player", "matches_coming_off", "matches", min_den=1),
    ratio("minutes_per_appearance", "Playing Time", "player", "minutes_on_field", "matches", min_den=1),
]

# ---- Team vs opponent, built from team_match_stats (exactly 2 rows per match) ----
REL_STEMS = ["xg_shot", "shots", "passes", "passes_to_final_third", "touch_in_box", "recoveries"]
REL_NOTE = "Built from team_match_stats (own row vs the opponent row of the same match); per-match average, not per exposure."
RELATIVE: list[Derived] = [
    count("xg_diff", "Territory & Dominance", "team", "own_xg_shot - opp_xg_shot", REL_NOTE, origin="match"),
    count("shot_diff", "Territory & Dominance", "team", "own_shots - opp_shots", REL_NOTE, origin="match"),
    pct("xg_share", "Territory & Dominance", "team", "own_xg_shot", "own_xg_shot + opp_xg_shot", note=REL_NOTE, origin="match"),
    pct("shot_share", "Territory & Dominance", "team", "own_shots", "own_shots + opp_shots", note=REL_NOTE, origin="match"),
    pct("pass_share", "Territory & Dominance", "team", "own_passes", "own_passes + opp_passes",
        note="Possession proxy (Wyscout has no direct possession stat). " + REL_NOTE, origin="match"),
    pct("territory_share", "Territory & Dominance", "team", "own_passes_to_final_third",
        "own_passes_to_final_third + opp_passes_to_final_third", note="Field-tilt proxy. " + REL_NOTE, origin="match"),
    pct("box_touch_share", "Territory & Dominance", "team", "own_touch_in_box", "own_touch_in_box + opp_touch_in_box",
        note=REL_NOTE, origin="match"),
    pct("recovery_share", "Territory & Dominance", "team", "own_recoveries", "own_recoveries + opp_recoveries",
        note=REL_NOTE, origin="match"),
]


# --- Computation ---

def _eval(df: pd.DataFrame, terms) -> pd.Series | None:
    """Sum coef * total_<stem>; None if any column is missing (stat doesn't apply to this table)."""
    total = 0.0
    for coef, stem in terms:
        col = f"total_{stem}"
        if col not in df:
            return None
        total = total + coef * pd.to_numeric(df[col], errors="coerce")
    return total


def _apply(df: pd.DataFrame, entries: list[Derived], scope: str, divisor: str) -> tuple[pd.DataFrame, list[dict]]:
    new: dict[str, pd.Series] = {}
    rows: list[dict] = []
    per = df[divisor].where(df[divisor] > 0) if divisor in df else None
    for d in entries:
        if d.scope not in ("both", scope):
            continue
        num = _eval(df, d.num)
        den = _eval(df, d.den) if d.den else None
        if num is None or (d.den and den is None):
            continue
        base = {"scope": scope, "weight": None, "fit": None, "origin": "derived",
                "groups": ", ".join(d.groups), "description": d.note, "min_den": d.min_den if d.den else None}
        if d.kind == "count":
            cols = {f"total_{d.name}": (num, "derived_total", render_terms(d.num), None),
                    f"average_{d.name}": (num / per, "derived_per_exposure" if d.origin == "totals" else "derived_per_match",
                                          f"total_{d.name}", d.per)}
        else:
            prefix, scale, method = (("percent_", 100.0, "derived_percent") if d.kind == "percent"
                                     else ("ratio_", 1.0, "derived_ratio"))
            cols = {f"{prefix}{d.name}": (scale * num / den.where(den >= d.min_den), method,
                                          render_terms(d.num), render_terms(d.den))}
        for col, (series, method, n_txt, d_txt) in cols.items():
            if col in df.columns or col in new:
                print(f"  derived: skipping {col} (column already exists)")
                continue
            new[col] = series
            rows.append({**base, "stat_col": col, "method": method, "numerator": n_txt, "denominator": d_txt})
    out = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1) if new else df
    return out, rows


def add_derived(df: pd.DataFrame, scope: str) -> tuple[pd.DataFrame, list[dict]]:
    """Append derived columns to a rolled-up table (needs total_* columns and `exposure`)."""
    return _apply(df, REGISTRY, scope, "exposure")


def add_relative_team(con, df: pd.DataFrame, include_friendlies: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    """Add team-vs-opponent stats from team_match_stats, merged on (team_id, year, term)."""
    kinds = "('regular','friendly')" if include_friendlies else "('regular')"
    sums = ",\n".join(f"sum(a.total_{s}) AS total_own_{s}, sum(b.total_{s}) AS total_opp_{s}" for s in REL_STEMS)
    rel = con.execute(f"""
        SELECT a.team_id, t.year, t.term, count(*) AS n_rel_matches, {sums}
        FROM wyscout.team_match_stats a
        JOIN wyscout.team_match_stats b ON a.match_id = b.match_id AND a.team_id <> b.team_id
        JOIN wyscout.matches x ON x.match_id = a.match_id
        JOIN wyscout.season_terms t ON t.season_id = x.season_id
        WHERE t.kind IN {kinds} AND t.term <> 'Unknown'
        GROUP BY a.team_id, t.year, t.term""").df()
    df = df.merge(rel, on=["team_id", "year", "term"], how="left")
    df, rows = _apply(df, RELATIVE, "team", "n_rel_matches")
    def raw_row(col, method, text):
        return {"scope": "team", "stat_col": col, "method": method, "numerator": text, "denominator": None,
                "weight": None, "fit": None, "origin": "derived", "groups": "Territory & Dominance",
                "description": REL_NOTE, "min_den": None}

    raw = [raw_row(f"total_{side}_{s}", "derived_match_sum",
                   f"sum of total_{s} for {'the team' if side == 'own' else 'its opponents'} over the same matches")
           for s in REL_STEMS for side in ("own", "opp") if f"total_{side}_{s}" in df]
    raw.append(raw_row("n_rel_matches", "derived_match_count", "number of matches in the own-vs-opponent sums"))
    return df, raw + rows


# --- Base stat groups (raw stems; each expands to total_<stem>, average_<stem>, percent_<stem> where they exist) ---
# A stem can sit in several groups on purpose (e.g. progressive_run is both Progression and Dribbling).
# "both" stems exist for teams and players; "team" / "player" stems exist for only that table.
def _g(both="", team="", player=""):
    return {"both": both.split(), "team": team.split(), "player": player.split()}


BASE_GROUPS: dict[str, dict[str, list[str]]] = {
    "Shooting": _g(
        both="goals shots head_shots xg_shot penalties successful_penalties direct_free_kicks "
             "direct_free_kicks_on_target free_kicks free_kicks_on_target",
        player="shots_on_target"),
    "Chance Creation": _g(
        both="assists shot_assists shot_on_target_assists key_passes successful_key_passes smart_passes "
             "successful_smart_passes through_passes successful_through_passes crosses successful_crosses corners",
        player="xg_assist second_assists third_assists"),
    "Passing": _g(
        both="passes successful_passes lateral_passes successful_lateral_passes back_passes successful_back_passes "
             "long_passes successful_long_passes received_pass linkup_plays successful_linkup_plays"),
    "Progression": _g(
        both="forward_passes successful_forward_passes vertical_passes successful_vertical_passes passes_to_final_third "
             "successful_passes_to_final_third progressive_run touch_in_box accelerations attacking_actions "
             "successful_attacking_actions",
        player="progressive_passes successful_progressive_passes"),
    "Dribbling": _g(
        both="dribbles successful_dribbles new_successful_dribbles offensive_duels offensive_duels_won "
             "new_offensive_duels_won progressive_run accelerations",
        player="fouls_suffered"),
    "Duels": _g(
        both="duels duels_won new_duels_won aerial_duels aerial_duels_won defensive_duels defensive_duels_won "
             "new_defensive_duels_won offensive_duels offensive_duels_won new_offensive_duels_won loose_ball_duels "
             "loose_ball_duels_won"),
    "Aerial": _g(
        both="aerial_duels aerial_duels_won field_aerial_duels field_aerial_duels_won head_shots"),
    "Defending": _g(
        both="defensive_actions defensive_duels defensive_duels_won new_defensive_duels_won interceptions "
             "loose_ball_duels loose_ball_duels_won",
        team="successful_defensive_actions shots_against xg_shot_against conceded_goals clean_sheets",
        player="successful_defensive_action sliding_tackles successful_sliding_tackles clearances shots_blocked "
               "dribbles_against dribbles_against_won"),
    "Pressing & Recovery": _g(
        both="recoveries opponent_half_recoveries dangerous_opponent_half_recoveries pressing_duels pressing_duels_won",
        team="ppda", player="counterpressing_recoveries"),
    "Ball Security": _g(both="losses own_half_losses dangerous_own_half_losses missed_balls"),
    "Discipline": _g(both="fouls yellow_cards red_cards direct_red_cards offsides",
                     team="opponent_offsides", player="fouls_suffered"),
    "Goalkeeping": _g(
        both="gk_saves gk_exits gk_successful_exits gk_aerial_duels gk_aerial_duels_won",
        team="gk_goal_kicks gk_goal_kicks_success long_goal_kicks short_goal_kicks",
        player="gk_shots_against gk_conceded_goals gk_clean_sheets xg_save goal_kicks goal_kicks_long goal_kicks_short "
               "successful_goal_kicks"),
    "Results": _g(team="matches matches_tagged goals conceded_goals clean_sheets"),
    "Territory & Dominance": _g(),  # engineered only: needs the opponent's row of the same match
    "Playing Time": _g(player="matches matches_in_start matches_substituted matches_coming_off minutes_on_field minutes_tagged"),
}


def _code_list(stems) -> str:
    return ", ".join(f"`{s}`" for s in stems) if stems else "-"


def _derived_cols(d: Derived) -> list[str]:
    if d.kind == "count":
        return [f"total_{d.name}", f"average_{d.name}"]
    return [f"{'percent_' if d.kind == 'percent' else 'ratio_'}{d.name}"]


def build_groups_doc() -> str:
    L = ["# Stat groups",
         "",
         "_Generated by `python -m wyscout.derived --write-groups` from `BASE_GROUPS` and the derived-stat registry in "
         "`wyscout/derived.py` - edit those, not this file._",
         "",
         "Stats are organised into functional groups for team and player analysis. Each group lists",
         "",
         "1. **Raw stats** - stems from the Wyscout data. A stem such as `key_passes` expands to `total_key_passes`, "
         "`average_key_passes` (per exposure) and `percent_key_passes` wherever those columns exist.",
         "2. **Engineered stats** - built in `wyscout/derived.py` (details in [DERIVED_STATS.md](DERIVED_STATS.md)). "
         "Column prefixes: `total_` summed count, `average_` per exposure (per match for opponent-based stats), "
         "`percent_` 0-100 rate, `ratio_` plain ratio.",
         "",
         "A stat may belong to more than one group on purpose (e.g. `progressive_run` is both Progression and Dribbling, "
         "`head_shots` is both Shooting and Aerial). Scope is `team`, `player` or `both`.",
         "",
         "## Overview",
         "",
         "| Group | Raw (team) | Raw (player) | Engineered (team) | Engineered (player) |",
         "|---|---|---|---|---|"]
    everything = REGISTRY + RELATIVE

    def eng(g, scope):
        return [d for d in everything if g in d.groups and d.scope in ("both", scope)]

    for g in GROUPS:
        b = BASE_GROUPS.get(g, _g())
        L.append(f"| [{g}](#{re.sub(r'[^a-z0-9]+', '-', g.lower()).strip('-')}) | {len(b['both']) + len(b['team'])} | "
                 f"{len(b['both']) + len(b['player'])} | {len(eng(g, 'team'))} | {len(eng(g, 'player'))} |")
    L.append("")
    for g in GROUPS:
        b = BASE_GROUPS.get(g, _g())
        L += [f"## {g}", "", "### Raw stats", ""]
        if b["both"]:
            L.append(f"- **Team and player:** {_code_list(b['both'])}")
        if b["team"]:
            L.append(f"- **Team only:** {_code_list(b['team'])}")
        if b["player"]:
            L.append(f"- **Player only:** {_code_list(b['player'])}")
        if not (b["both"] or b["team"] or b["player"]):
            L.append("- None (this group exists only through engineered stats).")
        L += ["", "### Engineered stats", ""]
        members = [d for d in everything if g in d.groups]
        if not members:
            L.append("- None yet.")
        else:
            L += ["| Column(s) | Type | Scope | Formula |", "|---|---|---|---|"]
            for d in members:
                cols = ", ".join(f"`{c}`" for c in _derived_cols(d))
                kind = {"count": "total + per match" if d.origin == "match" else "total + per exposure",
                        "percent": "percent (0-100)", "ratio": "ratio"}[d.kind]
                if d.kind == "count":
                    formula = render_terms(d.num, "")
                else:
                    wrap = f"({render_terms(d.den, '')})" if len(d.den) > 1 else render_terms(d.den, "")
                    formula = f"{'100 * ' if d.kind == 'percent' else ''}{render_terms(d.num, '')} / {wrap}"
                if d.origin == "match":
                    formula = formula.replace("own_", "own ").replace("opp_", "opp ")
                L.append(f"| {cols} | {kind} | {d.scope} | `{formula}` |")
        L.append("")
    return "\n".join(L)


# --- Documentation ---

_FACTS = [
    ("Dribbles are a subset of offensive duels", "dribbles <= offensive_duels and successful_dribbles <= offensive_duels_won in 100% of 64,954 player rows and all team-match rows"),
    ("duels = aerial + defensive + offensive + loose-ball", "Exact in 100% of rows using aerial_duels (which includes goalkeeper aerials); field_aerial_duels excludes them (the gap is exactly gk_aerial_duels)"),
    ("losses ~ failed passes + failed offensive duels", "2.93M losses vs 2.87M (within 2%), so losses already is the combined turnover count"),
    ("xg_shot includes penalties", "Rows with exactly one shot that was a penalty: xG = 0.76 every time (2,051 rows)"),
    ("dangerous_own_half_losses is a subset of own_half_losses", "100% of rows"),
    ("gk_shots_against is shots on target faced", "gk_saves + gk_conceded_goals = gk_shots_against in 100% of rows"),
    ("Passes with a direction", "forward + lateral + back covers ~86% of passes (6.44M of 7.47M); the remainder is unexplained"),
    ("Vertical passes are not nested in forward passes", "vertical <= forward in only ~65% of rows"),
    ("Shots are not nested in box touches", "shots <= touch_in_box in only ~80% of rows"),
    ("Team goal kicks", "long + short = 107% of gk_goal_kicks for teams; player goal_kicks_long + goal_kicks_short matches goal_kicks within 0.1%"),
    ("Opponent join", "team_match_stats has exactly 2 rows per match (29,986 matches), so own-vs-opponent is unambiguous"),
]

_NOT_ADDED = [
    ("giveaways (failed passes + failed offensive duels)", "Redundant: the API's losses column is already that total (within 2%)."),
    ("failed_dribbles + failed_offensive_duels", "Dribbles are a subset of offensive duels - would double count."),
    ("gk_sweeper_rate = gk_exits / gk_shots_against", "gk_shots_against is shots on target faced, not all shots, so it's built under an honest name: gk_exits_per_shot_on_target_faced."),
    ("shots_on_target_rate", "The API already provides percent_shots_on_target."),
    ("Plain success rates (pass accuracy, duel win %, ...)", "Already supplied by the API's percent_* columns."),
    ("Winning/losing duel rates with the new_* columns", "Not built here; use the API's percent_new_* columns (new_* are redefinitions - defensive win 65% vs 23% old - and must not be mixed with the old ones in one formula)."),
]


def build_docs() -> str:
    L = ["# Derived (engineered) stats",
         "",
         "_Generated by `python -m wyscout.derived --write-docs` from the registry in `wyscout/derived.py` - edit the registry, not this file._",
         "",
         "These stats are computed in `wyscout.rollup` right after the totals are summed across competitions, and are stored",
         "in `team_term_stats` / `player_term_stats`. Each one is also listed in the `stat_formulas` table with",
         "`origin = 'derived'`, its formula, group and description. No API calls are involved: every stat is a formula over",
         "columns that already existed.",
         "",
         "## How to read the column names",
         "",
         "| Prefix | What it is | Notes |",
         "|---|---|---|",
         "| `total_<name>` | Summed count | Only for count-type stats. |",
         "| `average_<name>` | `total_<name>` / `exposure` | Same basis as the API's own `average_*` columns: per *exposure* (roughly per match scaled by tagged time). **This is not strictly per 90.** Opponent-based stats divide by `n_rel_matches` instead (plain per match). |",
         "| `percent_<name>` | 100 * numerator / denominator | Shares and rates, 0-100. |",
         "| `ratio_<name>` | numerator / denominator | Ratios that are not percentages (e.g. xG per shot). |",
         "",
         f"Rates are computed as a ratio of sums, never an average of per-match ratios, and are NULL when the summed denominator is below `min_den` (default {MIN_DEN}) so tiny samples don't dominate leaderboards.",
         "Count stats get `total_` + `average_`; shares and ratios get a single rate column, because a total or per-exposure version of a ratio isn't meaningful.",
         "",
         "In the tables, formulas use bare stat names: `xg_shot` means the `total_xg_shot` column. Scope is `team`, `player` or `both`.",
         ""]
    everything = REGISTRY + RELATIVE
    for g in GROUPS:
        members = [d for d in everything if g in d.groups]
        if not members:
            continue
        L += [f"## {g}", "", "| Column(s) | Type | Formula | Scope | Notes |", "|---|---|---|---|---|"]
        for d in members:
            if d.kind == "count":
                cols = f"`total_{d.name}`, `average_{d.name}`"
                kind = "total + per match" if d.origin == "match" else "total + per exposure"
                formula = render_terms(d.num, "").replace("|", "/")
            else:
                prefix = "percent_" if d.kind == "percent" else "ratio_"
                cols, kind = f"`{prefix}{d.name}`", "percent (0-100)" if d.kind == "percent" else "ratio"
                formula = f"{render_terms(d.num, '')} / ({render_terms(d.den, '')})" if len(d.den) > 1 else f"{render_terms(d.num, '')} / {render_terms(d.den, '')}"
                if d.kind == "percent":
                    formula = "100 * " + formula
            formula = formula.replace("own_", "own ").replace("opp_", "opp ") if d.origin == "match" else formula
            note = d.note + (f" (min denominator {d.min_den:g})" if d.kind != "count" and d.min_den != MIN_DEN else "")
            L.append(f"| {cols} | {kind} | `{formula}` | {d.scope} | {note} |")
        L.append("")
    L += ["## Opponent-based stats: raw sums",
          "",
          "For the team-vs-opponent stats the table also keeps the underlying sums so they can be reproduced: "
          + ", ".join(f"`total_own_{s}` / `total_opp_{s}`" for s in REL_STEMS)
          + ", plus `n_rel_matches` (matches used). They come from `team_match_stats` joined to the opposing team's row in the same match, "
          "so they can differ slightly from the season-stats-based `total_*` columns (the season endpoint includes matches without event data). "
          "Some team-terms have only one match (`n_rel_matches = 1`), where a blowout gives extreme per-match values "
          "(e.g. xG diff of -13); filter on `n_rel_matches` before ranking teams on `average_*` opponent stats.",
          "",
          "## Facts these formulas rely on (checked against the data)",
          "",
          "| Assumption | Result |", "|---|---|"]
    L += [f"| {a} | {b} |" for a, b in _FACTS]
    L += ["", "## Considered and deliberately not built", "", "| Idea | Why not |", "|---|---|"]
    L += [f"| {a} | {b} |" for a, b in _NOT_ADDED]
    L += ["",
          "## Known caveat (existing column, not derived)",
          "",
          "`total_ppda` is a per-match ratio (about 8.8 on average), not a count, so its rolled-up `total_ppda` is a meaningless sum; "
          "use `average_ppda`. It is not touched here.",
          ""]
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write-docs", action="store_true", help="regenerate wyscout/DERIVED_STATS.md")
    ap.add_argument("--write-groups", action="store_true", help="regenerate wyscout/STAT_GROUPS.md")
    args = ap.parse_args()
    if args.write_docs:
        path = Path(__file__).with_name("DERIVED_STATS.md")
        path.write_text(build_docs(), encoding="utf-8")
        print(f"wrote {path}")
    if args.write_groups:
        path = Path(__file__).with_name("STAT_GROUPS.md")
        path.write_text(build_groups_doc(), encoding="utf-8")
        print(f"wrote {path}")
    if not (args.write_docs or args.write_groups):
        print(f"{len(REGISTRY)} registry stats + {len(RELATIVE)} opponent-based stats")
