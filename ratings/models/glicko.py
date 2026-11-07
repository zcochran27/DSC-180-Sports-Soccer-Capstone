"""Glicko-style rating: an Elo-like rating plus an uncertainty ("rating deviation", RD).

A team's RD grows while it sits idle and when a new season starts, and shrinks as it plays.
A team with a large RD (few or old games) has its rating moved more by each result.

Settings of a model (new_model arguments):
    hfa        home advantage, in rating points
    rd0        starting RD for a new team
    rd_max     RD never grows beyond this
    c_week     how much RD grows per week without a match
    season_c   how much RD jumps at the start of a season
    carry      share of a team's rating above/below 1500 kept when a new season starts
"""
import math

Q = math.log(10) / 400.0
START_RATING = 1500.0


def new_model(hfa=60.0, rd0=200.0, rd_max=250.0, c_week=6.0, season_c=120.0, carry=0.75, name="glicko"):
    return {"kind": "glicko", "name": name, "hfa": hfa, "rd0": rd0, "rd_max": rd_max, "c_week": c_week,
            "season_c": season_c, "carry": carry,
            "ratings": {},       # team id -> rating (a team not in here has START_RATING)
            "rd": {},            # team id -> rating deviation (a team not in here has rd0)
            "last_played": {}}   # team id -> date of last match


def get_rating(model, team):
    return model["ratings"].get(team, START_RATING)


def get_rd(model, team):
    return model["rd"].get(team, model["rd0"])


def opponent_weight(rd):
    """Results against an opponent whose rating is uncertain count for less."""
    return 1.0 / math.sqrt(1.0 + 3 * Q * Q * rd * rd / math.pi ** 2)


def start_season(model, year):
    for team in list(model["ratings"]):
        model["ratings"][team] = START_RATING + model["carry"] * (model["ratings"][team] - START_RATING)
        model["rd"][team] = min(math.hypot(get_rd(model, team), model["season_c"]), model["rd_max"])


def grow_rd_for_idle_time(model, team, date):
    if team in model["last_played"]:
        weeks_idle = max((date - model["last_played"][team]).days, 0) / 7.0
        new_rd = math.sqrt(get_rd(model, team) ** 2 + model["c_week"] ** 2 * weeks_idle)
        model["rd"][team] = min(new_rd, model["rd_max"])


def gap(model, match):
    return get_rating(model, match.home) - get_rating(model, match.away) + model["hfa"]


def gap_sd(model, match):
    """Uncertainty of the gap, from both teams' RD."""
    return math.hypot(get_rd(model, match.home), get_rd(model, match.away))


def update(model, match):
    for team in (match.home, match.away):
        grow_rd_for_idle_time(model, team, match.date)

    goal_diff = match.hg - match.ag
    home_score = 1.0 if goal_diff > 0 else 0.5 if goal_diff == 0 else 0.0

    # Work out both teams' new values before applying either, so neither uses the other's new rating.
    new_values = {}
    sides = ((match.home, match.away, home_score, model["hfa"]),
             (match.away, match.home, 1 - home_score, -model["hfa"]))
    for team, opponent, team_score, advantage in sides:
        weight = opponent_weight(get_rd(model, opponent))
        rating_diff = get_rating(model, team) + advantage - get_rating(model, opponent)
        expected = 1.0 / (1.0 + 10 ** (-weight * rating_diff / 400.0))
        precision = 1.0 / get_rd(model, team) ** 2 + Q * Q * weight * weight * expected * (1 - expected)
        new_rating = get_rating(model, team) + Q / precision * weight * (team_score - expected)
        new_values[team] = (new_rating, math.sqrt(1.0 / precision))

    for team, (rating, rd) in new_values.items():
        model["ratings"][team] = rating
        model["rd"][team] = rd
        model["last_played"][team] = match.date
