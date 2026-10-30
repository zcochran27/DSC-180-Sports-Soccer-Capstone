"""Wyscout v3 API starter: NCAA D1 men's and women's competitions, seasons, matches,
stats and events, each returned as a pandas DataFrame.

Setup:
    pip install -r requirements.txt
    copy .env.example to .env and fill in API_USERNAME / API_PASSWORD

Run the demo:
    python -m wyscout.api

Or import the functions:
    from wyscout.api import *
    comps = get_competitions("female")
    seasons = get_seasons(comps["compId"])
    matches = get_season_matches(seasons.loc[seasons["name"] == "2026 Fall", "seasonId"].iloc[0])
    events = get_match_events(matches["matchId"].iloc[0])

API quirks worth knowing:
  * Docs: the OpenAPI spec is at https://apidocs.wyscout.com/assets/specs/prod/current.yml
  * /competitions needs areaId as the alpha-3 code "USA". "US" or 840 return an empty list.
  * /competitions has no gender filter. Each competition carries gender "male"/"female",
    so filter on that column. Women's NCAA comps have names ending in "(W)".
  * Unknown query params are silently ignored, so a typo looks like "no results".
  * Errors often come back as HTTP 400 with the real code in the body:
    {"error": {"code": 403}} = not licensed, 404 = does not exist.
  * Rate limit is 12 requests/second per key (HTTP 429). The client retries 429/5xx.
  * hasDataAvailable is only reliable on /matches/{id}, not on season/team match lists.
  * Every school has separate men's and women's team ids and player ids.
"""
from __future__ import annotations

import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://apirest.wyscout.com/v3"

SESSION = requests.Session()
SESSION.auth = (os.getenv("API_USERNAME"), os.getenv("API_PASSWORD"))


class WyscoutError(RuntimeError):
    pass


def call_api(path: str, params: dict | None = None, attempts: int = 4) -> dict:
    """GET {BASE_URL}{path} and return the JSON body. Retries 429/5xx/timeouts with backoff."""
    url = path if path.startswith("http") else f"{BASE_URL}{path}"
    last = "request failed"
    for attempt in range(attempts):
        try:
            resp = SESSION.get(url, params=params or {}, timeout=(5, 30))
        except (requests.Timeout, requests.ConnectionError):
            last = "connection failed or timed out"
        else:
            if resp.status_code in (401, 403):
                raise WyscoutError(f"Access denied (HTTP {resp.status_code}) for {url}; check .env")
            if resp.status_code == 429 or resp.status_code >= 500:
                last = f"HTTP {resp.status_code}"
            else:
                payload = resp.json()
                error = payload.get("error") if isinstance(payload, dict) else None
                if error or not resp.ok:
                    raise WyscoutError(f"Wyscout error for {url}: HTTP {resp.status_code} {error}")
                return payload
        if attempt + 1 < attempts:
            time.sleep(min(2 ** attempt, 8))
    raise WyscoutError(f"{url} failed after {attempts} attempts: {last}")


# ── Competitions & seasons ────────────────────────────────────────────────────

def get_competitions(gender: str | None = None) -> pd.DataFrame:
    """All USA competitions on the account. gender: None, "male" or "female"."""
    comps = pd.json_normalize(call_api("/competitions", {"areaId": "USA"})["competitions"], sep="_")
    comps = comps.rename(columns={"wyId": "compId"})
    if gender is not None:
        comps = comps[comps["gender"] == gender].reset_index(drop=True)
    return comps


def get_seasons(comp_ids) -> pd.DataFrame:
    """Every season for one competition id or an iterable of them."""
    if isinstance(comp_ids, (int, str)):
        comp_ids = [comp_ids]
    frames = []
    for comp_id in comp_ids:
        rows = call_api(f"/competitions/{int(comp_id)}/seasons")["seasons"]
        df = pd.json_normalize(rows, sep="_")
        # Each row is {"seasonId", "season": {...}}; promote the nested season fields.
        df.columns = [c.removeprefix("season_") for c in df.columns]
        df = df.drop(columns=["wyId"], errors="ignore")
        df["compId"] = int(comp_id)
        frames.append(df)
    seasons = pd.concat(frames, ignore_index=True).drop(columns=["competitionId"], errors="ignore")
    for col in ("startDate", "endDate"):
        if col in seasons:
            seasons[col] = pd.to_datetime(seasons[col], errors="coerce")
    seasons["seasonId"] = pd.to_numeric(seasons["seasonId"])
    return seasons


# ── Matches ───────────────────────────────────────────────────────────────────

def get_season_matches(season_id: int) -> pd.DataFrame:
    """Fixture list for a season (label, date, status, ...). Fast: one request."""
    matches = pd.json_normalize(call_api(f"/seasons/{int(season_id)}/matches")["matches"], sep="_")
    if "dateutc" in matches:
        matches["dateutc"] = pd.to_datetime(matches["dateutc"], errors="coerce")
    matches["seasonId"] = int(season_id)
    return matches


def get_team_matches(team_id: int) -> pd.DataFrame:
    """Every match for a team (e.g. UCSD women = 68652)."""
    matches = pd.json_normalize(call_api(f"/teams/{int(team_id)}/matches")["matches"], sep="_")
    if "dateutc" in matches:
        matches["dateutc"] = pd.to_datetime(matches["dateutc"], errors="coerce")
    return matches


def get_match_details(match_id: int) -> pd.Series:
    """One match: competition/season ids, home/away team ids, score, hasDataAvailable."""
    raw = call_api(f"/matches/{int(match_id)}")
    teams = raw.get("teamsData", {}) or {}
    side = {t["side"]: int(tid) for tid, t in teams.items()}
    out = {k: v for k, v in raw.items() if not isinstance(v, (dict, list))}
    out["home_teamId"] = side.get("home")
    out["away_teamId"] = side.get("away")
    for s in ("home", "away"):
        tid = side.get(s)
        if tid is not None:
            out[f"{s}_score"] = teams[str(tid)].get("score")
    return pd.Series(out)


def get_match_raw(match_id: int) -> dict:
    """Unflattened /matches/{id} with coaches, players, teams, round and season expanded.

    Keeps the lists/dicts (referees, lineups, coaches, teamsData) that get_match_details drops.
    """
    return call_api(f"/matches/{int(match_id)}",
                    {"details": "coaches,players,teams,round,season,competition"})


def get_match_formations(match_id: int) -> dict:
    """Raw position/formation timeline for a match."""
    return call_api(f"/matches/{int(match_id)}/formations", {"fetch": "teams,players"})


# ── Dimensions: teams, players, rounds ────────────────────────────────────────

def get_season_teams(season_id: int) -> list[dict]:
    """Teams that played at least one match in the season (raw team dicts)."""
    return call_api(f"/seasons/{int(season_id)}/teams").get("teams", [])


def get_season_players(season_id: int, limit: int = 100, max_pages: int = 200) -> list[dict]:
    """Every player in a season. Paginated at 100 per page; stops on a short, empty or repeated page."""
    players, seen, page = [], set(), 1
    while page <= max_pages:
        batch = call_api(f"/seasons/{int(season_id)}/players",
                         {"limit": limit, "page": page, "details": "currentTeam"}).get("players", [])
        fresh = [p for p in batch if p.get("wyId") not in seen]
        if not fresh:  # empty page, or the API repeated a page it already gave us
            break
        seen.update(p.get("wyId") for p in fresh)
        players.extend(fresh)
        if len(batch) < limit:
            break
        page += 1
    return players


def get_team(team_id: int) -> dict:
    """Raw team detail (name, city, area, gender, children)."""
    return call_api(f"/teams/{int(team_id)}")


def get_player(player_id: int) -> dict:
    """Raw player detail (bio, role, foot, height, weight)."""
    return call_api(f"/players/{int(player_id)}")


def get_round(round_id: int) -> dict:
    """Raw round detail."""
    return call_api(f"/rounds/{int(round_id)}")


# ── Stats ─────────────────────────────────────────────────────────────────────

def _flatten_stats(payload: dict, ids: dict) -> dict:
    """Flatten Wyscout's {total, average, percent} blocks into total_x / average_x / percent_x."""
    row = dict(ids)
    for section in ("total", "average", "percent"):
        for key, value in (payload.get(section) or {}).items():
            row[f"{section}_{key}"] = value
    return row


def get_team_match_stats(match_id: int, team_id: int) -> pd.DataFrame:
    """One row of team advanced stats for one team in one match."""
    raw = call_api(f"/teams/{int(team_id)}/matches/{int(match_id)}/advancedstats")
    return pd.DataFrame([_flatten_stats(raw, {"matchId": int(match_id), "teamId": int(team_id)})])


def get_match_team_stats(match_id: int) -> pd.DataFrame:
    """Team advanced stats for both teams in a match (two rows)."""
    d = get_match_details(match_id)
    return pd.concat([get_team_match_stats(match_id, d[f"{s}_teamId"]) for s in ("home", "away")],
                     ignore_index=True)


def get_match_player_stats(match_id: int) -> pd.DataFrame:
    """One row per player in the match: minutes, main position, and all advanced stats."""
    raw = call_api(f"/matches/{int(match_id)}/advancedstats/players")
    rows = []
    for p in raw.get("players", []):
        ids = {k: p.get(k) for k in ("playerId", "matchId", "competitionId", "seasonId", "roundId")}
        positions = p.get("positions") or []
        if positions:
            ids["position_code"] = positions[0].get("position", {}).get("code")
            ids["position_name"] = positions[0].get("position", {}).get("name")
            ids["position_percent"] = positions[0].get("percent")
        rows.append(_flatten_stats(p, ids))
    return pd.DataFrame(rows)


def get_team_season_stats(team_id: int, comp_id: int, season_id: int) -> pd.DataFrame:
    """Season-to-date team advanced stats for one team in one competition season."""
    raw = call_api(f"/teams/{int(team_id)}/advancedstats",
                   {"compId": int(comp_id), "seasonId": int(season_id)})
    return pd.DataFrame([_flatten_stats(raw, {"teamId": int(team_id), "compId": int(comp_id),
                                              "seasonId": int(season_id)})])


def get_player_season_stats(player_id: int, comp_id: int, season_id: int) -> pd.DataFrame:
    """Season-to-date player advanced stats for one competition season."""
    raw = call_api(f"/players/{int(player_id)}/advancedstats",
                   {"compId": int(comp_id), "seasonId": int(season_id)})
    return pd.DataFrame([_flatten_stats(raw, {"playerId": int(player_id), "compId": int(comp_id),
                                              "seasonId": int(season_id)})])


# ── Events ────────────────────────────────────────────────────────────────────

def get_match_events(match_id: int) -> pd.DataFrame:
    """Every event in a match, flattened: one row per event.

    Nested objects become underscore columns (type_primary, location_x, player_id,
    team_id, pass_endLocation_x, shot_xg, possession_id, ...). type_secondary stays a
    list of tags. Coordinates are 0-100, attacking left to right for the team in possession.
    """
    raw = call_api(f"/matches/{int(match_id)}/events")
    events = pd.json_normalize(raw.get("events", []), sep="_")
    if events.empty:
        return events
    events["matchId"] = int(match_id)
    sort_cols = [c for c in ("matchPeriod", "minute", "second", "id") if c in events]
    return events.sort_values(sort_cols).reset_index(drop=True)


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pd.set_option("display.width", 160)

    for gender in ("male", "female"):
        comps = get_competitions(gender)
        print(f"\n{gender}: {len(comps)} competitions")
        print(comps[["compId", "name", "gender"]].head(10).to_string(index=False))

    # Pick the first women's competition and its latest season to demo the rest.
    comp_id = get_competitions("female")["compId"].iloc[0]
    seasons = get_seasons(comp_id).sort_values("startDate")
    print(f"\nseasons for comp {comp_id}:\n", seasons[["seasonId", "name", "startDate"]].to_string(index=False))

    season_id = seasons["seasonId"].iloc[-1]
    matches = get_season_matches(season_id)
    print(f"\n{len(matches)} matches in season {season_id}")
    print(matches.head().to_string(index=False))

    for match_id in matches["matchId"]:
        details = get_match_details(match_id)
        if details.get("hasDataAvailable"):
            break
    else:
        raise SystemExit("no match with data available in that season")

    print(f"\nmatch {match_id}:\n{details.to_string()}")
    print("\nteam stats:\n", get_match_team_stats(match_id).iloc[:, :10].to_string(index=False))
    players = get_match_player_stats(match_id)
    print(f"\nplayer stats: {players.shape}")
    print(players.iloc[:5, :10].to_string(index=False))
    events = get_match_events(match_id)
    print(f"\nevents: {events.shape}")
    print(events[[c for c in ("minute", "second", "type_primary", "team_name", "player_name",
                              "location_x", "location_y") if c in events]].head(15).to_string(index=False))
