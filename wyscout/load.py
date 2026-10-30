"""Load Wyscout NCAA soccer data into MotherDuck following schema.sql.

Setup:
    pip install -r requirements.txt
    add MOTHERDUCK_TOKEN (and optionally MOTHERDUCK_DB, default wyscout_dev) to .env

Examples:
    python -m wyscout.load --stages dims                       # areas, competitions, seasons
    python -m wyscout.load --season-ids 123456 --team-id 68652 # one season, one team's matches
    python -m wyscout.load --stages check                      # integrity queries only

Stages run in dependency order: dims -> rosters -> matches -> match-data -> season-stats -> check.
Every write is an upsert on the table's primary key and each fetch is recorded in ingest_log,
so reruns skip finished work (use --force to refetch) and never duplicate rows.

NOTE: response shapes for lineups, formations and season-players were taken from the OpenAPI
spec, not from live calls. Run one season first and check the counts before a full load.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import pandas as pd
from dotenv import load_dotenv

from . import api

load_dotenv()

SCHEMA_SQL = Path(__file__).with_name("schema.sql")
STAGES = ["dims", "rosters", "matches", "match-data", "season-stats", "check"]
WORKERS = 24
MIN_INTERVAL = 0.09  # ~11 req/s, under the 12 req/s key limit
BATCH = 100  # matches fetched/written per chunk


# ── Throttle ──────────────────────────────────────────────────────────────────

_lock = threading.Lock()
_next_slot = 0.0
_raw_call_api = api.call_api


def _throttled_call_api(*args, **kwargs):
    global _next_slot
    with _lock:
        now = time.monotonic()
        wait = max(0.0, _next_slot - now)
        _next_slot = max(now, _next_slot) + MIN_INTERVAL
    if wait:
        time.sleep(wait)
    return _raw_call_api(*args, **kwargs)


api.call_api = _throttled_call_api  # all api.get_* helpers resolve call_api at call time


# ── DuckDB helpers ────────────────────────────────────────────────────────────

def connect() -> duckdb.DuckDBPyConnection:
    token = os.getenv("MOTHERDUCK_TOKEN")
    if not token:
        raise SystemExit("MOTHERDUCK_TOKEN missing from .env")
    db = os.getenv("MOTHERDUCK_DB", "wyscout_dev")
    boot = duckdb.connect(f"md:?motherduck_token={token}")
    boot.execute(f"CREATE DATABASE IF NOT EXISTS {db}")
    boot.close()
    con = duckdb.connect(f"md:{db}?motherduck_token={token}")
    con.execute(SCHEMA_SQL.read_text(encoding="utf-8"))
    return con


def snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def _kind(series: pd.Series) -> str | None:
    """DuckDB type for a DataFrame column from its non-null values; None if all null."""
    vals = series.dropna()
    if vals.empty:
        return None
    if vals.map(lambda v: isinstance(v, (list, tuple))).all():
        return "VARCHAR[]"
    if vals.map(lambda v: isinstance(v, bool)).all():
        return "BOOLEAN"
    if vals.map(lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)).all():
        return "NUM"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "TIMESTAMP"
    return "VARCHAR"


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Convert dict/list oddities to storable values and NaN to None."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            kind = _kind(out[col])
            if kind == "VARCHAR[]":
                out[col] = out[col].map(lambda v: [str(x) for x in v] if isinstance(v, (list, tuple)) else None)
            else:
                out[col] = out[col].map(
                    lambda v: json.dumps(v) if isinstance(v, (dict, list, tuple)) else v)
    return out.astype(object).where(out.notna(), None)


_cols_cache: dict[str, set[str]] = {}
_pk_cache: dict[str, list[str]] = {}


def table_cols(con, table: str) -> set[str]:
    if table not in _cols_cache:
        _cols_cache[table] = {r[0] for r in con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='wyscout' AND table_name=?", [table]).fetchall()}
    return _cols_cache[table]


def table_pk(con, table: str) -> list[str]:
    if table not in _pk_cache:
        row = con.execute("SELECT constraint_column_names FROM duckdb_constraints() WHERE schema_name='wyscout' "
                          "AND table_name=? AND constraint_type='PRIMARY KEY'", [table]).fetchone()
        _pk_cache[table] = list(row[0]) if row else []
    return _pk_cache[table]


def concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def upsert(con, table: str, df: pd.DataFrame, ignore: bool = False) -> int:
    """INSERT OR REPLACE (or OR IGNORE) df into wyscout.<table>, adding unseen columns first.

    Rows are deduplicated on the primary key (DuckDB rejects duplicate keys inside one insert).
    New numeric columns are DOUBLE unless named like an id, so a first chunk of whole-number
    metrics can't lock a column to BIGINT.
    """
    if df is None or df.empty:
        return 0
    df = df.loc[:, ~df.columns.duplicated()]
    pk = [c for c in table_pk(con, table) if c in df.columns]
    if pk:
        df = df.drop_duplicates(subset=pk, keep="last")
    df = _clean(df)
    existing = table_cols(con, table)
    keep = []
    for col in df.columns:
        if col in existing:
            keep.append(col)
            continue
        kind = _kind(pd.Series(df[col]).infer_objects())
        if kind is None:
            continue  # all-null new column: nothing to store yet
        if kind == "NUM":
            kind = "BIGINT" if col == "id" or col.endswith("_id") else "DOUBLE"
        con.execute(f'ALTER TABLE wyscout.{table} ADD COLUMN "{col}" {kind}')
        existing.add(col)
        keep.append(col)
    df = df[keep]
    cols = ", ".join(f'"{c}"' for c in keep)
    verb = "INSERT OR IGNORE" if ignore else "INSERT OR REPLACE"
    con.register("_df", df)
    try:
        con.execute(f"{verb} INTO wyscout.{table} ({cols}) SELECT {cols} FROM _df")
    finally:
        con.unregister("_df")
    return len(df)


def done_keys(con, entity: str) -> set[str]:
    """Keys already fetched (ok or skipped) for an entity, in one query."""
    return {r[0] for r in con.execute(
        "SELECT key FROM wyscout.ingest_log WHERE entity=? AND status IN ('ok','skipped')", [entity]).fetchall()}


def log_rows(con, rows: list[dict]) -> None:
    """rows: dicts with entity, key, status, rows, http_code."""
    upsert(con, "ingest_log", pd.DataFrame(rows))


def pipeline(items: list, fetch, write, batch: int = BATCH, label: str = "") -> None:
    """Fetch items in parallel and write each chunk while the next chunk is downloading.

    fetch(item) runs in worker threads; write(list_of_results) runs in the main thread, so the
    DuckDB connection is only ever used from one thread. At most two chunks are in memory.
    """
    chunks = [items[i:i + batch] for i in range(0, len(items), batch)]
    done = 0
    with ThreadPoolExecutor(WORKERS) as pool:
        pending: list = []
        for chunk in chunks + [[]]:
            futures = [pool.submit(fetch, it) for it in chunk]
            if pending:
                write([f.result() for f in pending])
                done += len(pending)
                print(f"  {time.strftime('%H:%M:%S')} {label}: {done}/{len(items)}", flush=True)
            pending = futures


def error_code(exc: Exception) -> int | None:
    m = re.search(r"\b(403|404)\b", str(exc))
    return int(m.group(1)) if m else None


def pick(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Rename source columns to target names, keeping only those present."""
    present = {src: dst for src, dst in mapping.items() if src in df.columns}
    return df[list(present)].rename(columns=present)


def snake_cols(df: pd.DataFrame, renames: dict[str, str] | None = None) -> pd.DataFrame:
    renames = renames or {}
    return df.rename(columns=lambda c: renames.get(c, snake(c)))


def date_or_none(value):
    ts = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(ts) else ts.date()


# ── Stage: dims ───────────────────────────────────────────────────────────────

def load_dims(con) -> list[int]:
    """areas, competitions (USA, both genders), seasons. Returns all season ids."""
    areas = pd.DataFrame(api.call_api("/areas").get("areas", []))
    upsert(con, "areas", pick(areas, {"id": "area_id", "alpha2code": "alpha2code",
                                      "alpha3code": "alpha3code", "name": "name"}))
    comps = api.get_competitions()
    comp_rows = pick(comps, {"compId": "comp_id", "name": "name", "gender": "gender", "type": "type",
                             "format": "format", "category": "category",
                             "divisionLevel": "division_level", "area_id": "area_id", "gsmId": "gsm_id"})
    print(f"competitions: {upsert(con, 'competitions', comp_rows)}")
    seasons = api.get_seasons(comps["compId"].tolist())
    rows = pick(seasons, {"seasonId": "season_id", "compId": "comp_id", "name": "name",
                          "startDate": "start_date", "endDate": "end_date", "active": "active",
                          "gsmId": "gsm_id"})
    for col in ("start_date", "end_date"):
        rows[col] = rows[col].map(date_or_none)
    print(f"seasons: {upsert(con, 'seasons', rows)}")
    return seasons["seasonId"].astype(int).tolist()


# ── Stage: rosters (team_seasons, player_seasons, teams, players) ─────────────

def _team_rows(t: dict) -> dict:
    area = t.get("area") or {}
    return {"team_id": t.get("wyId"), "name": t.get("name"), "official_name": t.get("officialName"),
            "gender": t.get("gender"), "type": t.get("type"), "category": t.get("category"),
            "city": t.get("city"), "area_id": area.get("id"), "image_url": t.get("imageDataURL"),
            "gsm_id": t.get("gsmId")}


def _player_row(p: dict) -> dict:
    role = p.get("role") or {}
    return {"player_id": p.get("wyId"), "first_name": p.get("firstName"),
            "middle_name": p.get("middleName"), "last_name": p.get("lastName"),
            "short_name": p.get("shortName"), "birth_date": date_or_none(p.get("birthDate")),
            "birth_area_id": (p.get("birthArea") or {}).get("id"),
            "passport_area_id": (p.get("passportArea") or {}).get("id"),
            "height": p.get("height"), "weight": p.get("weight"), "foot": p.get("foot"),
            "role_code2": role.get("code2"), "role_code3": role.get("code3"),
            "role_name": role.get("name"), "gender": p.get("gender"), "status": p.get("status"),
            "image_url": p.get("imageDataURL"), "gsm_id": p.get("gsmId")}


def _fetch_roster(args):
    season_id, comp_id = args
    try:
        return season_id, comp_id, api.get_season_teams(season_id), api.get_season_players(season_id), None
    except api.WyscoutError as exc:
        return season_id, comp_id, [], [], exc


def load_rosters(con, seasons: dict[int, int], force: bool) -> None:
    """seasons: {season_id: comp_id}."""
    done = set() if force else done_keys(con, "season_rosters")
    todo = [(sid, cid) for sid, cid in seasons.items() if str(sid) not in done]
    print(f"rosters: {len(todo)}/{len(seasons)} seasons to load")

    def write(results):
        teams, kids, tseasons, players, pseasons, logs = [], [], [], [], [], []
        for season_id, comp_id, ts, ps, exc in results:
            if exc is not None:
                logs.append({"entity": "season_rosters", "key": season_id, "status": "skipped",
                             "http_code": error_code(exc)})
                continue
            teams += [_team_rows(t) for t in ts]
            kids += [{"team_id": c["wyId"], "name": c.get("name"), "parent_team_id": t["wyId"]}
                     for t in ts for c in t.get("children") or []]
            tseasons += [{"team_id": t["wyId"], "season_id": season_id, "comp_id": comp_id} for t in ts]
            players += [_player_row(p) for p in ps]
            pseasons += [{"player_id": p["wyId"], "season_id": season_id, "team_id": p.get("currentTeamId"),
                          "comp_id": comp_id} for p in ps if p.get("currentTeamId") is not None]
            logs.append({"entity": "season_rosters", "key": season_id, "status": "ok", "rows": len(ps)})
        upsert(con, "teams", pd.DataFrame(teams))
        upsert(con, "teams", pd.DataFrame(kids), ignore=True)
        upsert(con, "team_seasons", pd.DataFrame(tseasons))
        upsert(con, "players", pd.DataFrame(players))
        upsert(con, "player_seasons", pd.DataFrame(pseasons))
        log_rows(con, logs)

    pipeline(todo, _fetch_roster, write, batch=10, label="rosters")


# ── Stage: matches ────────────────────────────────────────────────────────────

def _person_row(p: dict, id_key: str) -> dict:
    return {id_key: p.get("wyId"), "first_name": p.get("firstName"), "last_name": p.get("lastName"),
            "short_name": p.get("shortName"), "birth_date": date_or_none(p.get("birthDate")),
            "birth_area_id": (p.get("birthArea") or {}).get("id"),
            "passport_area_id": (p.get("passportArea") or {}).get("id"),
            "gender": p.get("gender"), "status": p.get("status")}


def parse_match(raw: dict) -> dict[str, pd.DataFrame]:
    """Split a /matches/{id} detail payload into rows for each match table."""
    match_id = raw.get("wyId") or raw.get("matchId")
    teams = raw.get("teamsData") or {}
    side = {t.get("side"): int(tid) for tid, t in teams.items()}
    home, away = side.get("home"), side.get("away")
    hs = (teams.get(str(home)) or {}).get("score") if home else None
    as_ = (teams.get(str(away)) or {}).get("score") if away else None
    winner = raw.get("winner")
    match = {"match_id": match_id, "comp_id": raw.get("competitionId"), "season_id": raw.get("seasonId"),
             "round_id": raw.get("roundId"), "gameweek": raw.get("gameweek"),
             "date_utc": pd.to_datetime(raw.get("dateutc"), errors="coerce"),
             "status": raw.get("status"), "duration": raw.get("duration"), "label": raw.get("label"),
             "home_team_id": home, "away_team_id": away, "home_score": hs, "away_score": as_,
             "winner_team_id": winner if winner else None,
             "has_data_available": raw.get("hasDataAvailable")}
    refs, ref_dim = [], []
    for r in raw.get("referees") or []:
        ref = r.get("referee") or {}
        rid = r.get("refereeId") or ref.get("wyId")
        if rid is None:
            continue
        refs.append({"match_id": match_id, "referee_id": rid, "role": r.get("role") or "unknown"})
        ref_dim.append(_person_row({**ref, "wyId": rid}, "referee_id"))
    coaches, lineups = [], []
    for tid, t in teams.items():
        if t.get("coachId"):  # coachId 0/None = no coach recorded
            coaches.append({"match_id": match_id, "team_id": int(tid), "coach_id": t["coachId"]})
        formation = t.get("formation") or {}
        subs = {s.get("playerIn"): s for s in formation.get("substitutions") or []}
        outs = {s.get("playerOut"): s for s in formation.get("substitutions") or []}
        for starter, key in ((True, "lineup"), (False, "bench")):
            for p in formation.get(key) or []:
                pid = p.get("playerId") or (p.get("player") or {}).get("wyId")
                if pid is None:
                    continue
                if not starter and pid not in subs:
                    pass  # unused bench player still recorded, with no sub minutes
                lineups.append({"match_id": match_id, "player_id": pid, "team_id": int(tid),
                                "starter": starter, "shirt_number": p.get("shirtNumber"),
                                "position_code": ((p.get("position") or {}).get("code")),
                                "sub_in_min": (subs.get(pid) or {}).get("minute"),
                                "sub_out_min": (outs.get(pid) or {}).get("minute")})
    return {"matches": pd.DataFrame([match]), "match_referees": pd.DataFrame(refs),
            "referees": pd.DataFrame(ref_dim), "match_coaches": pd.DataFrame(coaches),
            "match_lineups": pd.DataFrame(lineups)}


def _fetch_match(match_id: int):
    try:
        return match_id, api.get_match_raw(match_id), None
    except api.WyscoutError as exc:
        return match_id, None, exc


def _fetch_season_matches(season_id: int):
    try:
        return api.get_season_matches(season_id)
    except (api.WyscoutError, KeyError):
        return pd.DataFrame()


PLAYED = ("Played", "Awarded")


def load_matches(con, seasons: dict[int, int], team_id: int | None, force: bool) -> None:
    allowed = None
    if team_id is not None:
        allowed = set(api.get_team_matches(team_id)["matchId"].astype(int))
    with ThreadPoolExecutor(WORKERS) as pool:
        lists = [m for m in pool.map(_fetch_season_matches, list(seasons)) if not m.empty]
    all_matches = concat(lists)
    ids = all_matches[all_matches["status"].isin(PLAYED)]["matchId"].astype(int).tolist()
    if allowed is not None:
        ids = [m for m in ids if m in allowed]
    done = set() if force else done_keys(con, "match_detail")
    todo = [m for m in ids if str(m) not in done]
    print(f"matches: {len(todo)}/{len(ids)} played matches to load")

    def write(results):
        parts: dict[str, list] = {}
        logs = []
        for match_id, raw, exc in results:
            if exc is not None:
                logs.append({"entity": "match_detail", "key": match_id, "status": "skipped",
                             "http_code": error_code(exc)})
                continue
            for table, df in parse_match(raw).items():
                parts.setdefault(table, []).append(df)
            logs.append({"entity": "match_detail", "key": match_id, "status": "ok", "rows": 1})
        for table, frames in parts.items():
            upsert(con, table, concat(frames))
        log_rows(con, logs)

    pipeline(todo, _fetch_match, write, batch=200, label="match details")
    backfill_dims(con)


def backfill_dims(con) -> None:
    """Fetch teams and players referenced by matches but missing from the dimension tables."""
    team_ids = [r[0] for r in con.execute("""
        SELECT DISTINCT t FROM (
            SELECT home_team_id t FROM wyscout.matches UNION SELECT away_team_id FROM wyscout.matches
            UNION SELECT team_id FROM wyscout.match_lineups)
        WHERE t IS NOT NULL AND t NOT IN (SELECT team_id FROM wyscout.teams)
          AND CAST(t AS VARCHAR) NOT IN (SELECT key FROM wyscout.ingest_log WHERE entity='team')""").fetchall()]
    player_ids = [r[0] for r in con.execute("""
        SELECT DISTINCT player_id FROM (
            SELECT player_id FROM wyscout.match_lineups UNION SELECT player_id FROM wyscout.player_match_stats
            UNION SELECT player_id FROM wyscout.events)
        WHERE player_id IS NOT NULL AND player_id NOT IN (SELECT player_id FROM wyscout.players)
          AND CAST(player_id AS VARCHAR) NOT IN (SELECT key FROM wyscout.ingest_log WHERE entity='player')
        """).fetchall()]
    print(f"backfill: {len(team_ids)} teams, {len(player_ids)} players")

    def fetch_team(tid):
        try:
            return tid, _team_rows(api.get_team(tid)), None
        except api.WyscoutError as exc:
            return tid, None, exc

    def fetch_player(pid):
        try:
            return pid, _player_row(api.get_player(pid)), None
        except api.WyscoutError as exc:
            return pid, None, exc

    for entity, table, ids, fetch in (("team", "teams", team_ids, fetch_team),
                                      ("player", "players", player_ids, fetch_player)):
        def write(results, entity=entity, table=table):
            upsert(con, table, pd.DataFrame([r for _, r, e in results if e is None]))
            log_rows(con, [{"entity": entity, "key": k, "status": "ok" if e is None else "skipped",
                            "http_code": error_code(e) if e else None} for k, _, e in results])
        pipeline(ids, fetch, write, batch=200, label=f"backfill {entity}s")


# ── Stage: match-data (stats, events, possessions, formations) ────────────────

STAT_RENAMES = {"competitionId": "comp_id", "matchId": "match_id", "teamId": "team_id",
                "playerId": "player_id", "seasonId": "season_id", "roundId": "round_id"}


def parse_events(events: pd.DataFrame, match_id: int, comp_id, season_id) -> dict[str, pd.DataFrame]:
    """Split flattened events into events / event_tags / possessions tables."""
    ev = snake_cols(events, {"id": "event_id", "matchPeriod": "period", "opponentTeam_id": "opponent_team_id",
                             "location_x": "loc_x", "location_y": "loc_y"})
    ev["match_id"], ev["comp_id"], ev["season_id"] = match_id, comp_id, season_id
    poss_cols = [c for c in ev.columns if c.startswith("possession_")]
    poss = pd.DataFrame()
    if "possession_id" in ev.columns:
        first = ev[ev["possession_id"].notna()].drop_duplicates("possession_id").copy()
        poss = pd.DataFrame({
            "match_id": match_id, "possession_id": first["possession_id"], "comp_id": comp_id,
            "season_id": season_id, "team_id": first.get("possession_team_id"),
            "formation": first.get("possession_team_formation"),
            "duration": pd.to_numeric(first.get("possession_duration"), errors="coerce"),
            "events_number": first.get("possession_events_number"),
            "start_period": first.get("period"), "start_minute": first.get("minute"),
            "start_second": first.get("second"),
            "start_x": first.get("possession_start_location_x"),
            "start_y": first.get("possession_start_location_y"),
            "end_x": first.get("possession_end_location_x"),
            "end_y": first.get("possession_end_location_y"),
            "types": first.get("possession_types"),
            "attack_flank": first.get("possession_attack_flank"),
            "attack_with_shot": first.get("possession_attack_with_shot"),
            "attack_with_shot_on_goal": first.get("possession_attack_with_shot_on_goal"),
            "attack_with_goal": first.get("possession_attack_with_goal"),
            "attack_xg": first.get("possession_attack_xg")})
        keep = {"possession_id", "possession_event_index"}
        ev = ev.drop(columns=[c for c in poss_cols if c not in keep])
    tags = ev[["match_id", "event_id", "type_secondary"]].explode("type_secondary") \
        if "type_secondary" in ev.columns else pd.DataFrame()
    if not tags.empty:
        tags = tags.dropna(subset=["type_secondary"]).rename(columns={"type_secondary": "tag"})
        tags = tags.drop_duplicates()
    return {"events": ev, "event_tags": tags, "possessions": poss}


def ensure_dim(con, table: str, key: str, id_col: str, values) -> dict:
    """Insert unseen values into a lookup table with the next integer ids; return {value: id}."""
    values = sorted({v for v in values if isinstance(v, str)})
    if not values:
        return {}
    known = dict(con.execute(f"SELECT {key}, {id_col} FROM wyscout.{table}").fetchall())
    nxt = max(known.values(), default=0)
    new = [v for v in values if v not in known]
    if new:
        rows = pd.DataFrame({key: new, id_col: range(nxt + 1, nxt + 1 + len(new))})
        upsert(con, table, rows)
        known.update(zip(rows[key], rows[id_col]))
    return known


# Formations come from video tagging: matches without video return this permanent 404.
NO_VIDEO = "No video"


def _retry(fn, attempts: int = 3, pause: float = 2.0):
    """Call fn(); the API sometimes answers 403/404 transiently, so retry before giving up."""
    for i in range(attempts):
        try:
            return fn()
        except api.WyscoutError as exc:
            if i + 1 == attempts or NO_VIDEO in str(exc):
                raise
            time.sleep(pause * (i + 1))


def _fetch_match_data(args):
    """Fetch the four per-match payloads independently; one failing call keeps the others."""
    match_id, comp_id, season_id, home, away = args
    out = {"match_id": match_id, "comp_id": comp_id, "season_id": season_id,
           "player_stats": pd.DataFrame(), "team_stats": pd.DataFrame(),
           "events": pd.DataFrame(), "formations": {}, "errors": []}
    parts = {
        "player_stats": lambda: api.get_match_player_stats(match_id),
        "team_stats": lambda: pd.concat([api.get_team_match_stats(match_id, t) for t in (home, away)],
                                        ignore_index=True),
        "events": lambda: api.get_match_events(match_id),
        "formations": lambda: api.get_match_formations(match_id),
    }
    for name, fn in parts.items():
        try:
            out[name] = _retry(fn)
        except api.WyscoutError as exc:
            if not (name == "formations" and NO_VIDEO in str(exc)):
                out["errors"].append(exc)
    return out


def parse_formations(raw: dict, match_id: int) -> pd.DataFrame:
    """Flatten /matches/{id}/formations: {teamId: {period: {startSec: {scheme: {...}}}}}.

    One row per (team, period, startSec); the innermost dict carries endSec and scheme.
    """
    rows = []
    for team_id, periods in raw.items():
        if not str(team_id).isdigit() or not isinstance(periods, dict):
            continue
        for period, starts in periods.items():
            for start_sec, schemes in (starts or {}).items():
                for scheme in schemes.values():
                    rows.append({"match_id": match_id, "team_id": int(team_id), "period": period,
                                 "start_sec": int(start_sec), "end_sec": scheme.get("endSec"),
                                 "formation": scheme.get("scheme")})
    return pd.DataFrame(rows)


def load_match_data(con, force: bool) -> None:
    matches = con.execute("""SELECT match_id, comp_id, season_id, home_team_id, away_team_id
                             FROM wyscout.matches WHERE has_data_available""").fetchall()
    done = set() if force else done_keys(con, "match_data")
    todo = [m for m in matches if str(m[0]) not in done]
    print(f"match-data: {len(todo)}/{len(matches)} matches with data to load")

    def write(results):
        ok = results  # every match is written with whatever parts succeeded
        logs = []
        if ok:
            ids = [o["match_id"] for o in ok]
            ps = concat([snake_cols(o["player_stats"], STAT_RENAMES) for o in ok])
            if not ps.empty:  # player stats carry no team id: take it from the lineups
                lineups = con.execute("SELECT match_id, player_id, team_id FROM wyscout.match_lineups "
                                      "WHERE match_id IN (SELECT UNNEST(?))", [ids]).df()
                ps = ps.drop(columns=["team_id"], errors="ignore").merge(lineups, on=["match_id", "player_id"], how="left")
            upsert(con, "player_match_stats", ps)
            upsert(con, "team_match_stats", concat([snake_cols(o["team_stats"], STAT_RENAMES) for o in ok]))
            parts = [parse_events(o["events"], o["match_id"], o["comp_id"], o["season_id"])
                     for o in ok if not o["events"].empty]
            if parts:
                events, tags = concat([p["events"] for p in parts]), concat([p["event_tags"] for p in parts])
                types = ensure_dim(con, "dim_event_types", "type_primary", "type_id", events["type_primary"])
                ensure_dim(con, "dim_event_tags", "tag", "tag_id", tags["tag"] if not tags.empty else [])
                events["type_id"] = events["type_primary"].map(types)
                con.execute("DELETE FROM wyscout.event_tags WHERE match_id IN (SELECT UNNEST(?))", [ids])
                upsert(con, "events", events)
                upsert(con, "event_tags", tags)
                upsert(con, "possessions", concat([p["possessions"] for p in parts]))
            upsert(con, "match_formations", concat([parse_formations(o["formations"], o["match_id"]) for o in ok]))
            # 'error' is not in done_keys, so matches with a failed call are retried on the next run
            logs += [{"entity": "match_data", "key": o["match_id"], "rows": len(o["events"]),
                      "status": "error" if o["errors"] else "ok",
                      "http_code": error_code(o["errors"][0]) if o["errors"] else None} for o in ok]
        log_rows(con, logs)

    pipeline(todo, _fetch_match_data, write, label="match data")
    backfill_dims(con)


# ── Stage: season-stats ───────────────────────────────────────────────────────

def load_season_stats(con, seasons: dict[int, int], force: bool) -> None:
    sids = list(seasons)
    for entity, table, fetch_fn, sql in (
        ("team_season_stats", "team_season_stats", api.get_team_season_stats,
         "SELECT team_id, comp_id, season_id FROM wyscout.team_seasons WHERE season_id IN (SELECT UNNEST(?))"),
        ("player_season_stats", "player_season_stats", api.get_player_season_stats,
         "SELECT DISTINCT player_id, comp_id, season_id FROM wyscout.player_seasons "
         "WHERE season_id IN (SELECT UNNEST(?))")):
        done = set() if force else done_keys(con, entity)
        todo = [r for r in con.execute(sql, [sids]).fetchall() if f"{r[0]}:{r[2]}" not in done]
        print(f"{entity}: {len(todo)} to load")

        def fetch(args, fetch_fn=fetch_fn):
            try:
                return args, fetch_fn(*args), None
            except api.WyscoutError as exc:
                return args, None, exc

        def write(results, entity=entity, table=table):
            upsert(con, table, concat([snake_cols(df, STAT_RENAMES) for _, df, e in results if e is None]))
            log_rows(con, [{"entity": entity, "key": f"{a[0]}:{a[2]}", "status": "ok" if e is None else "skipped",
                            "http_code": error_code(e) if e else None} for a, _, e in results])

        pipeline(todo, fetch, write, batch=200, label=entity)


# ── Stage: check ──────────────────────────────────────────────────────────────

CHECKS = {
    "matches without season": "SELECT count(*) FROM wyscout.matches m WHERE m.season_id NOT IN (SELECT season_id FROM wyscout.seasons)",
    "matches comp/season mismatch": "SELECT count(*) FROM wyscout.matches m JOIN wyscout.seasons s USING (season_id) WHERE m.comp_id <> s.comp_id",
    "events without match": "SELECT count(*) FROM wyscout.events e WHERE e.match_id NOT IN (SELECT match_id FROM wyscout.matches)",
    "events with unknown possession": "SELECT count(*) FROM wyscout.events e WHERE e.possession_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM wyscout.possessions p WHERE p.match_id=e.match_id AND p.possession_id=e.possession_id)",
    "event tags missing from dim": "SELECT count(*) FROM wyscout.event_tags WHERE tag NOT IN (SELECT tag FROM wyscout.dim_event_tags)",
    "player stats w/o player": "SELECT count(*) FROM wyscout.player_match_stats WHERE player_id NOT IN (SELECT player_id FROM wyscout.players)",
    "matches w/o team": "SELECT count(*) FROM wyscout.matches WHERE home_team_id NOT IN (SELECT team_id FROM wyscout.teams) OR away_team_id NOT IN (SELECT team_id FROM wyscout.teams)",
}


def run_checks(con) -> None:
    for table in ("competitions", "seasons", "teams", "players", "matches", "events", "possessions",
                  "event_tags", "player_match_stats", "team_match_stats"):
        n = con.execute(f"SELECT count(*) FROM wyscout.{table}").fetchone()[0]
        print(f"{table:22s} {n:>10,}")
    print("\nintegrity (all should be 0):")
    for name, sql in CHECKS.items():
        print(f"  {name:32s} {con.execute(sql).fetchone()[0]}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default=",".join(STAGES), help=f"comma list of: {', '.join(STAGES)}")
    ap.add_argument("--season-ids", help="comma list; default = every season loaded by dims")
    ap.add_argument("--team-id", type=int, help="only load matches involving this team")
    ap.add_argument("--force", action="store_true", help="refetch even if ingest_log says done")
    args = ap.parse_args()
    stages = args.stages.split(",")

    con = connect()
    if "dims" in stages:
        load_dims(con)
    q = "SELECT season_id, comp_id FROM wyscout.seasons"
    if args.season_ids:
        q += f" WHERE season_id IN ({','.join(str(int(s)) for s in args.season_ids.split(','))})"
    seasons = dict(con.execute(q).fetchall())
    if "rosters" in stages:
        load_rosters(con, seasons, args.force)
    if "matches" in stages:
        load_matches(con, seasons, args.team_id, args.force)
    if "match-data" in stages:
        load_match_data(con, args.force)
    if "season-stats" in stages:
        load_season_stats(con, seasons, args.force)
    if "check" in stages:
        run_checks(con)


if __name__ == "__main__":
    main()
