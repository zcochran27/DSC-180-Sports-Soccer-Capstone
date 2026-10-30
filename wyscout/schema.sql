-- Wyscout NCAA soccer schema for MotherDuck / DuckDB.
-- Ids are BIGINT, coordinates and stats DOUBLE. Idempotent: safe to run repeatedly.
-- Stat tables only declare their key columns; load_motherduck.py adds metric columns
-- (total_*, average_*, percent_*) as the API returns them.
-- DuckDB does not enforce foreign keys on upsert-heavy tables, so relationships are
-- documented in comments and checked by the integrity queries in load_motherduck.py.

CREATE SCHEMA IF NOT EXISTS wyscout;
SET search_path = 'wyscout';

-- ── Dimensions ───────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS areas (
    area_id      BIGINT PRIMARY KEY,
    alpha2code   VARCHAR,
    alpha3code   VARCHAR,
    name         VARCHAR
);

-- compId = Wyscout competition wyId. No gender filter in the API; gender is a column.
CREATE TABLE IF NOT EXISTS competitions (
    comp_id         BIGINT PRIMARY KEY,
    name            VARCHAR,
    gender          VARCHAR,
    type            VARCHAR,
    format          VARCHAR,
    category        VARCHAR,
    division_level  INTEGER,
    area_id         BIGINT,          -- -> areas
    gsm_id          BIGINT
);

-- Competition 1 -> N seasons. season_id is globally unique; comp_id says which competition.
-- (No extra UNIQUE constraints anywhere: DuckDB upserts need a single key per table.)
CREATE TABLE IF NOT EXISTS seasons (
    season_id   BIGINT PRIMARY KEY,
    comp_id     BIGINT NOT NULL,     -- -> competitions
    name        VARCHAR,
    start_date  DATE,
    end_date    DATE,
    active      BOOLEAN,
    gsm_id      BIGINT
);

CREATE TABLE IF NOT EXISTS rounds (
    round_id    BIGINT PRIMARY KEY,
    season_id   BIGINT,              -- -> seasons
    comp_id     BIGINT,              -- -> competitions
    name        VARCHAR,
    type        VARCHAR,
    start_date  DATE,
    end_date    DATE,
    gsm_id      BIGINT
);

-- Every school has separate men's and women's team ids.
CREATE TABLE IF NOT EXISTS teams (
    team_id         BIGINT PRIMARY KEY,
    name            VARCHAR,
    official_name   VARCHAR,
    gender          VARCHAR,
    type            VARCHAR,
    category        VARCHAR,
    city            VARCHAR,
    area_id         BIGINT,          -- -> areas
    image_url       VARCHAR,
    parent_team_id  BIGINT,          -- self reference, from the API's "children" lists
    gsm_id          BIGINT
);

-- Which teams played in which season (conference membership via comp_id).
CREATE TABLE IF NOT EXISTS team_seasons (
    team_id    BIGINT,
    season_id  BIGINT,
    comp_id    BIGINT,
    PRIMARY KEY (team_id, season_id)
);

CREATE TABLE IF NOT EXISTS players (
    player_id         BIGINT PRIMARY KEY,
    first_name        VARCHAR,
    middle_name       VARCHAR,
    last_name         VARCHAR,
    short_name        VARCHAR,
    birth_date        DATE,
    birth_area_id     BIGINT,
    passport_area_id  BIGINT,
    height            DOUBLE,
    weight            DOUBLE,
    foot              VARCHAR,
    role_code2        VARCHAR,
    role_code3        VARCHAR,
    role_name         VARCHAR,
    gender            VARCHAR,
    status            VARCHAR,
    image_url         VARCHAR,
    gsm_id            BIGINT
);

-- Roster per season; player -> team is time-varying so it lives here, not on players.
CREATE TABLE IF NOT EXISTS player_seasons (
    player_id  BIGINT,
    season_id  BIGINT,
    team_id    BIGINT,
    comp_id    BIGINT,
    PRIMARY KEY (player_id, season_id, team_id)
);

CREATE TABLE IF NOT EXISTS coaches (
    coach_id         BIGINT PRIMARY KEY,
    first_name       VARCHAR,
    last_name        VARCHAR,
    short_name       VARCHAR,
    birth_date       DATE,
    birth_area_id    BIGINT,
    passport_area_id BIGINT,
    current_team_id  BIGINT,
    gender           VARCHAR,
    status           VARCHAR
);

CREATE TABLE IF NOT EXISTS referees (
    referee_id       BIGINT PRIMARY KEY,
    first_name       VARCHAR,
    last_name        VARCHAR,
    short_name       VARCHAR,
    birth_date       DATE,
    birth_area_id    BIGINT,
    passport_area_id BIGINT,
    gender           VARCHAR,
    status           VARCHAR
);

-- ── Match facts ──────────────────────────────────────────────────────────────

-- Loaded from /matches/{id} (detail); hasDataAvailable is unreliable on list endpoints.
CREATE TABLE IF NOT EXISTS matches (
    match_id            BIGINT PRIMARY KEY,
    comp_id             BIGINT,      -- -> competitions
    season_id           BIGINT,      -- -> seasons
    round_id            BIGINT,      -- -> rounds
    gameweek            INTEGER,
    date_utc            TIMESTAMP,
    status              VARCHAR,
    duration            VARCHAR,
    label               VARCHAR,
    home_team_id        BIGINT,
    away_team_id        BIGINT,
    home_score          INTEGER,
    away_score          INTEGER,
    winner_team_id      BIGINT,
    has_data_available  BOOLEAN,
    ingested_at         TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS match_referees (
    match_id    BIGINT,
    referee_id  BIGINT,
    role        VARCHAR,             -- referee / firstAssistant / secondAssistant / ...
    PRIMARY KEY (match_id, referee_id, role)
);

CREATE TABLE IF NOT EXISTS match_coaches (
    match_id  BIGINT,
    team_id   BIGINT,
    coach_id  BIGINT,
    PRIMARY KEY (match_id, team_id)
);

CREATE TABLE IF NOT EXISTS match_lineups (
    match_id      BIGINT,
    player_id     BIGINT,
    team_id       BIGINT,
    starter       BOOLEAN,
    minutes       DOUBLE,
    position_code VARCHAR,
    shirt_number  INTEGER,
    sub_in_min    DOUBLE,
    sub_out_min   DOUBLE,
    PRIMARY KEY (match_id, player_id)
);

CREATE TABLE IF NOT EXISTS match_formations (
    match_id   BIGINT,
    team_id    BIGINT,
    period     VARCHAR,              -- 1H / 2H / E1 / E2 / P
    start_sec  INTEGER,              -- seconds from the start of the period
    end_sec    INTEGER,
    formation  VARCHAR,              -- scheme, e.g. 4-3-3
    PRIMARY KEY (match_id, team_id, period, start_sec)
);

-- ── Stats (wide: metric columns added on load) ───────────────────────────────

CREATE TABLE IF NOT EXISTS team_match_stats (
    match_id  BIGINT,
    team_id   BIGINT,
    PRIMARY KEY (match_id, team_id)
);

CREATE TABLE IF NOT EXISTS player_match_stats (
    match_id          BIGINT,
    player_id         BIGINT,
    team_id           BIGINT,
    comp_id           BIGINT,
    season_id         BIGINT,
    round_id          BIGINT,
    position_code     VARCHAR,
    position_name     VARCHAR,
    position_percent  DOUBLE,
    PRIMARY KEY (match_id, player_id)
);

-- A team/player can have one row per competition in a season (conference + tournament).
CREATE TABLE IF NOT EXISTS team_season_stats (
    team_id    BIGINT,
    comp_id    BIGINT,
    season_id  BIGINT,
    PRIMARY KEY (team_id, comp_id, season_id)
);

CREATE TABLE IF NOT EXISTS player_season_stats (
    player_id  BIGINT,
    comp_id    BIGINT,
    season_id  BIGINT,
    PRIMARY KEY (player_id, comp_id, season_id)
);

-- ── Events ───────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS dim_event_types (
    type_id       INTEGER PRIMARY KEY,
    type_primary  VARCHAR
);

CREATE TABLE IF NOT EXISTS dim_event_tags (
    tag          VARCHAR PRIMARY KEY,
    tag_id       INTEGER,
    category     VARCHAR,
    description  VARCHAR
);

-- One row per event. Nested blocks (pass_*, shot_*, carry_*, duels, infraction_*) are
-- flattened to typed columns added on load. type_secondary is the lossless tag list.
CREATE TABLE IF NOT EXISTS events (
    match_id               BIGINT,
    event_id               BIGINT,
    comp_id                BIGINT,
    season_id              BIGINT,
    period                 VARCHAR,
    minute                 INTEGER,
    second                 INTEGER,
    video_timestamp        VARCHAR,
    team_id                BIGINT,
    opponent_team_id       BIGINT,
    player_id              BIGINT,
    type_id                INTEGER,  -- -> dim_event_types
    type_primary           VARCHAR,
    type_secondary         VARCHAR[],
    loc_x                  DOUBLE,
    loc_y                  DOUBLE,
    possession_id          BIGINT,   -- -> possessions (match_id, possession_id)
    possession_event_index INTEGER,
    PRIMARY KEY (match_id, event_id)
);

CREATE TABLE IF NOT EXISTS event_tags (
    match_id  BIGINT,
    event_id  BIGINT,
    tag       VARCHAR,               -- -> dim_event_tags
    PRIMARY KEY (match_id, event_id, tag)
);

-- Possession ids are only unique within a match.
CREATE TABLE IF NOT EXISTS possessions (
    match_id                  BIGINT,
    possession_id             BIGINT,
    comp_id                   BIGINT,
    season_id                 BIGINT,
    team_id                   BIGINT,
    formation                 VARCHAR,
    duration                  DOUBLE,
    events_number             INTEGER,
    start_period              VARCHAR,
    start_minute              INTEGER,
    start_second              INTEGER,
    start_x                   DOUBLE,
    start_y                   DOUBLE,
    end_x                     DOUBLE,
    end_y                     DOUBLE,
    types                     VARCHAR[],
    attack_flank              VARCHAR,
    attack_with_shot          BOOLEAN,
    attack_with_shot_on_goal  BOOLEAN,
    attack_with_goal          BOOLEAN,
    attack_xg                 DOUBLE,
    PRIMARY KEY (match_id, possession_id)
);

-- ── Ops ──────────────────────────────────────────────────────────────────────

-- Makes the loader resumable and remembers 403 (not licensed) / 404 so they aren't retried.
CREATE TABLE IF NOT EXISTS ingest_log (
    entity      VARCHAR,
    key         VARCHAR,
    status      VARCHAR,             -- ok / skipped / error
    http_code   INTEGER,
    rows        INTEGER,
    fetched_at  TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (entity, key)
);

-- ── Season rollups across competitions (built by season_rollup.py) ───────────

-- One row per season: parsed year / Fall-Spring term. kind: regular / friendly / cancelled.
CREATE TABLE IF NOT EXISTS season_terms (
    season_id    BIGINT PRIMARY KEY,
    comp_id      BIGINT,
    name         VARCHAR,
    year         INTEGER,
    term         VARCHAR,            -- Fall / Spring / Full / Unknown
    kind         VARCHAR,
    term_source  VARCHAR             -- name or dates
);

-- How each average_* / percent_* column is recombined across competitions, with its fit.
CREATE TABLE IF NOT EXISTS stat_formulas (
    scope        VARCHAR,            -- team / player
    stat_col     VARCHAR,
    method       VARCHAR,            -- per_exposure / ratio / weighted_mean / exposure_weighted
    numerator    VARCHAR,
    denominator  VARCHAR,
    weight       VARCHAR,
    fit          DOUBLE,
    PRIMARY KEY (scope, stat_col)
);

-- One row per entity per year+term, all competitions combined; stat columns added on load.
CREATE TABLE IF NOT EXISTS team_term_stats (
    team_id    BIGINT,
    year       INTEGER,
    term       VARCHAR,
    n_comps    INTEGER,
    n_seasons  INTEGER,
    comp_ids   VARCHAR[],
    exposure   DOUBLE,
    PRIMARY KEY (team_id, year, term)
);

CREATE TABLE IF NOT EXISTS player_term_stats (
    player_id  BIGINT,
    year       INTEGER,
    term       VARCHAR,
    n_comps    INTEGER,
    n_seasons  INTEGER,
    comp_ids   VARCHAR[],
    exposure   DOUBLE,
    PRIMARY KEY (player_id, year, term)
);
