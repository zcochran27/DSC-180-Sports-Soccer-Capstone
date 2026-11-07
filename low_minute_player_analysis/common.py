"""Shared basics: the database connection and the position-group mapping.

    from low_minute_player_analysis import common
    con = common.connect()
"""
import os

import duckdb
from dotenv import load_dotenv

load_dotenv()

# Wyscout position codes (one per match) grouped into the 7 tactical groups used everywhere.
TACTICAL_MAPPING = {
    "gk": "GK",
    "lb": "Outside Backs", "rb": "Outside Backs", "lwb": "Outside Backs", "rwb": "Outside Backs",
    "rb5": "Outside Backs", "lb5": "Outside Backs",
    "cb": "Center Backs", "lcb": "Center Backs", "rcb": "Center Backs",
    "lcb3": "Center Backs", "rcb3": "Center Backs",
    "lw": "Wingers", "rw": "Wingers", "lwf": "Wingers", "rwf": "Wingers",
    "cf": "Center Forwards",
    "amf": "Attacking Midfielders", "lamf": "Attacking Midfielders", "ramf": "Attacking Midfielders",
    "dmf": "Central Midfielders", "ldmf": "Central Midfielders", "rdmf": "Central Midfielders",
    "lcmf": "Central Midfielders", "rcmf": "Central Midfielders",
    "lcmf3": "Central Midfielders", "rcmf3": "Central Midfielders",
}

# The 7 group names, each once, in the order above.
GROUPS = list(dict.fromkeys(TACTICAL_MAPPING.values()))


def connect():
    """Open a MotherDuck connection using MOTHERDUCK_TOKEN / MOTHERDUCK_DB from the .env file."""
    database = os.getenv("MOTHERDUCK_DB", "wyscout_dev")
    token = os.getenv("MOTHERDUCK_TOKEN")
    return duckdb.connect(f"md:{database}?motherduck_token={token}")
