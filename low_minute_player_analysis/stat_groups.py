"""Read wyscout/STAT_GROUPS.md and return the stat group of every player volume stat.

A stat can belong to several groups in that file (on purpose). Only player-applicable volume stats are kept:
"Team and player" / "Player only" raw stats and engineered `total_<name>` counts with scope both/player.
Team-only stats and the groups Results, Territory & Dominance and Playing Time (no per-90 volume stats) are skipped.

Main function for the notebooks:  stat_to_groups(stems) -> ({group: [stats]}, [stats in no group])
"""
import re
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "wyscout" / "STAT_GROUPS.md"
SKIP_GROUPS = {"Overview", "Results", "Territory & Dominance", "Playing Time"}

# STAT_GROUPS.md lists some stats in two groups on purpose. For the reconstruction each stat is
# transported once, so these stats are assigned to the group named here (user-approved mapping).
PRIMARY_GROUP = {
    "accelerations": "Dribbling",
    "aerial_duels": "Aerial", "aerial_duels_won": "Aerial",
    "defensive_duels": "Defending", "defensive_duels_won": "Defending",
    "new_defensive_duels_won": "Defending",
    "loose_ball_duels": "Duels", "loose_ball_duels_won": "Duels",
    "offensive_duels": "Dribbling", "offensive_duels_won": "Dribbling",
    "new_offensive_duels_won": "Dribbling",
    "failed_offensive_duels": "Ball Security", "failed_passes": "Ball Security",
    "fouls_suffered": "Discipline",
    "head_shots": "Shooting",
    "progressive_run": "Progression",
}


def load_stat_groups(path=PATH):
    """{group name: [stat names]} exactly as the markdown file lists them (a stat may appear in two groups)."""
    groups = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):                      # a new group heading
            current = line[3:].strip()
            if current not in SKIP_GROUPS:
                groups[current] = []
            continue
        if current is None or current in SKIP_GROUPS:
            continue

        raw_line = re.match(r"- \*\*(Team and player|Player only):\*\*(.*)", line)
        if raw_line:                                    # raw stats: names in backticks
            groups[current] += re.findall(r"`([a-z0-9_]+)`", raw_line.group(2))
            continue

        if line.startswith("| `total_"):                # engineered stats: table rows, scope in 3rd column
            cells = [c.strip() for c in line.strip("|").split("|")]
            if cells[2] in ("both", "player"):
                names = re.findall(r"`(total_[a-z0-9_]+)`", cells[0])
                groups[current] += [name[len("total_"):] for name in names]

    # remove repeated stats inside a group and drop empty groups
    return {g: list(dict.fromkeys(stats)) for g, stats in groups.items() if stats}


def load_single_groups(path=PATH):
    """Like load_stat_groups, but every stat appears in exactly one group (see PRIMARY_GROUP)."""
    multi = load_stat_groups(path)

    # which groups does each stat appear in?
    groups_of_stat = {}
    for group, stats in multi.items():
        for stat in stats:
            groups_of_stat.setdefault(stat, []).append(group)

    # pick one group per stat
    chosen = {}
    for stat, groups in groups_of_stat.items():
        if len(groups) == 1:
            chosen[stat] = groups[0]
        elif PRIMARY_GROUP.get(stat) in groups:
            chosen[stat] = PRIMARY_GROUP[stat]
        else:
            raise ValueError(f"{stat} is in {groups} but has no valid entry in PRIMARY_GROUP")

    single = {}
    for group, stats in multi.items():
        kept = [s for s in stats if chosen[s] == group]
        if kept:
            single[group] = kept
    return single


def stat_to_groups(stems):
    """({group: stats present in `stems`}, stats that are in no group). One group per stat."""
    stems_set = set(stems)
    groups = {}
    for group, stats in load_single_groups().items():
        present = [s for s in stats if s in stems_set]
        if present:
            groups[group] = present

    covered = {s for stats in groups.values() for s in stats}
    leftover = [s for s in stems if s not in covered]
    return groups, leftover
