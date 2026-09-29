"""Parse a pool's per-team availability table (how many live entries can still pick each team).

Splash shows, for the current week, one row per team (or per group of teams
with identical counts) such as

    1   49ers   1,123 / 2,463   46%
    22-32   Buccaneers, Chargers, Saints, ...   2,463 / 2,463   100%

Only the "available / live entries" count matters here; ranks and
percentages are ignored, and lines that aren't table rows (headings, the
takeaway blurb, the source line) are skipped. Feed the result to
survivor.simulation.rival_field.RivalFieldState.from_availability.
"""

from __future__ import annotations

import re
from pathlib import Path

from survivor.data.team_keys import ABBREVIATION_TO_FULL_NAME, to_abbreviation

_ROW_TAIL = re.compile(r"([\d,]+)\s*/\s*([\d,]+)\s+\d+%\s*$")
_RANK_PREFIX = re.compile(r"^\d+(?:[–—-]\d+)?\s+")  # "1 ", "22-32 ", or a range whose dash was dropped ("2232 ")
_NICKNAME_TO_ABBREVIATION = {full.split()[-1]: abbr for abbr, full in ABBREVIATION_TO_FULL_NAME.items()}


def _team_abbreviation(name: str) -> str:
    name = name.strip()
    if name in _NICKNAME_TO_ABBREVIATION:
        return _NICKNAME_TO_ABBREVIATION[name]
    return to_abbreviation(name)  # full names and abbreviations


def parse_availability(text: str) -> tuple[int, dict[str, int]]:
    """(live entries, {team abbreviation: entries that can still pick it}) from a pasted availability table.

    Raises if the rows disagree on the live-entry count, name a team more
    than once, or don't cover all 32 teams -- a truncated paste would
    otherwise read as "nobody has used the missing teams".
    """
    available: dict[str, int] = {}
    totals: set[int] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        tail = _ROW_TAIL.search(line)
        if tail is None:
            continue
        totals.add(int(tail.group(2).replace(",", "")))
        count = int(tail.group(1).replace(",", ""))
        names = _RANK_PREFIX.sub("", line[: tail.start()].strip())
        for name in names.split(","):
            try:
                team = _team_abbreviation(name)
            except KeyError as exc:
                raise ValueError(f"unrecognized team {name.strip()!r} in row: {line!r}") from exc
            if team in available:
                raise ValueError(f"team {team} appears more than once")
            available[team] = count

    if not available:
        raise ValueError("no availability rows found")
    if len(totals) != 1:
        raise ValueError(f"rows disagree on the number of live entries: {sorted(totals)}")
    missing = sorted(set(ABBREVIATION_TO_FULL_NAME) - set(available))
    if missing:
        raise ValueError(f"availability table is missing teams: {missing}")
    return totals.pop(), available


def load_availability(path: str | Path) -> tuple[int, dict[str, int]]:
    return parse_availability(Path(path).read_text(encoding="utf-8"))
