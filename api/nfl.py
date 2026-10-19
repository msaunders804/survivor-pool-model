"""NFL schedule facts for the web app: this week's matchups, and which pool week it is.

Same free ESPN scoreboard survivor.data.schedule_client already uses for the
model. Pulls are cached in memory, so pages can ask freely without hitting
ESPN on every load; if ESPN is unreachable, callers get an exception and
fall back to whatever's stored (the week never changes on a failed pull).
"""
from __future__ import annotations

import re
import time
from datetime import datetime, timedelta, timezone

import requests

from survivor.data.team_keys import to_abbreviation
from survivor.data.schedule_client import (
    SCOREBOARD_URL, add_effective_lock_times, compute_bye_teams, fetch_week_schedule, parse_week_schedule,
)

GAMES_TTL_SECONDS = 30 * 60
RESULTS_TTL_SECONDS = 10 * 60
CALENDAR_TTL_SECONDS = 12 * 60 * 60
# ESPN's NFL week runs Wednesday 3am ET to the next Wednesday. The pool's week turns a
# day earlier -- Tuesday 3am ET, once Monday night's game is over -- so a Tuesday
# availability paste lands on the new week.
POOL_WEEK_LEAD = timedelta(days=1)

_cache: dict[tuple, tuple[float, object]] = {}


def _cached(key: tuple, ttl: float, fetch):
    hit = _cache.get(key)
    if hit is not None and time.monotonic() - hit[0] < ttl:
        return hit[1]
    value = fetch()
    _cache[key] = (time.monotonic(), value)
    return value


def _utc(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(timezone.utc)


def week_games(week: int, year: int) -> dict:
    """{'week', 'games': [{home, away, kickoff, lock}], 'byes': [...]} for one week."""
    def fetch():
        games = add_effective_lock_times(parse_week_schedule(fetch_week_schedule(week, year), week))
        return {
            "week": week,
            "games": [
                {"home": g.home_team, "away": g.away_team, "kickoff": g.commence_time,
                 "lock": g.effective_lock_time.isoformat()}
                for g in games.sort_values("commence_time").itertuples()
            ],
            "byes": compute_bye_teams(games),
        }
    return _cached(("games", week, year), GAMES_TTL_SECONDS, fetch)


def week_results(week: int, year: int) -> dict[str, str]:
    """{team: 'won' | 'lost' | 'tie'} for that week's games that are final. Games still to be
    played or in progress are left out, so callers can't mistake them for a result."""
    return _cached(("results", week, year), RESULTS_TTL_SECONDS, lambda: parse_results(fetch_week_schedule(week, year)))


def parse_results(raw: dict) -> dict[str, str]:
    results = {}
    for event in raw.get("events", []):
        competition = event["competitions"][0]
        if not competition.get("status", {}).get("type", {}).get("completed"):
            continue
        teams = competition["competitors"]
        scores = [float(t.get("score") or 0) for t in teams]
        for team, score, other in zip(teams, scores, reversed(scores)):
            outcome = "tie" if score == other else "won" if score > other else "lost"
            results[to_abbreviation(team["team"]["displayName"])] = outcome
    return results


def season_calendar(year: int) -> list[tuple[int, datetime, datetime]]:
    """[(week, start, end), ...] for the regular season, from ESPN's own calendar."""
    def fetch():
        response = requests.get(SCOREBOARD_URL, params={"seasontype": 2, "year": year}, timeout=30)
        response.raise_for_status()
        return parse_calendar(response.json())
    return _cached(("calendar", year), CALENDAR_TTL_SECONDS, fetch)


def parse_calendar(raw: dict) -> list[tuple[int, datetime, datetime]]:
    weeks = []
    for season_type in raw.get("leagues", [{}])[0].get("calendar", []):
        if not isinstance(season_type, dict) or str(season_type.get("value")) != "2":
            continue  # regular season only
        for entry in season_type.get("entries", []):
            number = re.search(r"\d+", str(entry.get("value") or entry.get("label", "")))
            if number:
                weeks.append((int(number.group()), _utc(entry["startDate"]), _utc(entry["endDate"])))
    return weeks


def pool_week_at(calendar: list[tuple[int, datetime, datetime]], now: datetime) -> int | None:
    """The pool week at `now` (None in the offseason)."""
    shifted = now + POOL_WEEK_LEAD
    for week, start, end in calendar:
        if start <= shifted <= end:
            return week
    return None


def advance_league_week(db, league, now: datetime | None = None, calendar=None) -> bool:
    """Move league.current_week forward to this pool week. Never moves it back, and
    never past the league's last week, so a manual setting ahead of the calendar
    stands. Returns True if it changed. A failed ESPN pull leaves the week alone."""
    try:
        calendar = calendar if calendar is not None else season_calendar(league.year)
    except Exception:
        return False
    week = pool_week_at(calendar, now or datetime.now(timezone.utc))
    if week is None or week <= (league.current_week or 0) or week > league.week_end:
        return False
    league.current_week = week
    db.commit()
    return True
