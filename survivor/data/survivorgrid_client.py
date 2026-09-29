"""Scrape SurvivorGrid's weekly NFL survivor pick-percentage grid.

No official API or export -- survivorgrid.com/{year}/{week} is a plain,
server-rendered HTML table (confirmed via a raw pull, not a JS-only page),
so a direct requests + BeautifulSoup parse is reliable without a browser.

Two uses per the plan: Week 4's own national distribution feeds the
popularity model directly, and past seasons' grids fit the softmax
temperature (Phase 4).
"""

from __future__ import annotations

import re
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from survivor.data.storage import DEFAULT_STORE_ROOT
from survivor.data.team_keys import to_abbreviation

BASE_URL = "https://www.survivorgrid.com"
USER_AGENT = "Mozilla/5.0 (compatible; survivor-pool-research/1.0)"
TEAM_PATTERN = re.compile(r"[A-Z]+")
HTML_CACHE_DIR = DEFAULT_STORE_ROOT / "survivorgrid_html"


def fetch_week_html(year: int, week: int) -> str:
    response = requests.get(f"{BASE_URL}/{year}/{week}", headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    return response.text


def fetch_week_html_cached(year: int, week: int, cache_dir: Path | None = HTML_CACHE_DIR) -> tuple[str, bool]:
    """fetch_week_html, read from / saved to a disk cache. Returns (html, served_from_cache).

    Only for pages that no longer change -- a past season, or a week
    whose games have all been played -- since a cached page is never
    refetched. Callers can skip their courtesy delay when the second value
    is True. cache_dir=None bypasses the cache entirely (live pull, nothing
    written).
    """
    if cache_dir is None:
        return fetch_week_html(year, week), False
    path = cache_dir / f"{year}_{week}.html"
    if path.exists():
        return path.read_text(encoding="utf-8"), True
    html = fetch_week_html(year, week)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".html.tmp")
    tmp_path.write_text(html, encoding="utf-8")
    tmp_path.replace(path)  # atomic, so an interrupted write never leaves a truncated cache file
    return html, False


def parse_pick_grid(html: str) -> pd.DataFrame:
    """Parse the grid table into one row per team: expected_value, win_probability, pick_percentage, result.

    A team's cell reads e.g. "PHI" (upcoming week) or "PHI\xa0(W)" (a
    played week, with the result appended in a resultW/resultL span) -- the
    leading run of capital letters is the team abbreviation either way.
    result is "W", "L", or None (upcoming week, or a historical week with
    no game -- a bye).
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="grid")
    if table is None:
        raise ValueError("could not find the pick grid table in the page")

    rows = []
    for tr in table.find("tbody").find_all("tr"):
        cells = tr.find_all("td")
        team_match = TEAM_PATTERN.match(cells[3].get_text())
        if team_match is None:
            continue
        result_span = cells[3].find("span", class_=["resultW", "resultL"])
        result = result_span["class"][0][-1] if result_span else None  # "resultW" -> "W", "resultL" -> "L"
        rows.append(
            {
                "team": to_abbreviation(team_match.group()),
                "expected_value": _parse_float(cells[0].get_text(strip=True)),
                "win_probability": _parse_percent(cells[1].get_text(strip=True)),
                "pick_percentage": _parse_percent(cells[2].get_text(strip=True)),
                "result": result,
            }
        )
    return pd.DataFrame(rows, columns=["team", "expected_value", "win_probability", "pick_percentage", "result"])


def parse_schedule_grid(html: str, start_week: int) -> pd.DataFrame:
    """Parse the same grid's per-week opponent/spread columns into one row per (team, week).

    The page for /{year}/{week} shows one "gc" column per remaining week,
    from start_week through 18, one row per team. A cell reads e.g.
    "ARI<br><span>-9.5</span>" (team is home, favored by 9.5) or
    "@BUF<br><span>+3</span>" (team is away, a 3-point underdog) or "BYE".
    spread is always the row team's own spread, matching
    ratings.fit_team_ratings' home_spread convention when is_home is True.

    Combined with parse_pick_grid's win_probability/pick_percentage for
    start_week specifically, this is enough to fit ratings and run the
    simulator from a single page pull -- no other odds source needed,
    including for weeks that have already passed this season (useful for
    backtesting what the pipeline would have recommended at the time).
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="grid")
    if table is None:
        raise ValueError("could not find the pick grid table in the page")

    rows = []
    for tr in table.find("tbody").find_all("tr"):
        cells = tr.find_all("td")
        team_match = TEAM_PATTERN.match(cells[3].get_text())
        if team_match is None:
            continue
        team = to_abbreviation(team_match.group())

        # The trailing "fv" (future value / star rating) cell is only
        # present when there are future weeks left to rate -- a season's
        # final week page (e.g. /2023/18) omits it entirely, so detect it
        # rather than assume a fixed position (confirmed on real data: that
        # page has exactly 5 cells total, all of them real).
        has_trailing_fv_cell = "fv" in cells[-1].get("class", [])
        week_cells = cells[4:-1] if has_trailing_fv_cell else cells[4:]

        for offset, cell in enumerate(week_cells):
            week = start_week + offset
            if "bye" in cell.get("class", []):
                rows.append({"team": team, "week": week, "opponent": None, "is_home": None,
                             "spread": None, "is_bye": True})
                continue
            is_home = "rd" not in cell.get("class", [])
            opponent_match = TEAM_PATTERN.search(cell.get_text())
            spread_span = cell.find("span", class_="spread")
            rows.append(
                {
                    "team": team,
                    "week": week,
                    "opponent": to_abbreviation(opponent_match.group()) if opponent_match else None,
                    "is_home": is_home,
                    "spread": _parse_float(spread_span.get_text(strip=True)) if spread_span else None,
                    "is_bye": False,
                }
            )
    return pd.DataFrame(rows, columns=["team", "week", "opponent", "is_home", "spread", "is_bye"])


def dedupe_schedule_games(schedule_grid: pd.DataFrame) -> pd.DataFrame:
    """One row per real (week, game), from a grid with one row per (team, week).

    Prefers the row marked is_home=True. A neutral-site game has
    SurvivorGrid mark *both* sides "rd" (away) -- confirmed on the real
    LAR @ SF Week 1, 2026 game, which silently vanished from a naive
    is_home-only filter, dropping LAR (a legitimate 64.1%
    survival-probability candidate) entirely. Falls back to either row
    (arbitrary designated "home") for those, at the minor, unavoidable cost
    of attributing a small amount of home-field advantage to a team that
    didn't actually have it that week.
    """
    games = schedule_grid[~schedule_grid["is_bye"]].copy()
    games["game_key"] = games.apply(lambda r: (r["week"], frozenset({r["team"], r["opponent"]})), axis=1)

    rows = []
    for _, group in games.groupby("game_key"):
        home_rows = group[group["is_home"] == True]  # noqa: E712 (is_home is nullable, `is True` misses it)
        row = home_rows.iloc[0] if not home_rows.empty else group.iloc[0]
        rows.append({"week": row["week"], "home_team": row["team"], "away_team": row["opponent"], "home_spread": row["spread"]})
    return pd.DataFrame(rows, columns=["week", "home_team", "away_team", "home_spread"])


def fetch_season_to_date_games(
    year: int, through_week: int, delay_seconds: float = 1.0, cache_dir: Path | None = HTML_CACHE_DIR
) -> pd.DataFrame:
    """Every already-played week's real closing spreads, weeks 1..through_week.

    One fetch per week, each week's own page for that week's own real
    closing line (not a later page's stale lookahead projection -- see
    parse_schedule_grid's docstring for why that distinction matters). Built
    for fitting ratings.fit_team_ratings on multiple weeks of real market
    data instead of just the current week alone: a single week is just 16
    games informing 32 teams' ratings (severely underdetermined, see
    fit_team_ratings' docstring), so its projections for weeks beyond the
    real lookahead window carry real noise. Accumulating the whole season so
    far cuts that noise roughly by sqrt(games per team) -- confirmed on real
    2026 data (scripts/validate_ratings.py): fitting on Weeks 1-3 combined
    and projecting Week 4 brought lookahead MAE from 2.39 (Week 3 alone)
    down to 2.01, and combined with a small ridge, to about 1.73.

    Paced at 1 request/second out of courtesy -- unofficial, undocumented
    source with no published rate limit. Pages are cached on disk
    (cache_dir; None disables it): through_week should be a week whose games have all
    been played, so its closing spreads are final and never need
    refetching -- a later run only fetches the weeks it hasn't seen.
    """
    frames = []
    for week in range(1, through_week + 1):
        html, from_cache = fetch_week_html_cached(year, week, cache_dir)
        schedule_grid = parse_schedule_grid(html, start_week=week)
        all_games = dedupe_schedule_games(schedule_grid)
        week_games = all_games[all_games["week"] == week].dropna()
        frames.append(week_games)
        if not from_cache:
            time.sleep(delay_seconds)
    return pd.concat(frames, ignore_index=True)


def _parse_percent(text: str) -> float | None:
    text = text.strip()
    try:
        return float(text.rstrip("%")) / 100.0
    except ValueError:
        return None  # blank cell placeholder -- seen as "-", "--", "N/A" across seasons


def _parse_float(text: str) -> float | None:
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        return None  # blank cell placeholder -- seen as "-", "--", "N/A" across seasons


def fetch_pick_grid(year: int, week: int) -> pd.DataFrame:
    """One live pull for a single year/week, tagged with that year and week."""
    return _tag_pick_grid(parse_pick_grid(fetch_week_html(year, week)), year, week)


def _tag_pick_grid(df: pd.DataFrame, year: int, week: int) -> pd.DataFrame:
    df.insert(0, "week", week)
    df.insert(0, "year", year)
    return df


def fetch_historical_pick_grids(years: range, weeks: range, delay_seconds: float = 1.0) -> pd.DataFrame:
    """Pull multiple year/week grids for fitting the popularity model.

    delay_seconds throttles requests -- this is an unofficial, undocumented
    source with no published rate limit, so pace pulls to be a considerate
    scraper rather than to satisfy any stated quota.
    """
    frames = []
    for year in years:
        for week in weeks:
            try:
                # past seasons never change, so they're cached; the current one still is live
                html, from_cache = fetch_week_html_cached(year, week, HTML_CACHE_DIR if year < date.today().year else None)
            except requests.HTTPError:
                continue  # e.g. a week/year combination that doesn't exist
            frames.append(_tag_pick_grid(parse_pick_grid(html), year, week))
            if not from_cache:
                time.sleep(delay_seconds)
    return pd.concat(frames, ignore_index=True)
