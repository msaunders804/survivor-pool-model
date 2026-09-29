"""Backfill real week-by-week rating history from past seasons, to calibrate
DEFAULT_WEEKLY_RATING_STD from actual data instead of the current placeholder.

One fetch per (year, week): each page's OWN first "gc" column is that
week's real closing line at the time (confirmed by comparing the Week 1,
2026 page's forward-looking Week 2 projection against the Week 2 page's
actual Week 2 line -- they differed, -7.0 vs -6.5, so a later page's
early columns are NOT a stale future projection, they're the real market
line for that specific week). Free, keyless source, paced at 1
request/second out of courtesy.

Run: .venv/bin/python scripts/backfill_rating_history.py
"""

import time

import requests

from survivor.data.rating_history import record_weekly_ratings
from survivor.data.survivorgrid_client import dedupe_schedule_games, fetch_week_html_cached, parse_schedule_grid
from survivor.probability.ratings import DEFAULT_RIDGE, fit_team_ratings

YEARS = range(2023, 2026)  # 2023-2025
WEEKS = range(1, 19)
DELAY_SECONDS = 1.0


def main() -> None:
    recorded = 0
    skipped = 0
    for year in YEARS:
        for week in WEEKS:
            try:
                html, from_cache = fetch_week_html_cached(year, week)  # past seasons: never change
            except requests.HTTPError:
                skipped += 1
                continue

            schedule_grid = parse_schedule_grid(html, start_week=week)
            all_games = dedupe_schedule_games(schedule_grid)
            week_games = all_games[all_games["week"] == week][["home_team", "away_team", "home_spread"]].dropna()

            if len(week_games) < 2:  # not enough games to fit anything meaningful
                skipped += 1
                if not from_cache:
                    time.sleep(DELAY_SECONDS)
                continue

            fit = fit_team_ratings(week_games, ridge=DEFAULT_RIDGE)
            record_weekly_ratings(f"survivorgrid_{year}", year, week, fit.ratings, fit.home_field_advantage)
            recorded += 1
            print(f"{year} week {week:2d}: fitted {len(week_games)} games, "
                  f"HFA={fit.home_field_advantage:.2f}")
            if not from_cache:
                time.sleep(DELAY_SECONDS)

    print(f"\nRecorded {recorded} (year, week) snapshots, skipped {skipped} (missing pages or too few games).")


if __name__ == "__main__":
    main()
