"""Load Megan's weekly-job output into the database.

run_weekly.py already writes two files per run, unchanged by anything
here:
  data_store/pick_sheets/latest.csv   -- entry_id, team, used_teams_before
  data_store/my_entries/picks.csv     -- entry_id, week, team, survived

This script is the bridge: read those, upsert into `recommendations` and
`picks`. It's what a cron job calls right after run_weekly.py finishes --
the weekly job's own code never needs to know the database exists.

Run: python ingest.py --league-id 1 --week 4 --payout 576.95 --se 6.77 \
       --store-root /path/to/survivor-pool-model/data_store
(payout/se come from run_weekly.py's own stdout, since the CSV doesn't
carry them -- small thing to add to her script's CSV output later.)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from models import Entry, Pick, Recommendation, SessionLocal, init_db


def ingest_recommendation(db, league_id: int, week: int, pick_sheet_path: Path,
                           expected_payout: float | None, se: float | None) -> int:
    if not pick_sheet_path.exists():
        print(f"  no pick sheet at {pick_sheet_path}, skipping recommendation ingest")
        return 0
    n = 0
    with open(pick_sheet_path) as f:
        for row in csv.DictReader(f):
            existing = (
                db.query(Recommendation)
                .filter_by(league_id=league_id, week=week, entry_id=row["entry_id"])
                .one_or_none()
            )
            if existing is None:
                existing = Recommendation(league_id=league_id, week=week, entry_id=row["entry_id"])
                db.add(existing)
            existing.team = row["team"]
            existing.used_teams_before = row.get("used_teams_before", "")
            existing.expected_total_payout = expected_payout
            existing.payout_se = se
            n += 1
    return n


def ingest_picks(db, league_id: int, picks_csv_path: Path) -> int:
    if not picks_csv_path.exists():
        print(f"  no picks file at {picks_csv_path}, skipping picks ingest")
        return 0
    n = 0
    with open(picks_csv_path) as f:
        for row in csv.DictReader(f):
            entry = db.query(Entry).filter_by(league_id=league_id, entry_id=row["entry_id"]).one_or_none()
            if entry is None:
                print(f"  entry {row['entry_id']!r} not registered in league {league_id}, skipping")
                continue
            week = int(row["week"])
            existing = db.query(Pick).filter_by(entry_id_fk=entry.id, week=week).one_or_none()
            survived = {"True": True, "False": False}.get(row.get("survived", ""), None)
            if existing is None:
                existing = Pick(entry_id_fk=entry.id, week=week, team=row["team"], survived=survived)
                db.add(existing)
            else:
                existing.team = row["team"]
                existing.survived = survived
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league-id", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--store-root", type=Path, required=True, help="path to survivor-pool-model/data_store")
    parser.add_argument("--payout", type=float, default=None)
    parser.add_argument("--se", type=float, default=None)
    args = parser.parse_args()

    init_db()
    db = SessionLocal()
    try:
        n_rec = ingest_recommendation(
            db, args.league_id, args.week,
            args.store_root / "pick_sheets" / "latest.csv", args.payout, args.se,
        )
        n_pick = ingest_picks(db, args.league_id, args.store_root / "my_entries" / "picks.csv")
        db.commit()
        print(f"Ingested {n_rec} recommendation rows, {n_pick} pick rows.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
