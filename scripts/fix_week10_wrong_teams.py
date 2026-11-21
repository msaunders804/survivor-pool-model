"""One-off data fix: BMJ7 (entry_7) and BMJ8 (entry_8)'s Week 10 picks were saved with the wrong
team. The model's own stored recommendation for Week 10 (recommendations table, generated
2026-11-11, never since overwritten) already has the correct mapping; the picks table's Week 10
rows for these two entries don't match it.

Wrong (current) -> correct:
  entry_7 (BMJ7): wk10 BUF -> SEA
  entry_8 (BMJ8): wk10 LAR -> BUF
  entry_1 (BMJ1): wk10 LAR (already correct, untouched)

Confirmed via ESPN's Week 10, 2026 scoreboard: BUF, SEA and LAR all won, so `survived` stays
True on every row touched here -- only `team` changes.

Unlike the Week 7 fix, this never touches entry_id_fk/week on any row, so the
one_pick_per_entry_per_week unique constraint is never in play -- these are plain single-column
updates to existing rows.

Idempotent: asserts the known-wrong state before changing anything; aborts with no writes if the
database doesn't match (including because this has already been run).

Reads DATABASE_URL via api/models.py; never hardcode, print, or log it.

Run (dry run -- no commit, always safe):
    python scripts/fix_week10_wrong_teams.py --league-id 1

Run (apply -- commits only if the after-state matches exactly):
    python scripts/fix_week10_wrong_teams.py --league-id 1 --apply
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "api"))

from models import Entry, Pick, SessionLocal  # noqa: E402

WEEK = 10
# entry_id -> (wrong team expected now, correct team to write)
CORRECTIONS = {
    "entry_7": ("BUF", "SEA"),
    "entry_8": ("LAR", "BUF"),
}


def _pick(db, entry_row_id: int, week: int) -> Pick | None:
    return db.query(Pick).filter_by(entry_id_fk=entry_row_id, week=week).one_or_none()


def _print_state(label: str, db, entries: dict[str, Entry]) -> None:
    print(f"\n{label}:")
    for entry_id in sorted(entries):
        p = _pick(db, entries[entry_id].id, WEEK)
        team = p.team if p else "(no pick)"
        survived = "Won" if (p and p.survived) else "Lost" if (p and p.survived is False) else "Pending"
        print(f"  {entry_id}: wk{WEEK} {team} {survived}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--league-id", type=int, required=True)
    parser.add_argument("--apply", action="store_true", help="commit the fix (default: dry run -- always rolled back)")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        entries = {
            entry_id: db.query(Entry).filter_by(league_id=args.league_id, entry_id=entry_id).one()
            for entry_id in CORRECTIONS
        }

        _print_state("Current state", db, entries)

        picks = {entry_id: _pick(db, entry.id, WEEK) for entry_id, entry in entries.items()}
        mismatched = [
            entry_id for entry_id, (wrong_team, _) in CORRECTIONS.items()
            if picks[entry_id] is None or picks[entry_id].team != wrong_team
        ]
        if mismatched:
            raise SystemExit(
                f"\nAborting: Week {WEEK} rows for {mismatched} don't match the expected wrong state "
                "-- nothing changed. Either this fix has already been applied, or the data has moved "
                "since this script was written."
            )

        for entry_id, (_, correct_team) in CORRECTIONS.items():
            picks[entry_id].team = correct_team

        db.flush()
        _print_state("State after fix (not yet committed)", db, entries)

        still_wrong = [
            entry_id for entry_id, (_, correct_team) in CORRECTIONS.items()
            if _pick(db, entries[entry_id].id, WEEK).team != correct_team
        ]
        if still_wrong:
            db.rollback()
            raise SystemExit(f"\nAborting: {still_wrong} didn't end up correct -- rolled back, nothing changed.")

        if args.apply:
            db.commit()
            print("\nCommitted.")
        else:
            db.rollback()
            print("\nDry run only -- rolled back, nothing was written. Rerun with --apply to commit.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
