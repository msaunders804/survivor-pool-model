"""One-off data fix: BMJ1 (entry_1) and BMJ2 (entry_2)'s Week 7 picks were saved to the wrong
entry, and the error carried into Week 8 (BMJ2's Week 8 CIN pick actually belongs to BMJ1, who
should never have been eliminated).

Wrong (current):
  entry_1 (BMJ1): wk7 HOU Lost                       -- should be wk7 DAL Won, wk8 CIN Won
  entry_2 (BMJ2): wk7 DAL Won, wk8 CIN Won           -- should be wk7 HOU Lost, no wk8 pick

Fix: swap the team/result/confirmed values between the two existing Week 7 Pick rows (never
touching entry_id_fk or week on either row, so the one_pick_per_entry_per_week unique constraint
is never at risk), then reassign the Week 8 CIN row's entry_id_fk from entry_2 to entry_1 (safe:
entry_1 has no Week 8 row yet, so no collision).

Idempotent: asserts the known-wrong state before changing anything. If the database doesn't
match exactly -- including because this has already been run -- it aborts without writing
anything, rather than guessing.

Reads the connection string from DATABASE_URL via api/models.py; never hardcode, print, or log it.

Run (dry run -- no commit, always safe):
    python scripts/fix_week7_entry_swap.py --league-id 1

Run (apply -- commits only if the after-state matches exactly):
    python scripts/fix_week7_entry_swap.py --league-id 1 --apply
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "api"))

from models import Entry, Pick, SessionLocal  # noqa: E402

ENTRY_BMJ1 = "entry_1"
ENTRY_BMJ2 = "entry_2"
WEEK_SWAP = 7
WEEK_REASSIGN = 8
REASSIGN_TEAM = "CIN"

# The known-wrong state this script is allowed to fix. If the database's Week 7/8 rows for
# these two entries don't look exactly like this, abort rather than change anything.
EXPECTED_BEFORE = {
    ENTRY_BMJ1: {WEEK_SWAP: ("HOU", False)},
    ENTRY_BMJ2: {WEEK_SWAP: ("DAL", True), WEEK_REASSIGN: (REASSIGN_TEAM, True)},
}
EXPECTED_AFTER = {
    ENTRY_BMJ1: {WEEK_SWAP: ("DAL", True), WEEK_REASSIGN: (REASSIGN_TEAM, True)},
    ENTRY_BMJ2: {WEEK_SWAP: ("HOU", False)},
}


def _pick(db, entry_row_id: int, week: int) -> Pick | None:
    return db.query(Pick).filter_by(entry_id_fk=entry_row_id, week=week).one_or_none()


def _snapshot(db, entries: dict[str, Entry]) -> dict[str, dict[int, tuple[str, bool | None]]]:
    # Queries Pick directly by entry_id_fk rather than the entries' cached `.picks` relationship,
    # which otherwise still shows the pre-reassignment Week 8 owner after the flush below.
    return {
        entry_id: {
            p.week: (p.team, p.survived)
            for p in db.query(Pick).filter_by(entry_id_fk=entry.id).order_by(Pick.week).all()
        }
        for entry_id, entry in entries.items()
    }


def _week7_8(snapshot: dict) -> dict:
    return {
        entry_id: {w: v for w, v in picks.items() if w in (WEEK_SWAP, WEEK_REASSIGN)}
        for entry_id, picks in snapshot.items()
    }


def _outcome(survived: bool | None) -> str:
    return "Won" if survived else "Lost" if survived is False else "Pending"


def _print_state(label: str, snapshot: dict) -> None:
    print(f"\n{label}:")
    for entry_id in sorted(snapshot):
        picks = snapshot[entry_id]
        parts = ", ".join(f"wk{w} {team} {_outcome(s)}" for w, (team, s) in sorted(picks.items()))
        out = any(s is False for _, s in picks.values())
        print(f"  {entry_id}: {parts}  [{'Out' if out else 'Alive'}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--league-id", type=int, required=True)
    parser.add_argument("--apply", action="store_true", help="commit the fix (default: dry run -- always rolled back)")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        entries = {
            entry_id: db.query(Entry).filter_by(league_id=args.league_id, entry_id=entry_id).one()
            for entry_id in (ENTRY_BMJ1, ENTRY_BMJ2)
        }

        before = _snapshot(db, entries)
        _print_state("Current state", before)

        actual_before = _week7_8(before)
        if actual_before != EXPECTED_BEFORE:
            raise SystemExit(
                "\nAborting: Week 7/8 rows don't match the expected wrong state -- nothing changed.\n"
                f"  expected: {EXPECTED_BEFORE}\n  found:    {actual_before}\n"
                "Either this fix has already been applied, or the data has moved since this script "
                "was written. Re-check by hand before editing EXPECTED_BEFORE/EXPECTED_AFTER."
            )

        bmj1, bmj2 = entries[ENTRY_BMJ1], entries[ENTRY_BMJ2]
        wk7_bmj1, wk7_bmj2 = _pick(db, bmj1.id, WEEK_SWAP), _pick(db, bmj2.id, WEEK_SWAP)
        wk8_bmj2 = _pick(db, bmj2.id, WEEK_REASSIGN)

        wk7_bmj1.team, wk7_bmj2.team = wk7_bmj2.team, wk7_bmj1.team
        wk7_bmj1.survived, wk7_bmj2.survived = wk7_bmj2.survived, wk7_bmj1.survived
        wk7_bmj1.confirmed, wk7_bmj2.confirmed = wk7_bmj2.confirmed, wk7_bmj1.confirmed

        wk8_bmj2.entry_id_fk = bmj1.id  # reassign the Week 8 CIN row from BMJ2 to BMJ1

        db.flush()
        after = _snapshot(db, entries)
        _print_state("State after fix (not yet committed)", after)

        actual_after = _week7_8(after)
        if actual_after != EXPECTED_AFTER:
            db.rollback()
            raise SystemExit(
                "\nAborting: result doesn't match the target state -- rolled back, nothing changed.\n"
                f"  expected: {EXPECTED_AFTER}\n  found:    {actual_after}"
            )

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
