"""What the scheduled job runs each week. Two existing scripts back to
back -- run_weekly.py (hers, untouched) then ingest.py (reads its output
into the database). Nothing new here; this just chains the two commands
Megan runs by hand today, so a cron job can run them unattended.

ingest.py reads the expected payout / SE / objective from the
latest_summary.json that run_weekly.py writes, not from its stdout.

Run: python scripts/run_weekly_and_ingest.py --week 5 --league-id 1 \
       --n-rivals <Brent's latest count> --pot <latest pot> --record
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LEAGUE_START_WEEK = 4  # same constant as run_weekly.py: no availability table applies before a league's second week


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--week", type=int, default=None,
                         help="override the saved current week; default: read from the league config")
    parser.add_argument("--league-id", type=int, required=True)
    parser.add_argument("--n-rivals", type=int, default=None,
                         help="override Brent's last saved count; default: read from the league config he updates")
    parser.add_argument("--pot", type=float, default=None, help="override the saved pot; default: read from config")
    parser.add_argument("--n-paths", type=int, default=20000)
    parser.add_argument("--record", action="store_true", help="also commit picks to data_store/my_entries/")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "api"))
    from ingest import export_availability, export_picks
    from models import League, SessionLocal
    from nfl import advance_league_week
    from auto_results import fill_pending_results
    from seed import seed_if_empty

    seed_if_empty()
    store_root = REPO_ROOT / "data_store"

    # Before reading any history: fill in Won/Lost from final scores for picks still pending,
    # so an entry that lost last week isn't planned for.
    db = SessionLocal()
    try:
        league = db.get(League, args.league_id)
        if league is not None:
            if args.week is None and advance_league_week(db, league):
                print(f"Week advanced to {league.current_week} from the NFL calendar.")
            n_filled = fill_pending_results(db, league)
            if n_filled:
                print(f"Filled in {n_filled} pick result(s) from final scores.")
    finally:
        db.close()

    if args.week is None or args.n_rivals is None or args.pot is None:
        db = SessionLocal()
        try:
            league = db.get(League, args.league_id)
            if league is None:
                raise SystemExit(f"league {args.league_id} not found -- has Brent opened update.html yet?")
            args.week = args.week or league.current_week
            args.n_rivals = args.n_rivals or league.n_rivals
            args.pot = args.pot or league.pot
            print(f"Using league config: week={args.week}, n_rivals={args.n_rivals}, pot={args.pot} "
                  f"(last saved via update.html -- pass --week/--n-rivals/--pot to override)")
        finally:
            db.close()

    db = SessionLocal()
    try:
        # the job's disk is empty each run: restore earlier weeks' picks from the database
        n = export_picks(db, args.league_id, store_root / "my_entries" / "picks.csv", before_week=args.week)
        print(f"Restored {n} saved picks (weeks before {args.week}) from the database.")
    finally:
        db.close()

    availability_path = store_root / "availability.txt"
    availability_path.unlink(missing_ok=True)
    db = SessionLocal()
    try:
        n_alive = export_availability(db, args.league_id, args.week, availability_path)
    finally:
        db.close()
    if n_alive is None and args.week > LEAGUE_START_WEEK:
        print(f"No availability report pasted on the Ownership page for week {args.week} -- rivals assumed "
              f"fresh at --n-rivals {args.n_rivals}.")

    run_weekly_cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "run_weekly.py"),
        "--week", str(args.week), "--n-rivals", str(args.n_rivals),
        "--pot", str(args.pot), "--n-paths", str(args.n_paths),
    ]
    if n_alive is not None and args.week > LEAGUE_START_WEEK:
        run_weekly_cmd += ["--rival-availability", str(availability_path)]
        print(f"Using the pasted availability report: {n_alive} live entries.")
    if args.record:
        run_weekly_cmd.append("--record")

    print(f"$ {' '.join(run_weekly_cmd)}")
    result = subprocess.run(run_weekly_cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"run_weekly.py failed (exit {result.returncode}) -- nothing ingested.")

    ingest_cmd = [
        sys.executable, str(REPO_ROOT / "api" / "ingest.py"),
        "--league-id", str(args.league_id), "--week", str(args.week),
        "--store-root", str(store_root),
    ]
    print(f"\n$ {' '.join(ingest_cmd)}")
    result2 = subprocess.run(ingest_cmd, cwd=REPO_ROOT / "api", capture_output=True, text=True)
    print(result2.stdout)
    if result2.returncode != 0:
        print(result2.stderr, file=sys.stderr)
        raise SystemExit(f"ingest.py failed (exit {result2.returncode}).")

    print(f"\nDone. Week {args.week} recommendation is live in the database.")


if __name__ == "__main__":
    main()
