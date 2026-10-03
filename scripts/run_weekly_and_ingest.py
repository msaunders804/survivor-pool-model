"""What the scheduled job runs each week. Two existing scripts back to
back -- run_weekly.py (hers, untouched) then ingest.py (reads its output
into the database). Nothing new here; this just chains the two commands
Megan runs by hand today, so a cron job can run them unattended.

Reads expected payout / SE from run_weekly.py's own stdout rather than
needing a third flag -- it already prints a line shaped exactly like:
  Expected total payout: 576.95 (SE 6.77)

Run: python scripts/run_weekly_and_ingest.py --week 5 --league-id 1 \
       --n-rivals <Brent's latest count> --pot <latest pot> --record
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

PAYOUT_LINE = re.compile(r"Expected total payout:\s*([\d.]+)\s*\(SE\s*([\d.]+)\)")
REPO_ROOT = Path(__file__).resolve().parent.parent


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

    if args.week is None or args.n_rivals is None or args.pot is None:
        sys.path.insert(0, str(REPO_ROOT / "api"))
        from models import League, SessionLocal  # local import: only needed on this path
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

    run_weekly_cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "run_weekly.py"),
        "--week", str(args.week), "--n-rivals", str(args.n_rivals),
        "--pot", str(args.pot), "--n-paths", str(args.n_paths),
    ]
    if args.record:
        run_weekly_cmd.append("--record")

    print(f"$ {' '.join(run_weekly_cmd)}")
    result = subprocess.run(run_weekly_cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"run_weekly.py failed (exit {result.returncode}) -- nothing ingested.")

    match = PAYOUT_LINE.search(result.stdout)
    if match is None:
        raise SystemExit("run_weekly.py succeeded but its payout line wasn't found in stdout -- "
                          "check it still prints 'Expected total payout: X (SE Y)' unchanged.")
    payout, se = match.group(1), match.group(2)

    ingest_cmd = [
        sys.executable, str(REPO_ROOT / "api" / "ingest.py"),
        "--league-id", str(args.league_id), "--week", str(args.week),
        "--store-root", str(REPO_ROOT / "data_store"),
        "--payout", payout, "--se", se,
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
