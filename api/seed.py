"""Seed the database with Trainò's actual current state.

Not synthetic data -- this is what's in the Sep 27 to Oct 3 email thread:
Brent's 10 entries (BMJ1-BMJ10), all alive through Week 3, and Megan's
real Week 4 recommendation (576.95 expected payout, SE 6.77, 842 entries,
$16,840 pot). Run once against an empty database.
"""
from __future__ import annotations

from models import Entry, League, LeagueMembership, Pick, Recommendation, SessionLocal, User, init_db

WEEK4_RECOMMENDATION = {
    "entry_1": "MIN", "entry_2": "BAL", "entry_3": "SEA", "entry_4": "IND",
    "entry_5": "KC", "entry_6": "BUF", "entry_7": "MIN", "entry_8": "BAL",
    "entry_9": "GB", "entry_10": "DET",
}
EXPECTED_PAYOUT = 576.95
PAYOUT_SE = 6.77
ENTRY_LABELS = {f"entry_{i+1}": f"BMJ{i+1}" for i in range(10)}


def main() -> None:
    init_db()
    db = SessionLocal()
    try:
        brent = User(email="brent@example.com", password_hash="placeholder")
        jaclyn = User(email="jaclyn@example.com", password_hash="placeholder")
        megan = User(email="megan@example.com", password_hash="placeholder")
        db.add_all([brent, jaclyn, megan])
        db.flush()

        league = League(
            name="Trainò", created_by=brent.id, entry_cap=10, buy_in=20.0,
            rake_pct=0.10, pot=16_840.0, week_start=4, week_end=18, n_rivals=842,
        )
        db.add(league)
        db.flush()

        db.add_all([
            LeagueMembership(league_id=league.id, user_id=brent.id, is_creator=True),
            LeagueMembership(league_id=league.id, user_id=jaclyn.id, is_creator=False),
            LeagueMembership(league_id=league.id, user_id=megan.id, is_creator=False),
        ])

        for entry_id in WEEK4_RECOMMENDATION:
            db.add(Entry(league_id=league.id, owner_id=brent.id, entry_id=entry_id))
        db.flush()

        for entry_id, team in WEEK4_RECOMMENDATION.items():
            db.add(
                Recommendation(
                    league_id=league.id, week=4, entry_id=entry_id, team=team,
                    used_teams_before="", expected_total_payout=EXPECTED_PAYOUT, payout_se=PAYOUT_SE,
                )
            )

        db.commit()
        print(f"Seeded league {league.id} ('{league.name}') with {len(WEEK4_RECOMMENDATION)} entries "
              f"and the real Week 4 recommendation ({EXPECTED_PAYOUT}, SE {PAYOUT_SE}).")
    finally:
        db.close()


if __name__ == "__main__":
    main()
