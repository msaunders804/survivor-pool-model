"""Fill in Won/Lost on saved picks from ESPN final scores, so nobody has to.

Only touches picks still pending (survived is NULL) whose game is final, so
anything marked by hand on the Results page stays as it was. A tie follows
the league's tie_counts_as_loss rule. If ESPN can't be reached, nothing
changes and the picks stay pending for the next try.
"""
from __future__ import annotations

from models import Entry, Pick
from nfl import week_results


def fill_pending_results(db, league, results_for=week_results) -> int:
    """Resolve pending picks up to the league's current week. Returns how many were filled."""
    pending = (
        db.query(Pick)
        .join(Entry, Pick.entry_id_fk == Entry.id)
        .filter(Entry.league_id == league.id, Pick.survived.is_(None), Pick.week <= league.current_week)
        .all()
    )
    results_by_week: dict[int, dict | None] = {}
    filled = 0
    for pick in pending:
        if pick.week not in results_by_week:
            try:
                results_by_week[pick.week] = results_for(pick.week, league.year)
            except Exception:
                results_by_week[pick.week] = None
        outcome = (results_by_week[pick.week] or {}).get(pick.team)
        if outcome is None:
            continue  # not final yet, or ESPN unreachable
        pick.survived = outcome == "won" or (outcome == "tie" and not league.tie_counts_as_loss)
        filled += 1
    if filled:
        db.commit()
    return filled
