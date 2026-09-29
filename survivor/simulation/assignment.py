"""Maximum-survival assignment: the rollout's base policy for your own future picks.

Phase 5's rollout scores a candidate pick by simulating the rest of the
season assuming your entry plays a fixed, simple policy afterward --
"the maximum-survival team assignment over remaining weeks" per the plan.
This solves that as an assignment problem: one team per week, each team
used at most once, maximizing total survival probability. Equivalent to
minimizing summed -log(survival probability) via the Hungarian algorithm.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

MIN_SURVIVAL_PROBABILITY = 1e-9  # floor before log, so an ~impossible pick isn't literally -inf


def assign_max_survival_picks(
    weeks: list[int],
    survival_probability: dict[int, dict[str, float]],
) -> dict[int, str]:
    """One team per week maximizing total survival probability (product across weeks).

    survival_probability[week][team] must cover every team available that
    week (a team on bye that week should simply be absent from that week's
    dict). Raises if there are more weeks than available teams (can't pick
    a distinct team for every week) or a week has no available teams.
    """
    all_teams = sorted(set().union(*(d.keys() for d in survival_probability.values()))) if weeks else []
    if len(all_teams) < len(weeks):
        raise ValueError(f"only {len(all_teams)} distinct teams available for {len(weeks)} weeks")

    team_index = {team: i for i, team in enumerate(all_teams)}
    cost = np.full((len(weeks), len(all_teams)), fill_value=-np.log(MIN_SURVIVAL_PROBABILITY))

    for row, week in enumerate(weeks):
        week_probs = survival_probability[week]
        if not week_probs:
            raise ValueError(f"no available teams with a survival probability for week {week}")
        for team, prob in week_probs.items():
            cost[row, team_index[team]] = -np.log(max(prob, MIN_SURVIVAL_PROBABILITY))

    row_indices, col_indices = linear_sum_assignment(cost)
    return {weeks[row]: all_teams[col] for row, col in zip(row_indices, col_indices)}


def assign_max_survival_picks_batch(survival_probability: np.ndarray, available: np.ndarray) -> np.ndarray:
    """assign_max_survival_picks for many independent scenarios at once (one Hungarian solve each).

    survival_probability: shape (n_scenarios, n_weeks, n_teams), one candidate
    team set per column; available: bool, shape (n_weeks, n_teams), False
    where a team can't be picked that week (bye), the same for every
    scenario. Returns shape (n_scenarios, n_weeks) column indices, one team
    per week, no team twice, maximizing summed log survival -- the same
    result assign_max_survival_picks gives per scenario (identical cost
    matrix, so identical tie-breaking), minus its per-cell Python overhead,
    which was ~98% of the time; the solve itself is a small fraction.
    """
    n_weeks, n_teams = available.shape
    if n_teams < n_weeks:
        raise ValueError(f"only {n_teams} distinct teams available for {n_weeks} weeks")
    empty_weeks = np.flatnonzero(~available.any(axis=1))
    if empty_weeks.size:
        raise ValueError(f"no available teams with a survival probability for week index {empty_weeks[0]}")

    cost = -np.log(np.maximum(survival_probability, MIN_SURVIVAL_PROBABILITY))
    cost = np.where(available[np.newaxis], cost, -np.log(MIN_SURVIVAL_PROBABILITY))

    picks = np.empty((len(cost), n_weeks), dtype=int)
    for i, scenario_cost in enumerate(cost):
        _, picks[i] = linear_sum_assignment(scenario_cost)  # rows come back in order, one column per week
    return picks
