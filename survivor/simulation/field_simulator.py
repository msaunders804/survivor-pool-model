"""Monte Carlo simulation of the whole field: the core decision engine (Phase 5).

Two stages, matching the plan's layered design:

1. simulate_rival_field runs the expensive part ONCE per set of paths --
   sampling correlated team ratings, simulating game outcomes and the
   ~500-rival field's picks/eliminations week by week -- independent of any
   candidate pick you're considering. This is what "common random numbers"
   (the plan's variance-reduction note) means in practice here: your
   candidate choice can't affect ~500 rivals or real game outcomes, so
   scoring several candidates against the identical simulated world is both
   correct and what makes the standard error of the *difference* between
   candidates small.
2. score_candidate is the cheap part: given that shared simulation, walk
   each candidate's own pick (this week fixed, future weeks via the
   maximum-survival assignment base policy) through the same realized
   outcomes and compute its expected payout. Call it once per candidate
   you want to compare, against the same FieldSimulation.

Rivals start fresh (no used teams) at current_week unless a
rival_field.RivalFieldState says otherwise -- a fresh league (this season's
Week 4) is exactly that, and later weeks can pass a pool's per-team
availability counts instead of per-rival pick history.

Two rival models: "individual" simulates every rival's picks (the reference,
cost scales with the field size) and "ghost" estimates survivor counts from
a few ghost rivals per path (rival_field.py), independent of field size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from survivor.data.team_keys import ALL_TEAMS
from survivor.decision.popularity import DEFAULT_BETA, DEFAULT_GAMMA
from survivor.probability.current_week import DEFAULT_TIE_PROBABILITY
from survivor.probability.ratings import DEFAULT_WEEKLY_RATING_STD, sample_correlated_ratings
from survivor.probability.spread import rating_diff_to_spread, spread_to_win_prob
from survivor.simulation.assignment import assign_max_survival_picks_batch
from survivor.simulation.rival_field import (
    DEFAULT_N_GHOSTS,
    RivalFieldState,
    sample_used_sets,
    survival_probability_by_week,
    thin_survivors,
)

TEAM_INDEX = {team: i for i, team in enumerate(ALL_TEAMS)}
N_TEAMS = len(ALL_TEAMS)
_ELIMINATION_CHUNK_PATHS = 5000  # bounds team_elimination_week's (paths, weeks, teams) cost tensor to ~50 MB


@dataclass
class FieldSimulation:
    weeks: list[int]
    n_paths: int
    n_rivals: int
    pot: float
    true_ratings: dict[str, np.ndarray]  # team -> shape (n_paths,)
    survival_probability: np.ndarray  # shape (n_weeks, n_paths, n_teams)
    team_wins: np.ndarray  # bool, shape (n_weeks, n_paths, n_teams)
    playing: np.ndarray  # bool, shape (n_weeks, n_teams) -- False on a bye
    rival_survivors: np.ndarray  # int, shape (n_paths,) -- alive rivals at the end
    field_emptied_week: np.ndarray  # int, shape (n_paths,) -- -1 if never emptied
    emptying_cohort_size: np.ndarray  # int, shape (n_paths,) -- rivals eliminated in that week
    alive_count_by_week: np.ndarray  # int, shape (n_weeks, n_paths) -- for elimination-curve checks


def _week_survival_probability(
    true_ratings: dict[str, np.ndarray],
    home_field_advantage: float,
    games: pd.DataFrame,
    tie_probability: float,
    n_paths: int,
) -> np.ndarray:
    """survival probability per team for one week, shape (n_paths, n_teams). Non-playing teams are 0."""
    probs = np.zeros((n_paths, N_TEAMS))
    for _, game in games.iterrows():
        home, away = game["home_team"], game["away_team"]
        spread = rating_diff_to_spread(true_ratings[home], true_ratings[away], home_field_advantage)
        home_win = spread_to_win_prob(spread)
        probs[:, TEAM_INDEX[home]] = home_win * (1 - tie_probability)
        probs[:, TEAM_INDEX[away]] = (1 - home_win) * (1 - tie_probability)
    return probs


def _sample_outcomes(
    survival_prob_week: np.ndarray, games: pd.DataFrame, rng: np.random.Generator
) -> np.ndarray:
    """Which teams win outright this week, shape (n_paths, n_teams). A tie eliminates both pickers."""
    n_paths = survival_prob_week.shape[0]
    wins = np.zeros((n_paths, N_TEAMS), dtype=bool)
    for _, game in games.iterrows():
        home_idx, away_idx = TEAM_INDEX[game["home_team"]], TEAM_INDEX[game["away_team"]]
        home_survival = survival_prob_week[:, home_idx]
        away_survival = survival_prob_week[:, away_idx]
        draw = rng.random(n_paths)
        wins[:, home_idx] = draw < home_survival
        wins[:, away_idx] = (draw >= home_survival) & (draw < home_survival + away_survival)
        # remaining probability mass is a tie: neither team's picker survives
    return wins


def _simulate_rival_picks_and_eliminations(
    alive: np.ndarray,  # bool, shape (n_paths, n_rivals), mutated in place
    used: np.ndarray,  # bool, shape (n_paths, n_rivals, n_teams), mutated in place
    win_probability_week: np.ndarray,  # shape (n_paths, n_teams) -- raw win prob, not survival (popularity uses win prob)
    team_wins_week: np.ndarray,  # bool, shape (n_paths, n_teams)
    playing_week: np.ndarray,  # bool, shape (n_teams,)
    beta: float,
    gamma: float,
    rng: np.random.Generator,
) -> None:
    """Vectorized softmax pick + elimination for every (path, rival) at once."""
    n_paths, n_rivals, _ = used.shape

    available = playing_week[np.newaxis, np.newaxis, :] & ~used  # (n_paths, n_rivals, n_teams)
    scores = beta * win_probability_week[:, np.newaxis, :]  # broadcast over rivals
    scores = np.broadcast_to(scores, (n_paths, n_rivals, N_TEAMS)).copy()
    scores[~available] = -np.inf

    has_choice = available.any(axis=2) & alive  # (n_paths, n_rivals)

    # softmax over the team axis, safe for rows that are all -inf (no choice available)
    safe_scores = np.where(np.isneginf(scores), 0.0, scores)
    max_score = np.where(has_choice[:, :, np.newaxis], safe_scores.max(axis=2, keepdims=True), 0.0)
    exp_scores = np.exp(np.where(available, scores - max_score, -np.inf))
    exp_scores = np.nan_to_num(exp_scores, neginf=0.0)
    totals = exp_scores.sum(axis=2, keepdims=True)
    totals[totals == 0] = 1.0  # avoid div-by-zero for rows with no choice; picks masked out below
    probabilities = exp_scores / totals

    cumulative = np.cumsum(probabilities, axis=2)
    draw = rng.random((n_paths, n_rivals, 1))
    picked_index = np.argmax(cumulative >= draw, axis=2)  # (n_paths, n_rivals)

    path_idx, rival_idx = np.meshgrid(np.arange(n_paths), np.arange(n_rivals), indexing="ij")
    valid = has_choice
    used[path_idx[valid], rival_idx[valid], picked_index[valid]] = True

    picked_wins = team_wins_week[path_idx, picked_index]  # (n_paths, n_rivals)
    survives = picked_wins & valid
    newly_eliminated = alive & valid & ~survives
    # a rival alive with no available team to pick is eliminated too (edge case)
    newly_eliminated |= alive & ~has_choice
    alive &= ~newly_eliminated


def _field_emptying(alive_count_by_week: np.ndarray, n_alive: int, weeks: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """(field_emptied_week, emptying_cohort_size) per path from alive counts (n_weeks, n_paths).

    The emptying week is the first one whose end-of-week count is 0 with rivals alive going in (-1 / 0 if the
    field never empties); the cohort is everyone alive going into that week.
    """
    n_paths = alive_count_by_week.shape[1]
    alive_before = np.vstack([np.full(n_paths, n_alive), alive_count_by_week[:-1]])
    newly_emptied = (alive_before > 0) & (alive_count_by_week == 0)  # at most one week per path: counts never rise
    emptied_any = newly_emptied.any(axis=0)
    emptied_idx = newly_emptied.argmax(axis=0)
    field_emptied_week = np.where(emptied_any, np.asarray(weeks)[emptied_idx], -1)
    cohort_size = np.where(emptied_any, alive_before[emptied_idx, np.arange(n_paths)], 0)
    return field_emptied_week, cohort_size


def simulate_rival_field(
    schedule: pd.DataFrame,
    base_ratings: dict[str, float],
    home_field_advantage: float,
    current_week: int,
    final_week: int,
    n_paths: int,
    n_rivals: int,
    pot: float,
    weekly_rating_std: float = DEFAULT_WEEKLY_RATING_STD,
    tie_probability: float = DEFAULT_TIE_PROBABILITY,
    popularity_beta: float = DEFAULT_BETA,
    popularity_gamma: float = DEFAULT_GAMMA,
    current_week_survival_probability: dict[str, float] | None = None,
    rng: np.random.Generator | None = None,
    rival_model: str = "individual",
    rival_state: RivalFieldState | None = None,
    n_ghosts: int = DEFAULT_N_GHOSTS,
) -> FieldSimulation:
    """Simulate n_paths seasons of an n_rivals-entry field from current_week through final_week.

    current_week_survival_probability, if given, overrides the sampled-
    rating projection for current_week only with real market-derived
    probabilities (Phase 2) -- future weeks always use the rating
    projection (Phase 3), since real lines aren't available that far out.

    rival_state describes the live field at current_week (default: n_rivals
    rivals, all fresh); if given, its n_alive must equal n_rivals.
    rival_model picks how the field is simulated -- see the module
    docstring; n_ghosts only applies to "ghost".
    """
    if rival_model not in ("individual", "ghost"):
        raise ValueError(f"rival_model must be 'individual' or 'ghost', got {rival_model!r}")
    if rival_state is None:
        rival_state = RivalFieldState.fresh(n_rivals)
    elif rival_state.n_alive != n_rivals:
        raise ValueError(f"rival_state.n_alive ({rival_state.n_alive}) must equal n_rivals ({n_rivals})")
    rng = rng or np.random.default_rng()
    weeks = list(range(current_week, final_week + 1))

    true_ratings = sample_correlated_ratings(
        base_ratings, weekly_rating_std, weeks_ahead=len(weeks), n_paths=n_paths, rng=rng
    )
    for team in ALL_TEAMS:
        true_ratings.setdefault(team, np.zeros(n_paths))

    survival_probability = np.zeros((len(weeks), n_paths, N_TEAMS))
    team_wins = np.zeros((len(weeks), n_paths, N_TEAMS), dtype=bool)
    playing = np.zeros((len(weeks), N_TEAMS), dtype=bool)

    individual = rival_model == "individual"
    if individual:
        alive = np.ones((n_paths, n_rivals), dtype=bool)
        used = sample_used_sets(rival_state, n_paths * n_rivals, rng).reshape(n_paths, n_rivals, N_TEAMS)
    alive_count_by_week = np.zeros((len(weeks), n_paths), dtype=int)
    field_emptied_week = np.full(n_paths, -1, dtype=int)
    emptying_cohort_size = np.zeros(n_paths, dtype=int)

    for w_idx, week in enumerate(weeks):
        games = schedule[schedule["week"] == week]
        for _, game in games.iterrows():
            playing[w_idx, TEAM_INDEX[game["home_team"]]] = True
            playing[w_idx, TEAM_INDEX[game["away_team"]]] = True

        probs = _week_survival_probability(true_ratings, home_field_advantage, games, tie_probability, n_paths)
        if week == current_week and current_week_survival_probability:
            for team, p in current_week_survival_probability.items():
                probs[:, TEAM_INDEX[team]] = p

        survival_probability[w_idx] = probs
        wins = _sample_outcomes(probs, games, rng)
        team_wins[w_idx] = wins

        if not individual:
            continue  # the ghost model estimates the whole field after every week's outcomes are drawn

        win_probability_week = probs / (1 - tie_probability)  # undo the tie adjustment for popularity scoring
        alive_before = alive.sum(axis=1).copy()

        _simulate_rival_picks_and_eliminations(
            alive, used, win_probability_week, wins, playing[w_idx], popularity_beta, popularity_gamma, rng
        )

        alive_after = alive.sum(axis=1)
        alive_count_by_week[w_idx] = alive_after

        newly_emptied = (alive_before > 0) & (alive_after == 0) & (field_emptied_week == -1)
        field_emptied_week[newly_emptied] = week
        emptying_cohort_size[newly_emptied] = alive_before[newly_emptied].astype(int)

    if individual:
        rival_survivors = alive.sum(axis=1)
    else:
        q = survival_probability_by_week(
            survival_probability / (1 - tie_probability), team_wins, playing, rival_state,
            popularity_beta, n_ghosts, rng,
        )
        alive_count_by_week = thin_survivors(q, n_rivals, rng)
        rival_survivors = alive_count_by_week[-1]
        field_emptied_week, emptying_cohort_size = _field_emptying(alive_count_by_week, n_rivals, weeks)

    return FieldSimulation(
        weeks=weeks,
        n_paths=n_paths,
        n_rivals=n_rivals,
        pot=pot,
        true_ratings=true_ratings,
        survival_probability=survival_probability,
        team_wins=team_wins,
        playing=playing,
        rival_survivors=rival_survivors,
        field_emptied_week=field_emptied_week,
        emptying_cohort_size=emptying_cohort_size,
        alive_count_by_week=alive_count_by_week,
    )


def team_elimination_week(sim: FieldSimulation, team: str, used_teams_before: set[str] | None = None) -> np.ndarray:
    """Per-path elimination week for one candidate team's entry, -1 if it survives every week in sim.weeks.

    This week's pick is fixed to `team`; future weeks follow the
    max-survival assignment base policy (solved once per path, since
    ratings -- and so the optimal assignment -- differ by path).

    Shared by score_candidate (Phase 5, scoring one entry against the
    rival field) and survivor.decision.portfolio (Phase 6, scoring many of
    your own entries jointly): the expensive part here is the per-path
    assignment solve, which depends only on the candidate team and its
    used-teams history, not on how many of your entries end up on it. Phase
    6 calls this once per candidate team and reuses the result across every
    allocation that uses that team.
    """
    used_teams_before = used_teams_before or set()
    excluded = used_teams_before | {team}
    future_weeks = sim.weeks[1:]
    paths = np.arange(sim.n_paths)

    # this week's pick is fixed
    survived = sim.team_wins[0, :, TEAM_INDEX[team]][:, np.newaxis]
    if future_weeks:
        # teams eligible for at least one future week -- the same column set
        # (and alphabetical order) assign_max_survival_picks builds per path
        columns = sorted(
            t for t in ALL_TEAMS if t not in excluded and sim.playing[1:, TEAM_INDEX[t]].any()
        )
        column_index = np.array([TEAM_INDEX[t] for t in columns])
        available = sim.playing[1:][:, column_index]  # (n_future_weeks, n_columns)
        future_wins = np.empty((sim.n_paths, len(future_weeks)), dtype=bool)
        future_rows = np.arange(1, len(sim.weeks))[np.newaxis, :]
        # chunked so the (paths, weeks, teams) cost tensor stays a bounded size
        for start in range(0, sim.n_paths, _ELIMINATION_CHUNK_PATHS):
            chunk = slice(start, min(start + _ELIMINATION_CHUNK_PATHS, sim.n_paths))
            probs = sim.survival_probability[1:, chunk][:, :, column_index].transpose(1, 0, 2)
            picked = column_index[assign_max_survival_picks_batch(probs, available)]  # team indices
            future_wins[chunk] = sim.team_wins[future_rows, paths[chunk, np.newaxis], picked]
        survived = np.hstack([survived, future_wins])

    lost = ~survived
    first_loss = lost.argmax(axis=1)
    return np.where(lost.any(axis=1), np.asarray(sim.weeks)[first_loss], -1)


def score_candidate(sim: FieldSimulation, current_pick: str, used_teams_before: set[str] | None = None) -> np.ndarray:
    """Expected payout per path for one candidate entry: this week's pick, future weeks via max-survival.

    Returns an array of shape (n_paths,) -- the payout on each simulated
    path. Its mean is the candidate's Monte Carlo expected payout.

    Assumes this is your only entry in the field (the "+1" below). Scoring
    several of your own entries together -- where they can end up sharing a
    winner set with each other, not just with rivals -- needs
    survivor.decision.portfolio.score_allocation instead.
    """
    eliminated_week = team_elimination_week(sim, current_pick, used_teams_before)

    # survived the whole horizon -- shares the pot with however many rivals
    # also made it, or wins outright if the rival field emptied out at some
    # earlier week while your entry kept going (simplified to a full-pot
    # win rather than simulating a 1-entry "competition" for the remaining
    # weeks -- it changes nothing real).
    survived = sim.pot / (sim.rival_survivors + 1)
    outlasted_emptied_field = np.full(sim.n_paths, sim.pot)
    # eliminated the same week the rival field emptied out entirely: that
    # whole cohort splits the pot, per the plan's payout rule
    in_emptying_cohort = sim.pot / (sim.emptying_cohort_size + 1)

    return np.where(
        eliminated_week == -1,
        np.where(sim.field_emptied_week == -1, survived, outlasted_emptied_field),
        np.where(eliminated_week == sim.field_emptied_week, in_emptying_cohort, 0.0),
    )
