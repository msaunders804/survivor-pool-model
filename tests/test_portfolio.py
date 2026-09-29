import math
from itertools import product

import numpy as np
import pandas as pd
import pytest

from survivor.decision.portfolio import (
    MAX_ENTRIES,
    best_allocations,
    enumerate_allocations,
    greedy_local_allocation,
    greedy_local_entry_allocation,
    score_allocation,
    score_entries,
    validate_entry_count,
)
from survivor.simulation.field_simulator import (
    N_TEAMS,
    TEAM_INDEX,
    FieldSimulation,
    score_candidate,
    simulate_rival_field,
    team_elimination_week,
)


def test_validate_entry_count_accepts_the_league_cap():
    validate_entry_count(MAX_ENTRIES)  # no raise


def test_validate_entry_count_rejects_above_the_cap():
    with pytest.raises(ValueError):
        validate_entry_count(MAX_ENTRIES + 1)


def test_validate_entry_count_rejects_zero_or_negative():
    with pytest.raises(ValueError):
        validate_entry_count(0)
    with pytest.raises(ValueError):
        validate_entry_count(-1)


def test_enumerate_allocations_count_matches_stars_and_bars():
    allocations = enumerate_allocations(10, ["DET", "LAC", "JAX", "KC", "BUF"])
    assert len(allocations) == math.comb(10 + 5 - 1, 5 - 1)


def test_enumerate_allocations_every_allocation_sums_to_n_entries():
    allocations = enumerate_allocations(10, ["DET", "LAC", "JAX"])
    for allocation in allocations:
        assert sum(allocation.values()) == 10


def test_enumerate_allocations_omits_zero_count_teams():
    allocations = enumerate_allocations(2, ["DET", "LAC", "JAX"])
    all_teams_used = {team for allocation in allocations for team in allocation}
    assert all_teams_used == {"DET", "LAC", "JAX"}
    assert all(0 not in allocation.values() for allocation in allocations)


def test_enumerate_allocations_single_team_gets_everything():
    allocations = enumerate_allocations(5, ["DET"])
    assert allocations == [{"DET": 5}]


def test_enumerate_allocations_rejects_above_the_cap():
    with pytest.raises(ValueError):
        enumerate_allocations(MAX_ENTRIES + 1, ["DET", "LAC"])


def test_enumerate_allocations_rejects_empty_candidate_list():
    with pytest.raises(ValueError):
        enumerate_allocations(10, [])


def _blank_field_simulation(weeks, n_paths, n_rivals, pot, alive_count_by_week=None):
    return FieldSimulation(
        weeks=weeks,
        n_paths=n_paths,
        n_rivals=n_rivals,
        pot=pot,
        true_ratings={},
        survival_probability=np.full((len(weeks), n_paths, N_TEAMS), 0.5),
        team_wins=np.zeros((len(weeks), n_paths, N_TEAMS), dtype=bool),
        playing=np.zeros((len(weeks), N_TEAMS), dtype=bool),
        rival_survivors=np.zeros(n_paths, dtype=int),
        field_emptied_week=np.full(n_paths, -1, dtype=int),
        emptying_cohort_size=np.zeros(n_paths, dtype=int),
        alive_count_by_week=(
            alive_count_by_week if alive_count_by_week is not None else np.zeros((len(weeks), n_paths), dtype=int)
        ),
    )


def test_score_allocation_single_entry_matches_full_horizon_survival_formula():
    # one entry, survives both weeks, 1 rival also survives to the end
    weeks = [4, 5]
    sim = _blank_field_simulation(
        weeks, n_paths=1, n_rivals=3, pot=900.0, alive_count_by_week=np.array([[2], [1]])
    )
    sim.rival_survivors = np.array([1])

    elimination_weeks = {"BUF": np.array([-1])}
    payouts = score_allocation(sim, {"BUF": 1}, elimination_weeks)
    assert payouts[0] == pytest.approx(900.0 / (1 + 1))  # matches score_candidate's pot/(rival_survivors + 1)


def test_score_allocation_two_surviving_entries_split_with_each_other_too():
    # two of your own entries both survive to the end, alongside 1 rival --
    # the fix: three-way split (900/3 = 300 each, 600 total), not two
    # independent 900/(1+1) = 450 scores summing to a nonsensical 900 total
    weeks = [4, 5]
    sim = _blank_field_simulation(
        weeks, n_paths=1, n_rivals=3, pot=900.0, alive_count_by_week=np.array([[2], [1]])
    )
    sim.rival_survivors = np.array([1])

    elimination_weeks = {"BUF": np.array([-1]), "KC": np.array([-1])}
    payouts = score_allocation(sim, {"BUF": 1, "KC": 1}, elimination_weeks)
    assert payouts[0] == pytest.approx(900.0 * 2 / 3)


def test_score_allocation_true_emptying_accounts_for_your_other_entries():
    # the core multi-entry bug: rivals all die at week 4, entry A dies at
    # week 4 alongside them, entry B (a different team) is still alive and
    # doesn't die until week 5. The rival-only "field emptied at week 4"
    # signal (what score_candidate alone would use) would wrongly pay A a
    # cohort share it isn't entitled to, since the real field -- including
    # your own still-alive entry B -- didn't actually empty until week 5.
    # Only B, which died in the week the WHOLE field (rivals + your other
    # entry) emptied, should be paid, and it should take the entire pot
    # since by week 5 no rivals and no other entries remain.
    weeks = [4, 5]
    sim = _blank_field_simulation(
        weeks, n_paths=1, n_rivals=2, pot=1000.0, alive_count_by_week=np.array([[0], [0]])
    )
    sim.rival_survivors = np.array([0])

    elimination_weeks = {"A": np.array([4]), "B": np.array([5])}
    payouts = score_allocation(sim, {"A": 1, "B": 1}, elimination_weeks)
    assert payouts[0] == pytest.approx(1000.0)  # all of it goes to B; A gets none of it


def test_score_allocation_no_survivors_and_no_matching_cohort_scores_zero():
    weeks = [4]
    sim = _blank_field_simulation(weeks, n_paths=1, n_rivals=5, pot=500.0, alive_count_by_week=np.array([[3]]))
    sim.rival_survivors = np.array([3])

    elimination_weeks = {"BUF": np.array([4])}  # eliminated week 4, rivals didn't empty that week
    payouts = score_allocation(sim, {"BUF": 1}, elimination_weeks)
    assert payouts[0] == pytest.approx(0.0)


def test_score_allocation_missing_team_in_elimination_weeks_raises():
    weeks = [4]
    sim = _blank_field_simulation(weeks, n_paths=1, n_rivals=5, pot=500.0)
    with pytest.raises(ValueError):
        score_allocation(sim, {"BUF": 1}, elimination_weeks={})


TWO_WEEK_SCHEDULE = pd.DataFrame(
    [
        {"week": 4, "home_team": "BUF", "away_team": "NYJ"},
        {"week": 4, "home_team": "KC", "away_team": "LV"},
        {"week": 5, "home_team": "BUF", "away_team": "MIA"},
        {"week": 5, "home_team": "KC", "away_team": "DEN"},
    ]
)


def test_score_allocation_matches_score_candidate_for_a_single_entry():
    # regression check on a real simulated sim (not hand-built): scoring
    # one entry via the portfolio path must agree with score_candidate
    base_ratings = {team: 0.0 for team in ["BUF", "NYJ", "KC", "LV", "MIA", "DEN"]}
    rng = np.random.default_rng(42)
    sim = simulate_rival_field(
        TWO_WEEK_SCHEDULE, base_ratings, home_field_advantage=1.0, weekly_rating_std=1.0,
        current_week=4, final_week=5, n_paths=200, n_rivals=20, pot=1000.0, rng=rng,
    )
    expected = score_candidate(sim, "BUF")
    elimination_weeks = {"BUF": team_elimination_week(sim, "BUF")}
    actual = score_allocation(sim, {"BUF": 1}, elimination_weeks)
    np.testing.assert_allclose(actual, expected)


def test_best_allocations_sorted_best_first_and_allocations_sum_to_n_entries():
    base_ratings = {team: 0.0 for team in ["BUF", "NYJ", "KC", "LV", "MIA", "DEN"]}
    rng = np.random.default_rng(7)
    sim = simulate_rival_field(
        TWO_WEEK_SCHEDULE, base_ratings, home_field_advantage=1.0, weekly_rating_std=1.0,
        current_week=4, final_week=5, n_paths=300, n_rivals=30, pot=1000.0, rng=rng,
    )
    results = best_allocations(sim, ["BUF", "KC"], n_entries=3)

    means = [result.mean_payout for result in results]
    assert means == sorted(means, reverse=True)
    assert all(sum(result.allocation.values()) == 3 for result in results)
    assert len(results) == math.comb(3 + 2 - 1, 2 - 1)


def test_best_allocations_rejects_above_the_cap():
    sim = _blank_field_simulation([4], n_paths=1, n_rivals=1, pot=100.0)
    with pytest.raises(ValueError):
        best_allocations(sim, ["BUF", "KC"], n_entries=MAX_ENTRIES + 1)


def _real_sim(n_paths=300, n_rivals=30, seed=7):
    base_ratings = {team: 0.0 for team in ["BUF", "NYJ", "KC", "LV", "MIA", "DEN"]}
    rng = np.random.default_rng(seed)
    return simulate_rival_field(
        TWO_WEEK_SCHEDULE, base_ratings, home_field_advantage=1.0, weekly_rating_std=1.0,
        current_week=4, final_week=5, n_paths=n_paths, n_rivals=n_rivals, pot=1000.0, rng=rng,
    )


def test_best_allocations_gap_to_best_is_zero_for_the_top_result():
    sim = _real_sim()
    results = best_allocations(sim, ["BUF", "KC"], n_entries=3)
    assert results[0].gap_to_best == pytest.approx(0.0)
    assert results[0].gap_to_best_se == pytest.approx(0.0)


def test_best_allocations_gap_to_best_matches_mean_difference_for_others():
    sim = _real_sim()
    results = best_allocations(sim, ["BUF", "KC"], n_entries=3)
    for result in results[1:]:
        assert result.gap_to_best == pytest.approx(results[0].mean_payout - result.mean_payout, abs=1e-6)
        assert result.gap_to_best_se >= 0.0


def test_best_allocations_accepts_precomputed_elimination_weeks():
    sim = _real_sim()
    precomputed = {team: team_elimination_week(sim, team) for team in ["BUF", "KC"]}
    results = best_allocations(sim, ["BUF", "KC"], n_entries=3, elimination_weeks=precomputed)
    fresh = best_allocations(sim, ["BUF", "KC"], n_entries=3)
    assert [r.mean_payout for r in results] == [r.mean_payout for r in fresh]


def test_best_allocations_precomputed_elimination_weeks_missing_team_raises():
    sim = _real_sim()
    with pytest.raises(ValueError):
        best_allocations(sim, ["BUF", "KC"], n_entries=3, elimination_weeks={"BUF": team_elimination_week(sim, "BUF")})


def test_greedy_local_allocation_matches_exhaustive_search_on_a_small_case():
    # the real validation: on a candidate set small enough to enumerate
    # exactly, the heuristic should land on (or statistically tie) the true
    # best allocation, not just something plausible-looking.
    sim = _real_sim(n_paths=2000, seed=11)
    exhaustive = best_allocations(sim, ["BUF", "KC", "NYJ"], n_entries=5)
    greedy_result = greedy_local_allocation(sim, ["BUF", "KC", "NYJ"], n_entries=5)

    assert greedy_result.allocation == exhaustive[0].allocation


def test_greedy_local_allocation_rejects_above_the_cap():
    sim = _blank_field_simulation([4], n_paths=1, n_rivals=1, pot=100.0)
    with pytest.raises(ValueError):
        greedy_local_allocation(sim, ["BUF", "KC"], n_entries=MAX_ENTRIES + 1)


def test_greedy_local_allocation_rejects_empty_candidate_list():
    sim = _blank_field_simulation([4], n_paths=1, n_rivals=1, pot=100.0)
    with pytest.raises(ValueError):
        greedy_local_allocation(sim, [], n_entries=3)


def test_greedy_local_allocation_accepts_precomputed_elimination_weeks():
    sim = _real_sim()
    precomputed = {team: team_elimination_week(sim, team) for team in ["BUF", "KC"]}
    result = greedy_local_allocation(sim, ["BUF", "KC"], n_entries=3, elimination_weeks=precomputed)
    assert sum(result.allocation.values()) == 3


def test_greedy_local_allocation_scales_to_a_large_candidate_list():
    # the actual point of this function: candidate lists too large for
    # best_allocations to enumerate exhaustively (all 32 teams would be
    # C(24+32-1, 31) allocations for 24 entries -- intractable).
    sim = _real_sim(n_paths=50, n_rivals=20)
    result = greedy_local_allocation(sim, list(TEAM_INDEX), n_entries=10)
    assert sum(result.allocation.values()) == 10
    assert set(result.allocation) <= set(TEAM_INDEX)


def test_greedy_local_allocation_survives_a_team_emptying_mid_pass():
    # regression test: local search moved from_team's last unit away to one
    # to_team, removing from_team from the allocation dict entirely, then
    # kept trying *further* to_team candidates for that same now-absent
    # from_team within the same inner loop -- raised KeyError on
    # `trial[from_team] -= 1` once from_team was no longer a key. This
    #8-team, 2-week, 10-entry, all-zero-rating case reliably triggers it.
    base_ratings = {team: 0.0 for team in TEAM_INDEX}
    teams = list(TEAM_INDEX)[:8]
    schedule = pd.DataFrame(
        [
            {"week": 4, "home_team": teams[0], "away_team": teams[1]},
            {"week": 4, "home_team": teams[2], "away_team": teams[3]},
            {"week": 4, "home_team": teams[4], "away_team": teams[5]},
            {"week": 4, "home_team": teams[6], "away_team": teams[7]},
            {"week": 5, "home_team": teams[0], "away_team": teams[2]},
            {"week": 5, "home_team": teams[1], "away_team": teams[3]},
            {"week": 5, "home_team": teams[4], "away_team": teams[6]},
            {"week": 5, "home_team": teams[5], "away_team": teams[7]},
        ]
    )
    rng = np.random.default_rng(5)
    sim = simulate_rival_field(
        schedule, base_ratings, home_field_advantage=1.0, weekly_rating_std=1.0,
        current_week=4, final_week=5, n_paths=3000, n_rivals=100, pot=1000.0, rng=rng,
    )
    result = greedy_local_allocation(sim, teams, n_entries=10)  # must not raise KeyError
    assert sum(result.allocation.values()) == 10


# --- score_entries / greedy_local_entry_allocation: entries with diverged histories ---


def test_score_entries_matches_score_allocation_for_interchangeable_entries():
    sim = _real_sim()
    elimination_weeks = {"BUF": team_elimination_week(sim, "BUF"), "KC": team_elimination_week(sim, "KC")}
    allocation = {"BUF": 2, "KC": 1}
    by_team = score_allocation(sim, allocation, elimination_weeks)

    by_entry = score_entries(
        sim,
        {
            "e1": elimination_weeks["BUF"],
            "e2": elimination_weeks["BUF"],
            "e3": elimination_weeks["KC"],
        },
    )
    np.testing.assert_allclose(by_entry, by_team)


def test_score_entries_rejects_empty_input():
    sim = _blank_field_simulation([4], n_paths=1, n_rivals=1, pot=100.0)
    with pytest.raises(ValueError):
        score_entries(sim, {})


def test_greedy_local_entry_allocation_matches_exhaustive_search_with_per_entry_exclusions():
    # three entries with different prior histories, all still eligible for
    # every current-week candidate team, but each entry's own FUTURE
    # opportunity cost differs by what it already excluded (MIA vs. DEN vs.
    # nothing) -- the real subtlety this function exists for: same team
    # this week, different value per entry. Brute force every one of the
    # 4**3 = 64 valid assignments directly (not via best_allocations, which
    # can't express per-entry exclusions at all) and confirm the heuristic
    # finds the true best.
    sim = _real_sim(n_paths=1000, seed=13)
    used_teams_by_entry = {"e1": {"MIA"}, "e2": {"DEN"}, "e3": set()}
    candidate_teams = ["BUF", "NYJ", "KC", "LV"]

    best_mean, best_assignment = -1.0, None
    for teams in product(candidate_teams, repeat=3):
        assignment = dict(zip(used_teams_by_entry, teams))
        arrays = {
            entry_id: team_elimination_week(sim, team, used_teams_by_entry[entry_id])
            for entry_id, team in assignment.items()
        }
        mean = float(score_entries(sim, arrays).mean())
        if mean > best_mean:
            best_mean, best_assignment = mean, assignment

    result = greedy_local_entry_allocation(sim, used_teams_by_entry, candidate_teams)
    result_arrays = {
        entry_id: team_elimination_week(sim, team, used_teams_by_entry[entry_id])
        for entry_id, team in result.items()
    }
    result_mean = float(score_entries(sim, result_arrays).mean())

    assert result_mean == pytest.approx(best_mean, rel=1e-9)


def test_greedy_local_entry_allocation_respects_each_entrys_own_exclusions():
    sim = _real_sim()
    used_teams_by_entry = {"e1": {"BUF"}, "e2": set()}
    result = greedy_local_entry_allocation(sim, used_teams_by_entry, ["BUF", "KC"])
    assert result["e1"] != "BUF"  # e1 already used BUF, must not be re-recommended it


def test_greedy_local_entry_allocation_raises_when_an_entry_has_no_eligible_team():
    sim = _real_sim()
    used_teams_by_entry = {"e1": {"BUF", "KC"}}
    with pytest.raises(ValueError):
        greedy_local_entry_allocation(sim, used_teams_by_entry, ["BUF", "KC"])


def test_greedy_local_entry_allocation_rejects_empty_inputs():
    sim = _real_sim()
    with pytest.raises(ValueError):
        greedy_local_entry_allocation(sim, {}, ["BUF"])
    with pytest.raises(ValueError):
        greedy_local_entry_allocation(sim, {"e1": set()}, [])


def test_greedy_local_entry_allocation_rejects_above_the_cap():
    sim = _blank_field_simulation([4], n_paths=1, n_rivals=1, pot=100.0)
    used_teams_by_entry = {f"e{i}": set() for i in range(MAX_ENTRIES + 1)}
    with pytest.raises(ValueError):
        greedy_local_entry_allocation(sim, used_teams_by_entry, ["BUF", "KC"])


def test_greedy_local_entry_allocation_returns_one_team_per_entry():
    sim = _real_sim(n_paths=50, n_rivals=20)
    used_teams_by_entry = {"e1": set(), "e2": {"BUF"}, "e3": {"KC", "BUF"}}
    result = greedy_local_entry_allocation(sim, used_teams_by_entry, list(TEAM_INDEX))
    assert set(result) == {"e1", "e2", "e3"}
    for entry_id, team in result.items():
        assert team not in used_teams_by_entry[entry_id]


def test_greedy_local_entry_allocation_populates_a_shared_cache():
    # passing an external cache dict should end up populated with every
    # (used-teams, team) pair this call actually evaluated, so a caller can
    # reuse it for a second, related call (e.g. a sanity check restricted
    # to a subset of the same teams) instead of recomputing from scratch,
    # and can look up any entry's array directly afterward instead of
    # calling team_elimination_week again itself.
    sim = _real_sim(n_paths=50, n_rivals=20)
    used_teams_by_entry = {"e1": set(), "e2": set()}
    shared_cache: dict = {}

    result = greedy_local_entry_allocation(sim, used_teams_by_entry, ["BUF", "KC"], elimination_weeks=shared_cache)

    assert len(shared_cache) > 0
    for entry_id, team in result.items():
        key = (frozenset(used_teams_by_entry[entry_id]), team)
        assert key in shared_cache


def test_greedy_local_entry_allocation_reuses_a_shared_cache_across_calls():
    sim = _real_sim(n_paths=50, n_rivals=20)
    used_teams_by_entry = {"e1": set(), "e2": set()}
    shared_cache: dict = {}

    greedy_local_entry_allocation(sim, used_teams_by_entry, ["BUF", "KC", "NYJ"], elimination_weeks=shared_cache)
    size_after_first_call = len(shared_cache)

    # a second call restricted to teams already covered by the first
    # shouldn't add any new cache entries
    greedy_local_entry_allocation(sim, used_teams_by_entry, ["BUF", "KC"], elimination_weeks=shared_cache)
    assert len(shared_cache) == size_after_first_call


def _reference_score_entries(sim, elimination_weeks_by_entry):
    """Straightforward per-entry scoring, the definition the additive-totals version must match."""
    n_paths = sim.n_paths
    weeks = np.array(sim.weeks)
    your_alive_after = np.zeros((len(weeks), n_paths), dtype=int)
    for elim in elimination_weeks_by_entry.values():
        for w_idx, week in enumerate(sim.weeks):
            your_alive_after[w_idx] += (elim == -1) | (elim > week)
    total_alive_after = sim.alive_count_by_week + your_alive_after
    is_zero = total_alive_after == 0
    has_emptied = is_zero.any(axis=0)
    first_zero_idx = is_zero.argmax(axis=0)
    before = np.vstack([np.full(n_paths, sim.n_rivals + len(elimination_weeks_by_entry)), total_alive_after[:-1]])
    cohort_size = before[first_zero_idx, np.arange(n_paths)]
    true_emptied_week = np.where(has_emptied, weeks[first_zero_idx], -1)
    full = sum((e == -1).astype(int) for e in elimination_weeks_by_entry.values())
    cohort = sum((e == true_emptied_week).astype(int) for e in elimination_weeks_by_entry.values())
    return np.where(
        full > 0,
        sim.pot * full / np.where(full > 0, sim.rival_survivors + full, 1),
        np.where(cohort > 0, sim.pot * cohort / np.where(cohort > 0, cohort_size, 1), 0.0),
    )


def test_score_entries_matches_a_per_entry_reference_including_emptied_fields():
    # few rivals and many entries so the whole field empties on plenty of
    # paths -- that's where the cohort-split branch is exercised
    sim = _real_sim(n_paths=400, n_rivals=3)
    rng = np.random.default_rng(0)
    arrays = {f"e{i}": rng.choice([-1, *sim.weeks], size=sim.n_paths) for i in range(6)}

    assert (sim.alive_count_by_week.min(axis=0) == 0).any()
    np.testing.assert_array_equal(score_entries(sim, arrays), _reference_score_entries(sim, arrays))
