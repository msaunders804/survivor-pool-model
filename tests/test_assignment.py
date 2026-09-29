import pytest

import numpy as np

from survivor.simulation.assignment import assign_max_survival_picks, assign_max_survival_picks_batch


def test_simple_two_week_assignment_avoids_reusing_a_team():
    survival_probability = {
        1: {"A": 0.9, "B": 0.6},
        2: {"A": 0.8, "B": 0.7},
    }
    assignment = assign_max_survival_picks([1, 2], survival_probability)
    assert set(assignment.values()) == {"A", "B"}  # each team used once


def test_finds_global_optimum_not_greedy():
    # Greedy would take A in week 1 (0.9 > B's 0.6), forcing B (0.5) in week 2:
    # product = 0.9 * 0.5 = 0.45. The better assignment is B week1, A week2:
    # 0.6 * 0.95 = 0.57.
    survival_probability = {
        1: {"A": 0.9, "B": 0.6},
        2: {"A": 0.95, "B": 0.5},
    }
    assignment = assign_max_survival_picks([1, 2], survival_probability)
    assert assignment == {1: "B", 2: "A"}


def test_bye_team_absent_from_week_is_never_assigned_that_week():
    survival_probability = {
        1: {"A": 0.9, "C": 0.5},  # B is on bye week 1
        2: {"B": 0.9, "C": 0.4},
    }
    assignment = assign_max_survival_picks([1, 2], survival_probability)
    assert assignment[1] != "B"


def test_three_weeks_three_teams_uses_each_team_exactly_once():
    survival_probability = {
        1: {"A": 0.9, "B": 0.7, "C": 0.5},
        2: {"A": 0.85, "B": 0.75, "C": 0.55},
        3: {"A": 0.8, "B": 0.6, "C": 0.65},
    }
    assignment = assign_max_survival_picks([1, 2, 3], survival_probability)
    assert set(assignment.values()) == {"A", "B", "C"}
    assert set(assignment.keys()) == {1, 2, 3}


def test_more_weeks_than_teams_raises():
    survival_probability = {1: {"A": 0.9}, 2: {"A": 0.9}, 3: {"A": 0.9}}
    with pytest.raises(ValueError):
        assign_max_survival_picks([1, 2, 3], survival_probability)


def test_week_with_no_available_teams_raises():
    survival_probability = {1: {"A": 0.9, "B": 0.8}, 2: {}}
    with pytest.raises(ValueError):
        assign_max_survival_picks([1, 2], survival_probability)


def test_empty_weeks_returns_empty_assignment():
    assert assign_max_survival_picks([], {}) == {}


def test_batch_matches_the_single_scenario_assignment():
    rng = np.random.default_rng(0)
    teams = ["A", "B", "C", "D"]
    probs = rng.uniform(0.05, 0.95, size=(20, 3, 4))
    available = np.array([[True, True, True, True], [True, False, True, True], [True, True, True, False]])

    batch = assign_max_survival_picks_batch(probs, available)

    for i in range(len(probs)):
        lookup = {w: {teams[t]: probs[i, w, t] for t in range(4) if available[w, t]} for w in range(3)}
        single = assign_max_survival_picks([0, 1, 2], lookup)
        assert [teams[t] for t in batch[i]] == [single[w] for w in range(3)]


def test_batch_more_weeks_than_teams_raises():
    with pytest.raises(ValueError, match="distinct teams"):
        assign_max_survival_picks_batch(np.full((2, 3, 2), 0.5), np.ones((3, 2), dtype=bool))


def test_batch_week_with_no_available_teams_raises():
    available = np.array([[True, True], [False, False]])
    with pytest.raises(ValueError, match="no available teams"):
        assign_max_survival_picks_batch(np.full((2, 2, 2), 0.5), available)
