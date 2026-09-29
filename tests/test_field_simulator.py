import numpy as np
import pandas as pd
import pytest

import survivor.simulation.field_simulator as field_simulator
from survivor.simulation.assignment import assign_max_survival_picks
from survivor.simulation.field_simulator import (
    ALL_TEAMS,
    N_TEAMS,
    TEAM_INDEX,
    FieldSimulation,
    score_candidate,
    simulate_rival_field,
    team_elimination_week,
)


def _blank_field_simulation(weeks, n_paths, n_rivals, pot):
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
        alive_count_by_week=np.zeros((len(weeks), n_paths), dtype=int),
    )


def test_score_candidate_three_payout_branches():
    # path0: survives everything, rival field never empties -> shares the pot
    # path1: eliminated the same week the rival field empties -> shares that cohort's pot
    # path2: survives week4 but loses its assigned future pick -> gets nothing
    weeks = [4, 5]
    sim = _blank_field_simulation(weeks, n_paths=3, n_rivals=10, pot=900.0)
    sim.playing[0, TEAM_INDEX["BUF"]] = True
    sim.playing[1, TEAM_INDEX["KC"]] = True  # the only team available for the week-5 assignment

    # week 4 (BUF): path0 wins, path1 loses, path2 wins
    sim.team_wins[0, 0, TEAM_INDEX["BUF"]] = True
    sim.team_wins[0, 1, TEAM_INDEX["BUF"]] = False
    sim.team_wins[0, 2, TEAM_INDEX["BUF"]] = True

    # week 5 (KC, forced by being the only playing, non-excluded team): path0 wins, path2 loses
    sim.team_wins[1, 0, TEAM_INDEX["KC"]] = True
    sim.team_wins[1, 2, TEAM_INDEX["KC"]] = False
    sim.survival_probability[1, :, TEAM_INDEX["KC"]] = 0.6

    sim.rival_survivors = np.array([1, 0, 1])
    sim.field_emptied_week = np.array([-1, 4, -1])
    sim.emptying_cohort_size = np.array([0, 5, 0])

    payouts = score_candidate(sim, "BUF")

    assert payouts[0] == pytest.approx(900.0 / (1 + 1))  # survives to the end, splits with 1 rival
    assert payouts[1] == pytest.approx(900.0 / (5 + 1))  # eliminated with the emptying cohort
    assert payouts[2] == pytest.approx(0.0)  # eliminated later, field didn't empty that week


def test_score_candidate_outlasts_emptied_field_wins_full_pot():
    weeks = [4]
    sim = _blank_field_simulation(weeks, n_paths=1, n_rivals=5, pot=1000.0)
    sim.playing[0, TEAM_INDEX["BUF"]] = True
    sim.team_wins[0, 0, TEAM_INDEX["BUF"]] = True
    sim.field_emptied_week = np.array([4])  # rivals emptied out at week 4, your entry survived it
    sim.emptying_cohort_size = np.array([5])

    payouts = score_candidate(sim, "BUF")
    assert payouts[0] == pytest.approx(1000.0)


def test_score_candidate_single_week_no_future_assignment_needed():
    weeks = [4]
    sim = _blank_field_simulation(weeks, n_paths=2, n_rivals=3, pot=300.0)
    sim.playing[0, TEAM_INDEX["BUF"]] = True
    sim.team_wins[0, 0, TEAM_INDEX["BUF"]] = True
    sim.team_wins[0, 1, TEAM_INDEX["BUF"]] = False
    sim.rival_survivors = np.array([2, 0])
    sim.field_emptied_week = np.array([-1, -1])

    payouts = score_candidate(sim, "BUF")
    assert payouts[0] == pytest.approx(300.0 / 3)
    assert payouts[1] == pytest.approx(0.0)


TWO_WEEK_SCHEDULE = pd.DataFrame(
    [
        {"week": 4, "home_team": "BUF", "away_team": "NYJ"},
        {"week": 4, "home_team": "KC", "away_team": "LV"},
        {"week": 5, "home_team": "BUF", "away_team": "MIA"},
        {"week": 5, "home_team": "KC", "away_team": "DEN"},
    ]
)


def test_simulate_rival_field_shapes_and_monotonic_elimination():
    base_ratings = {team: 0.0 for team in ["BUF", "NYJ", "KC", "LV", "MIA", "DEN"]}
    rng = np.random.default_rng(0)
    sim = simulate_rival_field(
        TWO_WEEK_SCHEDULE, base_ratings, home_field_advantage=1.0, weekly_rating_std=1.0,
        current_week=4, final_week=5, n_paths=50, n_rivals=20, pot=1000.0, rng=rng,
    )
    assert sim.alive_count_by_week.shape == (2, 50)
    assert sim.rival_survivors.shape == (50,)
    # elimination is monotonic: alive count can't go up week over week
    assert (sim.alive_count_by_week[1] <= sim.alive_count_by_week[0]).all()
    assert (sim.alive_count_by_week <= 20).all()
    assert (sim.alive_count_by_week >= 0).all()
    assert (sim.rival_survivors == sim.alive_count_by_week[-1]).all()


def test_simulate_rival_field_emptying_bookkeeping_is_consistent():
    base_ratings = {team: 0.0 for team in ["BUF", "NYJ", "KC", "LV", "MIA", "DEN"]}
    rng = np.random.default_rng(1)
    sim = simulate_rival_field(
        TWO_WEEK_SCHEDULE, base_ratings, home_field_advantage=1.0, weekly_rating_std=1.0,
        current_week=4, final_week=5, n_paths=200, n_rivals=8, pot=1000.0, rng=rng,
    )
    emptied = sim.field_emptied_week != -1
    # wherever the field emptied, rival_survivors must be exactly 0
    assert (sim.rival_survivors[emptied] == 0).all()
    # wherever it never emptied, rival_survivors must be > 0
    assert (sim.rival_survivors[~emptied] > 0).all()
    assert (sim.emptying_cohort_size[emptied] > 0).all()


def test_expected_payout_can_favor_the_underdog_over_the_favorite_via_leverage():
    # BUF (86.6% win prob) and MIA (13.4% win prob) in an otherwise
    # ratings-neutral field. This is not a bug: it's the plan's central
    # thesis in miniature ("expected payout... differs [from survival
    # probability] because payout depends on how many rivals survive with
    # you"). Almost every rival concentrates on the two ~87% favorites
    # (BUF and NE), so when BUF wins you split the pot with dozens of
    # co-survivors; almost nobody picks MIA, so on the rare path where it
    # wins you split with almost no one. The rare, low-competition win can
    # beat the frequent, crowded one in expectation.
    base_ratings = {"BUF": 15.0, "NYJ": 0.0, "MIA": -15.0, "NE": 0.0}
    schedule = pd.DataFrame(
        [
            {"week": 4, "home_team": "BUF", "away_team": "NYJ"},
            {"week": 4, "home_team": "MIA", "away_team": "NE"},
        ]
    )

    rng = np.random.default_rng(2)
    sim = simulate_rival_field(
        schedule, base_ratings,
        home_field_advantage=0.0, weekly_rating_std=1.0,
        current_week=4, final_week=4, n_paths=2000, n_rivals=100, pot=1000.0, rng=rng,
    )
    fav_payouts = score_candidate(sim, "BUF")
    dog_payouts = score_candidate(sim, "MIA")

    assert (fav_payouts > 0).mean() > (dog_payouts > 0).mean()  # BUF wins (survives) far more often...
    assert dog_payouts.mean() > fav_payouts.mean()  # ...yet MIA's rarer win pays out more on average


def test_near_impossible_win_has_near_zero_expected_payout_regardless_of_leverage():
    # However extreme the leverage, a team that essentially never wins
    # still can't have a meaningfully positive expected payout.
    base_ratings = {"BUF": 60.0, "NYJ": 0.0, "MIA": -60.0, "NE": 0.0}
    schedule = pd.DataFrame(
        [
            {"week": 4, "home_team": "BUF", "away_team": "NYJ"},
            {"week": 4, "home_team": "MIA", "away_team": "NE"},
        ]
    )
    rng = np.random.default_rng(3)
    sim = simulate_rival_field(
        schedule, base_ratings,
        home_field_advantage=0.0, weekly_rating_std=1.0,
        current_week=4, final_week=4, n_paths=2000, n_rivals=100, pot=1000.0, rng=rng,
    )
    dog_payouts = score_candidate(sim, "MIA")
    assert dog_payouts.mean() < 1.0  # pot is 1000; this should be near enough to 0


def _reference_team_elimination_week(sim, team, used_teams_before=None):
    """The original one-path-at-a-time definition team_elimination_week must match exactly."""
    excluded = (used_teams_before or set()) | {team}
    future_weeks = sim.weeks[1:]
    eliminated_week = np.full(sim.n_paths, -1, dtype=int)
    for path in range(sim.n_paths):
        picks = {sim.weeks[0]: team}
        if future_weeks:
            survival_lookup = {}
            for w_idx, week in enumerate(future_weeks, start=1):
                survival_lookup[week] = {
                    other: sim.survival_probability[w_idx, path, TEAM_INDEX[other]]
                    for other in ALL_TEAMS
                    if other not in excluded and sim.playing[w_idx, TEAM_INDEX[other]]
                }
            picks.update(assign_max_survival_picks(future_weeks, survival_lookup))
        for w_idx, week in enumerate(sim.weeks):
            if not sim.team_wins[w_idx, path, TEAM_INDEX[picks[week]]]:
                eliminated_week[path] = week
                break
    return eliminated_week


def _six_week_sim_with_byes(n_paths=40):
    teams = ALL_TEAMS[:10]
    rows = []
    for week in range(4, 10):
        playing = [t for i, t in enumerate(teams) if (i + week) % 5 != 0]  # two teams on bye each week
        rows += [{"week": week, "home_team": playing[i], "away_team": playing[i + 1]} for i in range(0, len(playing), 2)]
    ratings = {t: r for t, r in zip(teams, np.linspace(-3, 3, len(teams)))}
    return simulate_rival_field(
        pd.DataFrame(rows), ratings, home_field_advantage=1.0, weekly_rating_std=1.5,
        current_week=4, final_week=9, n_paths=n_paths, n_rivals=5, pot=100.0, rng=np.random.default_rng(3),
    )


@pytest.mark.parametrize("team,used", [(ALL_TEAMS[1], None), (ALL_TEAMS[2], {ALL_TEAMS[3]}), (ALL_TEAMS[6], {ALL_TEAMS[0], ALL_TEAMS[7]})])
def test_team_elimination_week_matches_the_per_path_reference(team, used, monkeypatch):
    sim = _six_week_sim_with_byes()
    # tiny chunks so paths straddle several chunk boundaries
    monkeypatch.setattr(field_simulator, "_ELIMINATION_CHUNK_PATHS", 7)

    np.testing.assert_array_equal(team_elimination_week(sim, team, used), _reference_team_elimination_week(sim, team, used))


def test_team_elimination_week_single_week_needs_no_assignment():
    sim = _six_week_sim_with_byes()
    one_week = FieldSimulation(**{**sim.__dict__, "weeks": sim.weeks[:1], "survival_probability": sim.survival_probability[:1],
                                  "team_wins": sim.team_wins[:1], "playing": sim.playing[:1],
                                  "alive_count_by_week": sim.alive_count_by_week[:1]})
    team = ALL_TEAMS[1]

    np.testing.assert_array_equal(team_elimination_week(one_week, team), _reference_team_elimination_week(one_week, team))
