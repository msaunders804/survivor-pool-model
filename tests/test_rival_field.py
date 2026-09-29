import numpy as np
import pandas as pd
import pytest

import survivor.simulation.field_simulator as field_simulator
from survivor.data.availability import parse_availability
from survivor.simulation.rival_field import (
    N_TEAMS,
    TEAM_INDEX,
    RivalFieldState,
    fit_conditional_poisson,
    sample_fixed_size_subsets,
    sample_used_sets,
    survival_probability_by_week,
    thin_survivors,
)
from tests.test_availability import SPLASH_TABLE


def _state_from_table():
    n_alive, available = parse_availability(SPLASH_TABLE)
    return RivalFieldState.from_availability(n_alive, picks_made=2, available=available)


# --- RivalFieldState ---


def test_fresh_state_has_no_used_teams():
    state = RivalFieldState.fresh(500)

    assert state.n_alive == 500 and state.picks_made == 0 and state.used_counts is None


def test_state_rejects_counts_that_dont_match_picks_made():
    counts = np.zeros(N_TEAMS, dtype=int)
    counts[0] = 10  # 10 used picks, but 2 picks each for 10 rivals would be 20

    with pytest.raises(ValueError, match="sum to"):
        RivalFieldState(n_alive=10, picks_made=2, used_counts=counts)


def test_state_rejects_a_count_above_the_live_entries():
    counts = np.zeros(N_TEAMS, dtype=int)
    counts[0], counts[1] = 11, 9

    with pytest.raises(ValueError, match="between 0 and n_alive"):
        RivalFieldState(n_alive=10, picks_made=2, used_counts=counts)


def test_state_requires_counts_once_picks_have_been_made():
    with pytest.raises(ValueError, match="required"):
        RivalFieldState(n_alive=10, picks_made=1)


def test_from_availability_rejects_unknown_team():
    with pytest.raises(KeyError, match="XXX"):
        RivalFieldState.from_availability(10, 1, {"XXX": 5})


# --- conditional Poisson ---


def test_fit_reproduces_target_inclusion_probabilities():
    target = np.array([0.7, 0.5, 0.3, 0.25, 0.15, 0.1])
    k = 2

    odds = fit_conditional_poisson(target, k)
    samples = sample_fixed_size_subsets(odds, k, 200_000, np.random.default_rng(0))

    np.testing.assert_allclose(samples.mean(axis=0), target, atol=0.005)


def test_samples_always_have_exactly_k_teams():
    odds = fit_conditional_poisson(np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.5]), 4)

    samples = sample_fixed_size_subsets(odds, 4, 5000, np.random.default_rng(1))

    assert (samples.sum(axis=1) == 4).all()


def test_fit_rejects_probabilities_not_summing_to_k():
    with pytest.raises(ValueError, match="expected"):
        fit_conditional_poisson(np.array([0.5, 0.5, 0.5]), 2)


def test_sample_used_sets_for_a_fresh_league_is_all_unused_and_draws_nothing():
    rng = np.random.default_rng(3)
    before = rng.bit_generator.state

    used = sample_used_sets(RivalFieldState.fresh(100), 50, rng)

    assert not used.any()
    assert rng.bit_generator.state == before  # a fresh league must not disturb the random stream


def test_sample_used_sets_recovers_real_usage_rates():
    state = _state_from_table()

    used = sample_used_sets(state, 60_000, np.random.default_rng(4))

    assert (used.sum(axis=1) == 2).all()
    rates = used.mean(axis=0)
    np.testing.assert_allclose(rates, state.used_counts / state.n_alive, atol=0.006)
    assert rates[TEAM_INDEX["SF"]] == pytest.approx(1340 / 2463, abs=0.006)
    assert not used[:, TEAM_INDEX["TB"]].any()  # no live entry has used the untouched teams


def test_sample_used_sets_always_includes_a_team_every_rival_has_used():
    counts = np.zeros(N_TEAMS, dtype=int)
    counts[TEAM_INDEX["SF"]] = 10  # everyone
    counts[TEAM_INDEX["KC"]] = 4
    counts[TEAM_INDEX["BUF"]] = 6
    state = RivalFieldState(n_alive=10, picks_made=2, used_counts=counts)

    used = sample_used_sets(state, 1000, np.random.default_rng(5))

    assert used[:, TEAM_INDEX["SF"]].all()
    assert (used.sum(axis=1) == 2).all()
    assert used[:, TEAM_INDEX["KC"]].mean() == pytest.approx(0.4, abs=0.05)


# --- survivor estimator ---


def _tiny_world(n_paths=1, weeks=2, n_teams_playing=4):
    """A few playing teams with fixed win probabilities and outcomes, the rest on bye."""
    playing = np.zeros((weeks, N_TEAMS), dtype=bool)
    playing[:, :n_teams_playing] = True
    win_prob = np.zeros((weeks, n_paths, N_TEAMS))
    win_prob[:, :, :n_teams_playing] = np.array([0.8, 0.6, 0.4, 0.2])[:n_teams_playing]
    wins = np.zeros((weeks, n_paths, N_TEAMS), dtype=bool)
    wins[:, :, :2] = True  # the two stronger teams win every week
    return win_prob, wins, playing


def test_week_one_survival_is_exact_for_a_fresh_field():
    win_prob, wins, playing = _tiny_world()
    beta = 3.0

    q = survival_probability_by_week(win_prob, wins, playing, RivalFieldState.fresh(100), beta, 8, np.random.default_rng(0))

    weights = np.exp(beta * win_prob[0, 0, :4])
    assert q[0, 0] == pytest.approx(weights[:2].sum() / weights.sum())


def test_second_week_survival_matches_exact_enumeration():
    win_prob, wins, playing = _tiny_world()
    beta = 3.0
    w = np.exp(beta * win_prob[0, 0, :4])  # same probabilities both weeks

    # a survivor picked winner `first` (team 0 or 1) in week 1, then must pick the *other* winner in week 2
    exact = sum(w[first] / w.sum() * w[1 - first] / np.delete(w, first).sum() for first in (0, 1))

    # ghosts split between the two week-1 winners, so this is a sample average -- use many
    q = survival_probability_by_week(
        win_prob, wins, playing, RivalFieldState.fresh(100), beta, 20_000, np.random.default_rng(0)
    )

    assert q[1, 0] == pytest.approx(exact, rel=0.01)


def test_survival_probability_never_increases_and_is_zero_once_no_team_is_left():
    win_prob, wins, playing = _tiny_world(weeks=3)  # only 2 winning teams for 3 weeks: no 3rd distinct winner exists

    q = survival_probability_by_week(win_prob, wins, playing, RivalFieldState.fresh(100), 3.0, 8, np.random.default_rng(0))

    assert (np.diff(q, axis=0) <= 1e-12).all()
    assert q[2, 0] == 0.0


def test_burned_favorite_lowers_survival_when_it_wins():
    win_prob, wins, playing = _tiny_world(weeks=1)
    counts = np.zeros(N_TEAMS, dtype=int)
    counts[0] = 100  # every rival has already used the strongest team (the week's winner)
    counts_state = RivalFieldState(n_alive=100, picks_made=1, used_counts=counts)

    q_burned = survival_probability_by_week(win_prob, wins, playing, counts_state, 3.0, 8, np.random.default_rng(0))
    q_fresh = survival_probability_by_week(win_prob, wins, playing, RivalFieldState.fresh(100), 3.0, 8, np.random.default_rng(0))

    assert q_burned[0, 0] < q_fresh[0, 0]


def test_thin_survivors_is_monotone_and_starts_from_n_alive():
    q = np.tile(np.array([[0.8], [0.5], [0.2], [0.0]]), (1, 2000))

    alive = thin_survivors(q, 500, np.random.default_rng(0))

    assert (alive[0] <= 500).all()
    assert (np.diff(alive, axis=0) <= 0).all()
    assert (alive[3] == 0).all()
    assert alive[0].mean() == pytest.approx(500 * 0.8, rel=0.01)
    assert alive[1].mean() == pytest.approx(500 * 0.5, rel=0.01)


# --- through simulate_rival_field ---

_SCHEDULE = pd.DataFrame(
    [
        {"week": week, "home_team": home, "away_team": away}
        for week in (4, 5, 6)
        for home, away in (("BUF", "NYJ"), ("KC", "LV"), ("DET", "GB"), ("PHI", "DAL"))
    ]
)
_RATINGS = {t: r for t, r in zip(["BUF", "NYJ", "KC", "LV", "DET", "GB", "PHI", "DAL"], np.linspace(3, -3, 8))}


def _ghost_sim(**overrides):
    kwargs = dict(
        schedule=_SCHEDULE, base_ratings=_RATINGS, home_field_advantage=1.0, current_week=4, final_week=6,
        n_paths=400, n_rivals=200, pot=1000.0, rng=np.random.default_rng(0), rival_model="ghost",
    )
    kwargs.update(overrides)
    return field_simulator.simulate_rival_field(**kwargs)


def test_ghost_model_produces_a_consistent_field_simulation():
    sim = _ghost_sim()

    assert sim.alive_count_by_week.shape == (3, 400)
    assert (sim.alive_count_by_week[0] <= 200).all()
    assert (np.diff(sim.alive_count_by_week, axis=0) <= 0).all()
    np.testing.assert_array_equal(sim.rival_survivors, sim.alive_count_by_week[-1])
    emptied = sim.field_emptied_week != -1
    assert (sim.rival_survivors[emptied] == 0).all()
    assert (sim.emptying_cohort_size[emptied] > 0).all() and (sim.emptying_cohort_size[~emptied] == 0).all()


def test_ghost_and_individual_agree_on_average_survival():
    ghost = _ghost_sim(n_paths=1500)
    individual = _ghost_sim(n_paths=1500, rival_model="individual", rng=np.random.default_rng(1))

    for w in range(3):
        g, i = ghost.alive_count_by_week[w], individual.alive_count_by_week[w]
        se = np.sqrt(g.var() / 1500 + i.var() / 1500)
        assert abs(g.mean() - i.mean()) < 4 * se


def test_field_emptying_helper_finds_first_zero_week_and_cohort():
    alive = np.array([[5, 3, 0], [2, 0, 0], [0, 0, 0]])  # 3 paths, weeks 4-6

    weeks, cohort = field_simulator._field_emptying(alive, 5, [4, 5, 6])

    np.testing.assert_array_equal(weeks, [6, 5, 4])
    np.testing.assert_array_equal(cohort, [2, 3, 5])


def test_field_emptying_helper_reports_paths_that_never_empty():
    weeks, cohort = field_simulator._field_emptying(np.array([[4], [3]]), 5, [4, 5])

    assert weeks[0] == -1 and cohort[0] == 0


def test_rival_state_must_match_n_rivals():
    with pytest.raises(ValueError, match="must equal n_rivals"):
        _ghost_sim(rival_state=RivalFieldState.fresh(300))


def test_unknown_rival_model_is_rejected():
    with pytest.raises(ValueError, match="rival_model"):
        _ghost_sim(rival_model="nope")


def test_both_models_can_start_from_a_pool_state():
    counts = np.zeros(N_TEAMS, dtype=int)
    counts[TEAM_INDEX["BUF"]] = 120
    counts[TEAM_INDEX["KC"]] = 80  # 200 live rivals, one pick each
    state = RivalFieldState(n_alive=200, picks_made=1, used_counts=counts)

    for model in ("ghost", "individual"):
        sim = _ghost_sim(rival_state=state, rival_model=model, n_paths=200)
        assert (sim.alive_count_by_week[0] <= 200).all()
