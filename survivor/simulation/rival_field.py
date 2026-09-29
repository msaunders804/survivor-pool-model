"""Rival field from aggregate data: state, used-team sampling, and a ghost-rival survivor estimator.

Two problems this solves together:

1. What we actually know about rivals. A fresh league (every rival has every
   team available -- this season's Week 4) is trivial, but later weeks only
   give a pool-wide *availability count* per team ("1,123 of 2,463 live
   entries can still pick the 49ers"), not each rival's own used-team set.
   RivalFieldState holds that aggregate; sample_used_sets turns it into
   plausible individual used-team sets -- the maximum-entropy distribution
   over sets of the right size that reproduces every team's usage rate
   exactly (conditional Poisson sampling). The one thing it can't recover is
   which teams go *together*; max-entropy assumes no dependence beyond the
   marginals and the no-repeat, fixed-size constraint.

2. Cost that doesn't scale with field size. Rivals never interact -- given
   a simulated path (its game results and win probabilities), each rival's
   picks and survival are independent of every other rival's. So the
   *expected* number alive after each week is N times the survival
   probability of a typical rival, and that probability can be estimated
   from a few "ghost" rivals per path instead of simulating all N:

   Each ghost, every week, picks only among teams that WON that week
   (drawn in proportion to the same softmax weights real rivals use, over
   the ghost's still-available teams) and carries a weight multiplied each
   week by the probability mass it put on winners -- exactly its
   probability of surviving that week. The mean ghost weight through week t
   is q_t, an unbiased estimate of one rival's chance of being alive after
   week t. Survivor counts then come from binomial thinning: N_t ~
   Binomial(N_{t-1}, q_t / q_{t-1}). Because ghosts survive by construction,
   late-season paths where the real field is down to a handful (the ones
   payouts are decided on) cost the same as early ones, and 500 vs 2,500
   rivals costs the same.

   Exact at week 1 (all rivals share one used-set, so q_1 is exact); after
   that q_t is a Monte Carlo estimate over the ghost sample, and thinning
   treats rivals as exchangeable with common survival odds -- exact in
   expectation, slightly overstating the variance of survivor counts when
   rivals' used-sets differ. validate against the per-rival simulator
   (field_simulator's "individual" model) before relying on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from survivor.data.team_keys import ALL_TEAMS

N_TEAMS = len(ALL_TEAMS)
TEAM_INDEX = {team: i for i, team in enumerate(ALL_TEAMS)}
DEFAULT_N_GHOSTS = 64
_PATH_CHUNK = 1000  # bounds the (paths, ghosts, teams) working arrays


@dataclass(frozen=True)
class RivalFieldState:
    """The live rival field at the start of the simulated horizon.

    n_alive: rivals still alive.
    picks_made: picks each live rival has already made (0 = fresh league).
    used_counts: shape (N_TEAMS,), live rivals who have used each team, in
      ALL_TEAMS order; None exactly when picks_made == 0. Every live rival
      has used exactly picks_made distinct teams, so these sum to
      picks_made * n_alive.
    """

    n_alive: int
    picks_made: int = 0
    used_counts: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.n_alive < 1:
            raise ValueError(f"n_alive must be at least 1, got {self.n_alive}")
        if self.picks_made < 0:
            raise ValueError(f"picks_made must be non-negative, got {self.picks_made}")
        if self.picks_made == 0:
            if self.used_counts is not None and np.any(self.used_counts):
                raise ValueError("used_counts must be empty for a fresh league (picks_made == 0)")
            return
        if self.used_counts is None:
            raise ValueError("used_counts is required when picks_made > 0")
        counts = np.asarray(self.used_counts)
        if counts.shape != (N_TEAMS,):
            raise ValueError(f"used_counts must have shape ({N_TEAMS},), got {counts.shape}")
        if counts.min() < 0 or counts.max() > self.n_alive:
            raise ValueError("each team's used count must be between 0 and n_alive")
        expected = self.picks_made * self.n_alive
        if counts.sum() != expected:
            raise ValueError(
                f"used counts sum to {int(counts.sum())}, but {self.n_alive} live entries with "
                f"{self.picks_made} picks each should sum to {expected} -- check the data or picks_made"
            )

    @classmethod
    def fresh(cls, n_alive: int) -> "RivalFieldState":
        """Every rival alive with every team available."""
        return cls(n_alive=n_alive, picks_made=0, used_counts=None)

    @classmethod
    def from_availability(cls, n_alive: int, picks_made: int, available: dict[str, int]) -> "RivalFieldState":
        """From per-team availability counts (live entries that can still pick each team).

        A team missing from `available` is treated as available to everyone.
        """
        unknown = set(available) - set(TEAM_INDEX)
        if unknown:
            raise KeyError(f"unrecognized team abbreviations: {sorted(unknown)}")
        used = np.zeros(N_TEAMS, dtype=int)
        for team, count in available.items():
            used[TEAM_INDEX[team]] = n_alive - count
        return cls(n_alive=n_alive, picks_made=picks_made, used_counts=used)


# --- conditional Poisson sampling (fixed-size subsets with given inclusion rates) ---


def _suffix_symmetric(odds: np.ndarray, k: int) -> np.ndarray:
    """S[i, j] = e_j(odds[i:]), the j-th elementary symmetric polynomial of the tail."""
    n = len(odds)
    table = np.zeros((n + 1, k + 1))
    table[n, 0] = 1.0
    for i in range(n - 1, -1, -1):
        table[i, 0] = 1.0
        table[i, 1:] = table[i + 1, 1:] + odds[i] * table[i + 1, :-1]
    return table


def _inclusion_probabilities(odds: np.ndarray, k: int) -> np.ndarray:
    """P(team t is in the size-k subset) under conditional Poisson with these odds."""
    n = len(odds)
    suffix = _suffix_symmetric(odds, k)
    prefix = np.zeros((n + 1, k + 1))
    prefix[0, 0] = 1.0
    for i in range(n):
        prefix[i + 1, 0] = 1.0
        prefix[i + 1, 1:] = prefix[i, 1:] + odds[i] * prefix[i, :-1]
    total = suffix[0, k]
    without = np.array([np.dot(prefix[t, :k], suffix[t + 1, k - 1 :: -1][:k]) for t in range(n)])
    return odds * without / total


def fit_conditional_poisson(inclusion: np.ndarray, k: int, tolerance: float = 1e-10, max_iterations: int = 1000) -> np.ndarray:
    """Odds whose size-k conditional Poisson subsets include team t with probability inclusion[t].

    inclusion must lie strictly inside (0, 1) and sum to k. Each step
    rescales every team's odds by target / current inclusion probability
    (Chen, Dempster & Liu 1994), which converges for feasible targets --
    matching log-odds instead can oscillate.
    """
    inclusion = np.asarray(inclusion, dtype=float)
    if not np.isclose(inclusion.sum(), k):
        raise ValueError(f"inclusion probabilities sum to {inclusion.sum():.6f}, expected {k}")
    log_odds = np.log(inclusion / (1 - inclusion))
    for _ in range(max_iterations):
        current = _inclusion_probabilities(np.exp(log_odds - log_odds.mean()), k)
        error = np.abs(current - inclusion).max()
        if error < tolerance:
            break
        log_odds += np.log(inclusion) - np.log(current)
    else:
        raise RuntimeError(f"conditional Poisson fit did not converge (max error {error:.2e})")
    return np.exp(log_odds - log_odds.mean())


def sample_fixed_size_subsets(odds: np.ndarray, k: int, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """n_samples conditional-Poisson subsets of size exactly k, shape (n_samples, len(odds)) bool."""
    n = len(odds)
    suffix = _suffix_symmetric(odds, k)
    remaining = np.full(n_samples, k)
    chosen = np.zeros((n_samples, n), dtype=bool)
    for i in range(n):
        needed = remaining > 0
        # P(include i | need j more from teams i..n-1) = odds[i] * e_{j-1}(tail i+1) / e_j(tail i)
        with np.errstate(divide="ignore", invalid="ignore"):
            probability = np.where(
                needed, odds[i] * suffix[i + 1, np.maximum(remaining - 1, 0)] / suffix[i, remaining], 0.0
            )
        take = rng.random(n_samples) < probability
        chosen[:, i] = take
        remaining -= take
    return chosen


def sample_used_sets(state: RivalFieldState, n_sets: int, rng: np.random.Generator) -> np.ndarray:
    """Plausible used-team sets for live rivals, shape (n_sets, N_TEAMS) bool.

    Fresh league: all False. Otherwise: max-entropy sets of size
    state.picks_made whose per-team usage rate matches state.used_counts.
    Teams no live rival has used are never included; teams every live rival
    has used are always included.
    """
    used = np.zeros((n_sets, N_TEAMS), dtype=bool)
    if state.picks_made == 0:
        return used

    counts = np.asarray(state.used_counts)
    always = counts == state.n_alive
    never = counts == 0
    free = ~(always | never)
    used[:, always] = True

    k_free = state.picks_made - int(always.sum())
    if k_free == 0 or not free.any():
        return used
    rates = counts[free] / state.n_alive
    odds = fit_conditional_poisson(rates, k_free)
    used[:, free] = sample_fixed_size_subsets(odds, k_free, n_sets, rng)
    return used


# --- ghost-rival survivor estimator ---


def survival_probability_by_week(
    win_probability: np.ndarray,
    team_wins: np.ndarray,
    playing: np.ndarray,
    state: RivalFieldState,
    beta: float,
    n_ghosts: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """q[w, path]: chance one typical rival is still alive after week w on that path. Shape (n_weeks, n_paths).

    win_probability: (n_weeks, n_paths, N_TEAMS), the raw win probability
    rivals' softmax uses (not the tie-adjusted survival probability);
    team_wins: bool, same shape, teams that won outright; playing:
    (n_weeks, N_TEAMS). See the module docstring for the estimator.
    """
    n_weeks, n_paths, _ = win_probability.shape
    q = np.zeros((n_weeks, n_paths))

    for start in range(0, n_paths, _PATH_CHUNK):
        chunk = slice(start, min(start + _PATH_CHUNK, n_paths))
        n_chunk = chunk.stop - chunk.start
        used = sample_used_sets(state, n_chunk * n_ghosts, rng).reshape(n_chunk, n_ghosts, N_TEAMS)
        weight = np.ones((n_chunk, n_ghosts))
        path_rows = np.arange(n_chunk)[:, np.newaxis]
        ghost_cols = np.arange(n_ghosts)[np.newaxis, :]

        for w in range(n_weeks):
            probs = win_probability[w, chunk]
            offset = np.where(playing[w], probs, -np.inf).max(axis=1, keepdims=True)  # softmax-invariant shift
            base = np.where(playing[w], np.exp(beta * (probs - offset)), 0.0)  # (paths, teams)

            available_weight = base[:, np.newaxis, :] * ~used  # (paths, ghosts, teams)
            total = available_weight.sum(axis=2)
            winner_weight = available_weight * team_wins[w, chunk][:, np.newaxis, :]
            winner_total = winner_weight.sum(axis=2)

            # probability this ghost survives the week; 0 if it has no team left to pick
            weight *= np.divide(winner_total, total, out=np.zeros_like(total), where=total > 0)
            q[w, chunk] = weight.mean(axis=1)

            # advance the ghost by picking a winner in proportion to its softmax weight
            cumulative = np.cumsum(winner_weight, axis=2)
            draw = rng.random((n_chunk, n_ghosts, 1)) * winner_total[:, :, np.newaxis]
            picked = (cumulative > draw).argmax(axis=2)  # a dead ghost (no winner available) lands on 0; its weight is already 0
            used[path_rows, ghost_cols, picked] = True
    return q


def thin_survivors(q: np.ndarray, n_alive: int, rng: np.random.Generator) -> np.ndarray:
    """Alive-rival counts per week from survival probabilities q (n_weeks, n_paths): binomial thinning."""
    n_weeks, n_paths = q.shape
    alive = np.zeros((n_weeks, n_paths), dtype=int)
    previous_q = np.ones(n_paths)
    previous_alive = np.full(n_paths, n_alive)
    for w in range(n_weeks):
        ratio = np.divide(q[w], previous_q, out=np.zeros(n_paths), where=previous_q > 0)
        alive[w] = rng.binomial(previous_alive, np.clip(ratio, 0.0, 1.0))
        previous_q, previous_alive = q[w], alive[w]
    return alive
