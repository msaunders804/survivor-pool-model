"""Joint allocation of your entries across candidate teams (Phase 6).

Ten entries all on one team is one bet at ten times the stake, so entries
must be optimized jointly rather than picking the best team for each entry
independently -- see the plan's Portfolio layer note. Three search methods,
the first two for entries that are still interchangeable (a fresh season
start -- no prior picks distinguishing one entry from another):

- best_allocations: exhaustive. Enumerates every way to split n_entries
  across a short list of candidate teams (stars and bars: choosing counts
  that sum to n_entries is equivalent to placing len(candidate_teams) - 1
  dividers among n_entries items). Exact, but the allocation count is
  C(n_entries + k - 1, k - 1) for k candidate teams -- fine for the plan's
  "top 5 or so" (1,001 for 10 entries), intractable for anything like all
  ~32 teams playing a week (tens of millions).
- greedy_local_allocation: heuristic. Builds one allocation greedily (each
  entry goes wherever it adds the most expected payout given the entries
  already placed) then refines it with pairwise local search (try moving
  one entry between two teams, keep it if it helps, repeat to a local
  optimum). Not guaranteed optimal, but its per-step cost doesn't explode
  with the candidate list size, so it's the one to use for a large
  candidate set. Validated against best_allocations on small cases where
  exhaustive search is still checkable (test_portfolio.py).

Both accept a precomputed `elimination_weeks` dict (from
field_simulator.team_elimination_week, one array per candidate team) so a
weekly batch job can compute it once for every team playing that week --
this part IS cheap to do broadly, it's O(candidate teams) Hungarian-
assignment solves, not combinatorial -- and every interactive re-plan
during the week reuses it instead of recomputing.

Once entries have diverged (Week 5 onward: different prior picks, some
possibly already eliminated -- see survivor.data.my_entries), they stop
being interchangeable, for a subtle reason: even two entries put on the
*same* team this week aren't equivalent anymore, because each one's own
future max-survival assignment excludes whatever it used in *prior* weeks,
which can differ per entry. team_elimination_week already supports this
(it takes a used_teams_before per call), it just needs calling once per
(entry, team) pair instead of once per team.

- score_entries / greedy_local_entry_allocation: the entry-aware
  equivalents. score_entries is the shared core both score_allocation and
  the entry-aware search build on -- it works from each entry's own
  elimination-week array and doesn't care about team identity, only
  whether entries happen to share an array. greedy_local_entry_allocation
  is the same greedy-then-local-search pattern, but each entry can only be
  assigned a team outside its own used-teams history, and results are
  cached by (used-teams signature, team) so entries who haven't diverged
  yet don't pay for duplicate Hungarian-assignment solves. Validated
  against exhaustive enumeration on a small case with per-entry exclusions
  (test_portfolio.py).

MAX_ENTRIES: this season's actual Splash Sports league caps a single
participant at 25 entries, and that cap is enforced here as a hard input
validation rather than a soft warning, since silently truncating or scoring
past it would produce a recommendation you can't legally submit. This
number is specific to the current league, not a property of the model --
see the "League configuration (future)" note in plan.md's Phase 6 section
for how a multi-league version should handle this instead of hardcoding it.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from survivor.simulation.field_simulator import FieldSimulation, team_elimination_week

MAX_ENTRIES = 25  # this league's cap (Splash Sports, 2026 season) -- see module docstring


@dataclass
class AllocationResult:
    allocation: dict[str, int]
    mean_payout: float
    standard_error: float
    # gap_to_best / gap_to_best_se are 0.0 for the best allocation itself,
    # and for any result not produced by a ranked comparison (e.g.
    # greedy_local_allocation, which searches for one allocation rather
    # than ranking several). Computed from the same per-path payouts as the
    # best allocation (common random numbers), so this is a genuine paired
    # comparison -- its standard error is usually much smaller than
    # sqrt(this.standard_error**2 + best.standard_error**2) would suggest,
    # since both allocations share the same simulated field. A caller can
    # treat an allocation as statistically tied with the best one when
    # gap_to_best is within a couple of gap_to_best_se of zero.
    gap_to_best: float = 0.0
    gap_to_best_se: float = 0.0


def validate_entry_count(n_entries: int) -> None:
    """Raises ValueError if n_entries is out of the range this league allows."""
    if n_entries < 1:
        raise ValueError(f"n_entries must be at least 1, got {n_entries}")
    if n_entries > MAX_ENTRIES:
        raise ValueError(
            f"n_entries ({n_entries}) exceeds this league's cap of {MAX_ENTRIES}. "
            "If you're running this for a different league, update MAX_ENTRIES in "
            "survivor/decision/portfolio.py -- see plan.md's Phase 6 design note."
        )


def enumerate_allocations(n_entries: int, candidate_teams: list[str]) -> list[dict[str, int]]:
    """Every way to split n_entries indistinguishable entries across candidate_teams.

    Returns one dict per allocation, mapping team -> entry count, omitting
    teams assigned zero entries. Count is C(n_entries + k - 1, k - 1) for k
    candidate teams -- 1,001 for the plan's "10 entries across the top 5"
    example, so exhaustive enumeration is cheap at this league's entry cap.
    A much larger cap or candidate list would need a non-exhaustive search
    instead of this stars-and-bars enumeration.
    """
    validate_entry_count(n_entries)
    if not candidate_teams:
        raise ValueError("candidate_teams must be non-empty")

    k = len(candidate_teams)
    allocations = []
    # stars and bars: choose k - 1 divider positions among n_entries + k - 1 slots
    for dividers in combinations(range(n_entries + k - 1), k - 1):
        bounds = (-1, *dividers, n_entries + k - 1)
        counts = [bounds[i + 1] - bounds[i] - 1 for i in range(k)]
        allocations.append({team: c for team, c in zip(candidate_teams, counts) if c > 0})
    return allocations


def score_entries(sim: FieldSimulation, elimination_weeks_by_entry: dict[str, np.ndarray]) -> np.ndarray:
    """Expected total portfolio payout per path for a set of individual entries.

    elimination_weeks_by_entry: {entry_id: array}, one array (shape
    (sim.n_paths,), -1 where that entry survives the whole horizon) per
    entry -- as returned by field_simulator.team_elimination_week, called
    once per entry with *that entry's own* used_teams_before. This is the
    core primitive score_allocation and the entry-aware allocation search
    both build on: score_allocation (entries assumed interchangeable,
    grouped by team) expands a {team: count} allocation into `count`
    synthetic same-team entries and calls this; the entry-aware search
    (survivor.decision.portfolio's per-entry functions, for entries with
    diverged histories) calls this directly, one real entry per key.

    This is the fix field_simulator.score_candidate can't express: that
    function scores one entry as if it were your only one, splitting only
    against the rival field ("+1" for your entry). With several entries,
    two things it ignores start to matter: (1) several of your entries can
    survive to the final week together, splitting the pot between each
    other as well as with surviving rivals; (2) the week the *whole* field
    empties -- the event that triggers a cohort split -- has to account for
    your other still-alive entries, not just rivals. A rival-only "field
    emptied" week can be wrong once you have entries of your own still
    alive past it: an entry eliminated that week didn't actually die
    alongside the last survivors, because you had another entry keeping
    the field non-empty, so it isn't part of any split and should score
    zero, not a rival-cohort share it was never entitled to.
    """
    if not elimination_weeks_by_entry:
        raise ValueError("elimination_weeks_by_entry must be non-empty")

    totals = _Totals.zero(sim)
    for elim in elimination_weeks_by_entry.values():
        totals = totals + _Totals.of(sim, elim)
    return _payouts(sim, totals)


@dataclass
class _Totals:
    """Additive per-path aggregates of a set of entries -- all score_entries needs.

    Payout depends on the entries only through three sums, so a portfolio's
    score is a sum of per-entry contributions, and a search move (add,
    remove or swap one entry) is an add/subtract instead of a rescore over
    every entry:
      alive_after (n_weeks, n_paths): entries still alive after each week
      cohort (n_weeks, n_paths): entries eliminated in each week (the
        cohort that splits if that week empties the whole field)
      full (n_paths,): entries that survive the whole horizon
    """

    alive_after: np.ndarray
    cohort: np.ndarray
    full: np.ndarray
    n_entries: int

    @classmethod
    def zero(cls, sim: FieldSimulation) -> "_Totals":
        shape = (len(sim.weeks), sim.n_paths)
        return cls(np.zeros(shape, np.int16), np.zeros(shape, np.int16), np.zeros(sim.n_paths, np.int16), 0)

    @classmethod
    def of(cls, sim: FieldSimulation, elim: np.ndarray) -> "_Totals":
        """One entry's contribution, from its elimination-week array (-1 = survives the horizon)."""
        weeks = np.asarray(sim.weeks)[:, None]
        never = elim == -1
        return cls(
            (never | (elim > weeks)).astype(np.int16),
            (elim == weeks).astype(np.int16),
            never.astype(np.int16),
            1,
        )

    def scaled(self, k: int) -> "_Totals":
        return _Totals(self.alive_after * k, self.cohort * k, self.full * k, self.n_entries * k)

    def __add__(self, other: "_Totals") -> "_Totals":
        return _Totals(
            self.alive_after + other.alive_after,
            self.cohort + other.cohort,
            self.full + other.full,
            self.n_entries + other.n_entries,
        )

    def __sub__(self, other: "_Totals") -> "_Totals":
        return _Totals(
            self.alive_after - other.alive_after,
            self.cohort - other.cohort,
            self.full - other.full,
            self.n_entries - other.n_entries,
        )


def _payouts(sim: FieldSimulation, totals: _Totals) -> np.ndarray:
    """Per-path payout for aggregated entries -- the scoring rule score_entries documents."""
    n_paths = sim.n_paths
    paths = np.arange(n_paths)

    # rivals + yours, jointly -- this is the field the plan's payout rule
    # actually means, not rivals alone
    total_alive_after = sim.alive_count_by_week + totals.alive_after

    is_zero = total_alive_after == 0
    has_emptied = is_zero.any(axis=0)
    first_zero_idx = is_zero.argmax(axis=0)  # 0 where has_emptied is False; unused there

    initial_total = sim.n_rivals + totals.n_entries
    before = np.vstack([np.full(n_paths, initial_total), total_alive_after[:-1]])
    cohort_size = before[first_zero_idx, paths]

    my_full_survivors = totals.full.astype(int)
    # only read where no entry of yours survives the horizon, where the
    # cohort is exactly your entries eliminated in the week the field emptied
    my_cohort = np.where(has_emptied, totals.cohort[first_zero_idx, paths], 0).astype(int)

    # guard both denominators against 0/0 on paths where the branch that
    # uses them isn't the one np.where ends up selecting (np.where still
    # evaluates both operands eagerly)
    survivor_denom = np.where(my_full_survivors > 0, sim.rival_survivors + my_full_survivors, 1)
    cohort_denom = np.where(my_cohort > 0, cohort_size, 1)

    return np.where(
        my_full_survivors > 0,
        sim.pot * my_full_survivors / survivor_denom,
        np.where(my_cohort > 0, sim.pot * my_cohort / cohort_denom, 0.0),
    )


def score_allocation(
    sim: FieldSimulation,
    allocation: dict[str, int],
    elimination_weeks: dict[str, np.ndarray],
) -> np.ndarray:
    """Expected payout per path for a joint allocation of interchangeable entries across teams.

    elimination_weeks must have one array (shape (sim.n_paths,), -1 where
    that team's entry survives the whole horizon) per team in `allocation`
    -- as returned by field_simulator.team_elimination_week. Compute each
    candidate team's array once and reuse it across every allocation that
    uses that team; it doesn't depend on how many entries you put there,
    only on the team and (for now) an empty used-teams history, so the same
    array is valid for every allocation this call.

    A thin wrapper over score_entries: expands {team: count} into `count`
    synthetic same-team entries. Only valid when entries really are
    interchangeable (no prior used-teams history distinguishing them --
    true for a fresh season start). Once entries have diverged, use
    greedy_local_entry_allocation and score_entries directly instead.
    """
    missing = set(allocation) - set(elimination_weeks)
    if missing:
        raise ValueError(f"elimination_weeks missing entries for: {sorted(missing)}")

    contributions = {team: _Totals.of(sim, elimination_weeks[team]) for team in allocation}
    return _allocation_payouts(sim, allocation, contributions)


def _allocation_payouts(
    sim: FieldSimulation, allocation: dict[str, int], contributions: dict[str, _Totals]
) -> np.ndarray:
    """score_allocation from precomputed per-team contributions (computed once, reused across allocations)."""
    totals = _Totals.zero(sim)
    for team, count in allocation.items():
        totals = totals + contributions[team].scaled(count)
    return _payouts(sim, totals)


def _standard_error(payouts: np.ndarray, n_paths: int) -> float:
    return float(payouts.std(ddof=1) / np.sqrt(n_paths)) if n_paths > 1 else 0.0


def best_allocations(
    sim: FieldSimulation,
    candidate_teams: list[str],
    n_entries: int,
    used_teams_before: dict[str, set[str]] | None = None,
    elimination_weeks: dict[str, np.ndarray] | None = None,
) -> list[AllocationResult]:
    """Every allocation of n_entries across candidate_teams, ranked by mean expected payout.

    Pass a precomputed elimination_weeks (e.g. computed once for every team
    playing this week by a weekly batch job) to skip recomputing it here --
    it must cover every team in candidate_teams. Otherwise it's computed
    fresh for exactly candidate_teams.

    gap_to_best / gap_to_best_se on each result are a paired comparison
    against the best allocation, using the same simulated paths (common
    random numbers) -- see AllocationResult's docstring for how to read it.
    """
    validate_entry_count(n_entries)
    used_teams_before = used_teams_before or {}

    if elimination_weeks is None:
        elimination_weeks = {
            team: team_elimination_week(sim, team, used_teams_before.get(team))
            for team in candidate_teams
        }
    else:
        missing = set(candidate_teams) - set(elimination_weeks)
        if missing:
            raise ValueError(f"elimination_weeks missing candidate teams: {sorted(missing)}")

    contributions = {team: _Totals.of(sim, elimination_weeks[team]) for team in candidate_teams}
    scored = [
        (allocation, _allocation_payouts(sim, allocation, contributions))
        for allocation in enumerate_allocations(n_entries, candidate_teams)
    ]
    scored.sort(key=lambda item: item[1].mean(), reverse=True)
    best_payouts = scored[0][1]

    results = []
    for i, (allocation, payouts) in enumerate(scored):
        if i == 0:
            gap_to_best, gap_to_best_se = 0.0, 0.0
        else:
            diff = best_payouts - payouts  # paired, not independent -- see AllocationResult
            gap_to_best, gap_to_best_se = float(diff.mean()), _standard_error(diff, sim.n_paths)
        results.append(
            AllocationResult(
                allocation=allocation,
                mean_payout=float(payouts.mean()),
                standard_error=_standard_error(payouts, sim.n_paths),
                gap_to_best=gap_to_best,
                gap_to_best_se=gap_to_best_se,
            )
        )
    return results


def greedy_local_allocation(
    sim: FieldSimulation,
    candidate_teams: list[str],
    n_entries: int,
    used_teams_before: dict[str, set[str]] | None = None,
    elimination_weeks: dict[str, np.ndarray] | None = None,
) -> AllocationResult:
    """Heuristic search for a single good allocation, for candidate lists too large to enumerate exhaustively.

    Greedy construction (each entry goes wherever it adds the most expected
    payout given entries already placed) followed by pairwise local search
    (move one entry between two teams, keep it if it helps, repeat to a
    local optimum). Not guaranteed globally optimal -- use best_allocations
    instead when the candidate list is small enough to enumerate exactly
    (see module docstring for the tradeoff).
    """
    validate_entry_count(n_entries)
    if not candidate_teams:
        raise ValueError("candidate_teams must be non-empty")
    used_teams_before = used_teams_before or {}

    if elimination_weeks is None:
        elimination_weeks = {
            team: team_elimination_week(sim, team, used_teams_before.get(team))
            for team in candidate_teams
        }
    else:
        missing = set(candidate_teams) - set(elimination_weeks)
        if missing:
            raise ValueError(f"elimination_weeks missing candidate teams: {sorted(missing)}")

    contributions = {team: _Totals.of(sim, elimination_weeks[team]) for team in candidate_teams}

    def mean_of(totals: _Totals) -> float:
        return float(_payouts(sim, totals).mean())

    # running totals for `allocation`, updated by add/subtract per move
    # instead of rescoring every entry from scratch
    allocation: dict[str, int] = {}
    totals = _Totals.zero(sim)
    for _ in range(n_entries):
        best_team = max(candidate_teams, key=lambda team: mean_of(totals + contributions[team]))
        allocation[best_team] = allocation.get(best_team, 0) + 1
        totals = totals + contributions[best_team]

    current_mean = mean_of(totals)
    improved = True
    while improved:
        improved = False
        for from_team in list(allocation):
            for to_team in candidate_teams:
                if to_team == from_team:
                    continue
                if allocation.get(from_team, 0) == 0:
                    break  # an earlier move already this pass moved from_team's last entry away
                trial_totals = totals - contributions[from_team] + contributions[to_team]
                trial_mean = mean_of(trial_totals)
                if trial_mean > current_mean:
                    allocation[from_team] -= 1
                    if allocation[from_team] == 0:
                        del allocation[from_team]
                    allocation[to_team] = allocation.get(to_team, 0) + 1
                    totals, current_mean = trial_totals, trial_mean
                    improved = True

    payouts = _payouts(sim, totals)
    return AllocationResult(
        allocation=allocation,
        mean_payout=float(payouts.mean()),
        standard_error=_standard_error(payouts, sim.n_paths),
    )


def greedy_local_entry_allocation(
    sim: FieldSimulation,
    used_teams_by_entry: dict[str, set[str]],
    candidate_teams: list[str],
    elimination_weeks: dict[tuple[frozenset, str], np.ndarray] | None = None,
) -> dict[str, str]:
    """Heuristic per-entry team assignment for entries with diverged histories.

    Unlike greedy_local_allocation (which assumes entries are
    interchangeable), each entry here has its own used_teams_before history
    excluding it from some candidate_teams -- pass
    survivor.data.my_entries.used_teams_by_entry(alive_entries) once your
    entries have picked different teams in prior weeks and some may
    already be eliminated.

    elimination_weeks, if given, is used as the (used-teams signature, team)
    cache instead of a fresh one, and is mutated in place with whatever
    this call computes -- pass the same dict into multiple calls (e.g. a
    full-candidate-list recommendation and a smaller sanity-check list
    restricted to a subset of the same teams and entries) to skip
    recomputing Hungarian-assignment solves already done for the first, and
    inspect it afterward instead of recomputing per-entry arrays again for
    scoring. Keys are (frozenset(that entry's used_teams), team).

    Same greedy-then-local-search pattern as greedy_local_allocation, on
    individual entries instead of team counts: greedy construction places
    the most-constrained entries (fewest eligible teams) first, each going
    wherever its own eligible choices add the most expected payout given
    entries already placed; local search then tries moving each entry to a
    different one of its own eligible teams, keeping the move if it
    improves the total. team_elimination_week results are cached by
    (used-teams signature, team), so entries who happen to share an
    identical history so far -- common right after they first diverge --
    don't pay for duplicate Hungarian-assignment solves.

    Raises if any entry has no eligible team left among candidate_teams, or
    if more entries are passed than this league's MAX_ENTRIES allows. Not
    guaranteed globally optimal; validated against exhaustive enumeration
    on a small case with per-entry exclusions (test_portfolio.py). Returns
    entry_id -> recommended team.
    """
    validate_entry_count(len(used_teams_by_entry))
    if not used_teams_by_entry:
        raise ValueError("used_teams_by_entry must be non-empty")
    if not candidate_teams:
        raise ValueError("candidate_teams must be non-empty")

    eligible = {
        entry_id: [team for team in candidate_teams if team not in used]
        for entry_id, used in used_teams_by_entry.items()
    }
    stuck = sorted(entry_id for entry_id, teams in eligible.items() if not teams)
    if stuck:
        raise ValueError(f"no eligible team left among candidate_teams for: {stuck}")

    cache = elimination_weeks if elimination_weeks is not None else {}

    def contribution(entry_id: str, team: str) -> _Totals:
        key = (frozenset(used_teams_by_entry[entry_id]), team)
        if key not in cache:
            cache[key] = team_elimination_week(sim, team, used_teams_by_entry[entry_id])
        if key not in contributions:
            contributions[key] = _Totals.of(sim, cache[key])
        return contributions[key]

    def mean_of(totals: _Totals) -> float:
        return float(_payouts(sim, totals).mean())

    contributions: dict[tuple[frozenset, str], _Totals] = {}

    # most-constrained entries (fewest eligible teams) first, a standard
    # heuristic to avoid boxing in a tightly-constrained entry by filling
    # the flexible ones first
    order = sorted(eligible, key=lambda entry_id: len(eligible[entry_id]))
    assignment: dict[str, str] = {}
    totals = _Totals.zero(sim)  # running totals for `assignment`, moved by add/subtract
    for entry_id in order:
        best_team = max(eligible[entry_id], key=lambda team: mean_of(totals + contribution(entry_id, team)))
        assignment[entry_id] = best_team
        totals = totals + contribution(entry_id, best_team)

    entry_ids = list(used_teams_by_entry)
    current_mean = mean_of(totals)
    improved = True
    while improved:
        improved = False
        # single-entry moves
        for entry_id in entry_ids:
            for team in eligible[entry_id]:
                if team == assignment[entry_id]:
                    continue
                trial_totals = totals - contribution(entry_id, assignment[entry_id]) + contribution(entry_id, team)
                trial_mean = mean_of(trial_totals)
                if trial_mean > current_mean:
                    assignment[entry_id] = team
                    totals, current_mean = trial_totals, trial_mean
                    improved = True
        # pairwise swaps: a move that only helps when two entries change
        # together (e.g. each is better off with the other's current team)
        # can look like no improvement to either single-entry move alone
        for i, a in enumerate(entry_ids):
            for b in entry_ids[i + 1 :]:
                team_a, team_b = assignment[a], assignment[b]
                if team_a == team_b or team_b not in eligible[a] or team_a not in eligible[b]:
                    continue
                trial_totals = (
                    totals
                    - contribution(a, team_a)
                    - contribution(b, team_b)
                    + contribution(a, team_b)
                    + contribution(b, team_a)
                )
                trial_mean = mean_of(trial_totals)
                if trial_mean > current_mean:
                    assignment[a], assignment[b] = team_b, team_a
                    totals, current_mean = trial_totals, trial_mean
                    improved = True

    return assignment
