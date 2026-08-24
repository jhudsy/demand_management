"""Demand-management mechanisms: the pre-allocation filter.

Every mechanism answers the same question -- which proposals reach the funder --
and they differ in **who decides**. That is the axis the paper is about, so it is
the axis the code is organised on.

    self         the researcher picks among their own proposals
    institution  the organisation picks among its members' proposals
    mechanical   a rule on history decides, nobody exercises judgement
    funder       the funder screens cheaply before full proposals exist
    none         no filter

A mechanism takes the full set of *intended* proposals and returns a boolean mask
of those actually submitted, plus an accounting of effort spent on proposals that
were killed (which differs by mechanism and is the crux of the welfare argument).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from dm.evaluation import SIGMA_EVAL, observed_score


@dataclass
class Proposals:
    """The set of proposals a round could contain, before any filtering."""

    author: np.ndarray  # researcher index
    institution: np.ndarray  # institution index
    quality: np.ndarray  # latent quality of the proposal
    is_resubmission: np.ndarray  # bool

    @property
    def n(self) -> int:
        return self.quality.size


@dataclass
class TriageResult:
    submitted: np.ndarray  # bool mask over proposals
    # Effort spent on proposals that never reached the funder, as a multiple of
    # one full proposal's cost. Mechanisms differ sharply here: a cap announced
    # in advance stops the proposal being written, whereas an internal
    # competition discards work already done.
    wasted_effort: float
    # Effort spent on proposals that did reach the funder.
    submitted_effort: float
    # Review capacity consumed, in full-proposal-equivalents. Normally this is
    # just the number submitted, but a two-stage scheme also makes the funder
    # screen every outline, and that screening is not free. Left None by a rule
    # whose load is simply what it submitted; None rather than 0.0 so that a
    # genuine zero load is not silently recomputed.
    review_load: float | None = None

    def __post_init__(self) -> None:
        if self.review_load is None:
            self.review_load = float(np.count_nonzero(self.submitted))


class TriageRule(Protocol):
    name: str
    decider: str

    def apply(
        self, p: Proposals, state: "History", rng: np.random.Generator
    ) -> TriageResult: ...


@dataclass
class History:
    """Per-researcher state that mechanical rules condition on."""

    n_researchers: int
    barred_until: np.ndarray = field(default=None)
    # Headcount per institution. A size-proportional quota must scale with how
    # many researchers an institution employs, not with how many happen to
    # submit in a given round; the latter lets the quota grow with demand,
    # which is the opposite of what the instrument is for.
    inst_size: np.ndarray = field(default=None)
    proposal_resub_count: np.ndarray = field(default=None)  # per proposal, this round
    # Submissions per researcher over the last WINDOW_MAX rounds, most recent
    # first. A per-investigator cap defined over several rounds ("one proposal
    # every three years") needs this; a cap defined per round does not, and
    # restricting the instrument to the latter understates how hard it can
    # squeeze.
    sub_window: np.ndarray = field(default=None)  # (WINDOW_MAX, n)
    # Bottom-half rankings and awards over the same rolling window. A
    # cooling-off rule triggers on a record accumulated over a period, not
    # within a single round, so both have to be carried across rounds.
    strike_window: np.ndarray = field(default=None)  # (WINDOW_MAX, n)
    award_window: np.ndarray = field(default=None)  # (WINDOW_MAX, n)

    def __post_init__(self) -> None:
        if self.barred_until is None:
            self.barred_until = np.full(self.n_researchers, -1, dtype=np.int32)
        for name in ("sub_window", "strike_window", "award_window"):
            if getattr(self, name) is None:
                setattr(self, name, np.zeros(
                    (WINDOW_MAX, self.n_researchers), dtype=np.int32
                ))

    def record_round(
        self, subs: np.ndarray, awards: np.ndarray, strikes: np.ndarray
    ) -> None:
        """Advance the rolling windows by one round."""
        for name, counts in (
            ("sub_window", subs),
            ("award_window", awards),
            ("strike_window", strikes),
        ):
            w = np.roll(getattr(self, name), 1, axis=0)
            w[0] = counts
            setattr(self, name, w)

    def used_in_window(self, window: int) -> np.ndarray:
        """Submissions in the `window - 1` rounds preceding this one.

        Excludes the current round, because a per-investigator cap is applied
        before the round's submissions are recorded.
        """
        back = max(0, min(window - 1, WINDOW_MAX))
        if back == 0:
            return np.zeros(self.n_researchers, dtype=np.int32)
        return self.sub_window[:back].sum(axis=0)

    def record_over(self, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Submissions, awards and bottom-half rankings over the last `window`
        rounds, including the round just recorded. This is what a cooling-off
        rule conditions on."""
        w = max(1, min(window, WINDOW_MAX))
        return (
            self.sub_window[:w].sum(axis=0),
            self.award_window[:w].sum(axis=0),
            self.strike_window[:w].sum(axis=0),
        )


WINDOW_MAX = 8  # longest per-investigator cap window the history supports


def _full_cost(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask))


# ---------------------------------------------------------------------------
# none
# ---------------------------------------------------------------------------


@dataclass
class NoCap:
    name: str = "no_cap"
    decider: str = "none"
    entry_cost: float = 1.0

    def apply(self, p: Proposals, state: History, rng) -> TriageResult:
        keep = np.ones(p.n, dtype=bool)
        return TriageResult(keep, wasted_effort=0.0, submitted_effort=_full_cost(keep))


# ---------------------------------------------------------------------------
# self
# ---------------------------------------------------------------------------


@dataclass
class IndividualCap:
    """At most K proposals per researcher per `window` rounds, chosen by them.

    Announced in advance, so the proposals that lose the internal contest are
    never written: `wasted_effort` is zero, up to the effort-offset parameter
    handled by the caller.

    `window` matters for how hard the instrument can squeeze. At window=1 the
    tightest possible setting still lets every researcher submit once a round,
    which puts a floor under the review load somewhere near the number of active
    researchers. Real schemes are written over longer periods ("one proposal per
    investigator every three years"), and there is no such floor: the allowance
    is K minus whatever the researcher has already spent in the window.
    """

    k: int
    sigma_self: float = SIGMA_EVAL
    window: int = 1
    name: str = "individual_cap"
    decider: str = "self"
    entry_cost: float = 1.0

    def apply(self, p: Proposals, state: History, rng) -> TriageResult:
        signal = observed_score(p.quality, self.sigma_self, rng)
        keep = np.zeros(p.n, dtype=bool)
        used = state.used_in_window(self.window)
        order = np.argsort(-signal, kind="stable")
        counts: dict[int, int] = {}
        for j in order:
            a = int(p.author[j])
            allowance = self.k - int(used[a])
            if allowance > 0 and counts.get(a, 0) < allowance:
                counts[a] = counts.get(a, 0) + 1
                keep[j] = True
        return TriageResult(keep, 0.0, _full_cost(keep))


# ---------------------------------------------------------------------------
# institution
# ---------------------------------------------------------------------------


@dataclass
class InstitutionalCap:
    """At most M proposals per institution, chosen by an internal competition.

    Two properties distinguish this from the individual cap, and they drive
    most of the difference between the two in the results:

    1. The quota binds on a *group*, so slack at a weak institution cannot
       transfer to a strong one where it binds.
    2. The internal competition runs on proposals that have already been
       drafted, so the losers' effort is spent and discarded.
    """

    m: int | None = None  # flat quota
    per_researcher: float | None = None  # size-proportional quota
    sigma_inst: float = SIGMA_EVAL
    # Fraction of a full proposal's effort already spent when the internal
    # competition runs. 1.0 = full proposals are written then discarded;
    # a low value represents an internal outline stage.
    effort_sunk_at_triage: float = 1.0
    name: str = "institutional_cap"
    decider: str = "institution"
    entry_cost: float = 1.0

    def quota(self, size: int) -> int:
        if self.per_researcher is not None:
            return max(1, int(round(self.per_researcher * size)))
        assert self.m is not None
        return self.m

    def apply(self, p: Proposals, state: History, rng) -> TriageResult:
        signal = observed_score(p.quality, self.sigma_inst, rng)
        keep = np.zeros(p.n, dtype=bool)
        for k in np.unique(p.institution):
            idx = np.flatnonzero(p.institution == k)
            if state.inst_size is not None:
                size = int(state.inst_size[k])
            else:  # no population attached: fall back to active submitters
                size = int(np.unique(p.author[idx]).size)
            q = self.quota(size)
            best = idx[np.argsort(-signal[idx], kind="stable")[:q]]
            keep[best] = True
        killed = int(p.n - np.count_nonzero(keep))
        return TriageResult(
            keep,
            wasted_effort=killed * self.effort_sunk_at_triage,
            submitted_effort=_full_cost(keep),
        )


# ---------------------------------------------------------------------------
# mechanical
# ---------------------------------------------------------------------------


@dataclass
class CoolingOff:
    """EPSRC repeatedly-unsuccessful-applicant rule.

    Barred for `bar_rounds` if, within the look-back window, the researcher
    accumulated at least `min_strikes` proposals **ranked in the bottom half of
    the prioritisation list** (not merely unfunded) and their success rate over
    the window is below `success_threshold`.

    Stricter than the scheme it is drawn from: EPSRC's constrained applicants
    could still submit one proposal during the twelve months, whereas this bars
    them outright. Capacity matching absorbs the difference, since instruments
    are tuned to a review load rather than to a nominal setting.
    """

    bar_rounds: int = 1
    window: int = 2
    min_strikes: int = 3
    success_threshold: float = 0.25
    name: str = "cooling_off"
    decider: str = "mechanical"
    entry_cost: float = 1.0

    def apply(self, p: Proposals, state: History, rng) -> TriageResult:
        barred = state.barred_until[p.author] > 0
        keep = ~barred
        # A barred researcher does not write the proposal, so nothing is wasted.
        return TriageResult(keep, 0.0, _full_cost(keep))


@dataclass
class ResubmissionLimit:
    """A project may be resubmitted at most `max_resubmissions` times.

    Distinct from every other mechanism in *which* proposals it removes: it
    targets near-misses specifically, which is exactly the band where the
    original decision was least reliable.

    Binds on the project, which is stricter than the NIH rule it is drawn from:
    there the limit binds the labelled A0/A1 chain, and since 2014 an
    unsuccessful A1 may return as a new A0, so the same idea can come back
    indefinitely.
    """

    max_resubmissions: int = 1
    name: str = "resubmission_limit"
    decider: str = "mechanical"
    entry_cost: float = 1.0

    def apply(self, p: Proposals, state: History, rng) -> TriageResult:
        counts = getattr(state, "proposal_resub_count", None)
        if counts is None:
            return TriageResult(np.ones(p.n, dtype=bool), 0.0, float(p.n))
        over = p.is_resubmission & (counts > self.max_resubmissions)
        keep = ~over
        # The applicant knows the rule, so the barred revision is never written.
        return TriageResult(keep, 0.0, _full_cost(keep))


# ---------------------------------------------------------------------------
# funder
# ---------------------------------------------------------------------------


@dataclass
class TwoStage:
    """Expression-of-interest stage: the funder screens cheaply, then invites.

    The only instrument here whose discarded effort is the *cheap* stage, and
    the only one that can enlarge the pool it selects from rather than shrink
    it. That does not make it the benchmark: it buys its quality by running a
    larger competition, and it is the instrument least able to guarantee a deep
    capacity cut. `eoi_cost` is the outline's cost as a fraction of a full
    proposal.
    """

    invite_ratio: float = 0.3
    eoi_cost: float = 0.15
    sigma_eoi: float = 0.9  # the outline is a noisier signal than a full proposal
    # Screening one outline costs this fraction of reviewing a full proposal.
    # The funder still has to look at every EOI, so a two-stage scheme does not
    # get its volume reduction for free.
    eoi_review_cost: float = 0.2
    name: str = "two_stage"
    decider: str = "funder"
    # Entering costs only the outline, so volume responds. This is the point
    # the two-stage design has to survive: cheap entry means more entries, and
    # more shots at a noisy screen can be won on luck. It therefore tracks
    # `eoi_cost` by construction rather than being a second number that has to
    # be kept in step with it. Set it explicitly only to break that link on
    # purpose, as the mechanism test in exp10 does to suppress the volume
    # response while keeping the screen.
    entry_cost: float | None = None

    def __post_init__(self) -> None:
        if self.entry_cost is None:
            self.entry_cost = self.eoi_cost

    def apply(self, p: Proposals, state: History, rng) -> TriageResult:
        signal = observed_score(p.quality, self.sigma_eoi, rng)
        n_invite = max(1, int(round(self.invite_ratio * p.n)))
        keep = np.zeros(p.n, dtype=bool)
        keep[np.argsort(-signal, kind="stable")[:n_invite]] = True
        n_rejected = p.n - n_invite
        return TriageResult(
            keep,
            # Rejected applicants spent only the outline cost.
            wasted_effort=n_rejected * self.eoi_cost,
            # Invited applicants paid the outline cost too.
            submitted_effort=n_invite * (1.0 + self.eoi_cost),
            # The funder reviews every outline as well as the full proposals.
            review_load=n_invite + self.eoi_review_cost * p.n,
        )


# ---------------------------------------------------------------------------
# control
# ---------------------------------------------------------------------------


@dataclass
class RandomThinning:
    """Placebo arm: remove the same share of proposals, chosen at random.

    A cap does two things at once -- it shrinks the pool the funder chooses
    from (bad, with a fixed budget) and it removes the weaker proposals
    (good). This rule does the first and not the second, so the gap between a
    real mechanism and this one at matched volume IS the value of its triage.

    A mechanism that does no better than this is not selecting; it is only
    making the pile smaller.
    """

    keep_share: float = 0.5
    name: str = "random_thinning"
    decider: str = "control"
    entry_cost: float = 1.0

    def apply(self, p: Proposals, state: History, rng) -> TriageResult:
        n_keep = max(1, int(round(self.keep_share * p.n)))
        keep = np.zeros(p.n, dtype=bool)
        keep[rng.choice(p.n, size=n_keep, replace=False)] = True
        return TriageResult(keep, 0.0, _full_cost(keep))
