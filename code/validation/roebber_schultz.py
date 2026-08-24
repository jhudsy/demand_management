"""Replication of Roebber & Schultz (2011), PLoS ONE 6(4): e18680.

"Peer Review, Program Officers and Science Funding" — the only prior agent-based
model that simulates proposal-limiting funding policies, and therefore the
acceptance test for this project.

The model is reimplemented from the paper's description alone. Parameters the
paper states are in `RSConfig` with their published values; parameters it does
*not* state are collected in `InferredParams` with the reasoning for each choice
recorded in `notes/validation.md`. Every inferred value must be shown not to
change the qualitative conclusions.

Target table (their reported results):

    case  G1 succ  G2 succ  Qbar G1  Qbar G2  G2 share
    a     21.3%    34.6%    111.6    112.1    76.9%
    b     25.2%     9.9%    115.0    112.8    44.8%
    c     34.8%     4.5%    111.6    116.4    19.8%
    d     17.8%    13.5%    117.6    113.8    60.6%
    e     33.9%     5.6%    109.7    112.0    25.1%
    f     29.4%     4.8%    109.1    112.5    19.1%
    g     12.5%    17.4%    115.7    113.0    57.7%

Cases (f) and (g) are the demand-management cases and carry the two qualitative
signatures that matter most:

  (f) a per-scientist cap suppresses the targeted group *and lowers the
      untargeted group's mean funded quality* (115.0 -> 109.1) through reduced
      competition;
  (g) a cooling-off period *inverts its own distributional intent* — the
      high-volume group's funding share rises (44.8% -> 57.7%) while the other
      group's success rate halves.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum, IntEnum

import numpy as np

# --------------------------------------------------------------------------
# Parameters stated in the paper
# --------------------------------------------------------------------------

QS_MEAN = 100.0  # scientist quality ~ N(100, 10)
QS_SD = 10.0
QP_SD = 5.0  # proposal quality ~ N(Qs, 5)
HARRIED_SD = 5.0  # harried reviewer perceives N(Qp, 5)

TIME_UNITS_PER_YEAR = 2  # a time unit is six months
GRANT_DURATION_UNITS = 6  # three-year awards

# NOTE — the paper's prose and its figures disagree, and the figures win.
# The text describes the correct reviewer's bar as "the top 16% of all proposals
# or at least one standard deviation above the mean", which would be 110. But
# Figure 2 shows the implemented test as "Qp >= 105?", and only 105 reproduces
# the case (a) funding rate of 30.2%: Qp ~ N(100, sqrt(10^2 + 5^2) = 11.18), so
# P(Qp >= 105) = 0.33 while P(Qp >= 110) = 0.19. We follow the figure.
THRESHOLD_BASE = 105.0
# Case (c), "at least two standard deviations above the mean", using the
# scientist-quality SD of 10 as the unit consistent with the 105 above.
THRESHOLD_TOP2 = QS_MEAN + 2 * QS_SD

# The selfish reviewer declines below "90% of minimum threshold". See
# notes/validation.md item 6 — 0.9 * 110 = 99, which sits just below the
# population mean. Reproduced as written.
SELFISH_FLOOR_FRACTION = 0.9


class ReviewerType(IntEnum):
    # Integer-valued so the enum can index vectorised arrays directly.
    CORRECT = 0
    HARRIED = 1
    SELFISH = 2


class OfficerType(Enum):
    CORRECT = "correct"  # ranks on true proposal quality Qp
    REPUTATION = "reputation"  # substitutes scientist quality Qs


class Group(IntEnum):
    G1 = 0  # one proposal every two time units; pauses while funded
    G2 = 1  # one proposal every time unit regardless of funding


# --------------------------------------------------------------------------
# Parameters the paper does not state
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class InferredParams:
    """Values not given in the paper. See notes/validation.md.

    `g2_fraction` is not guessed: it is solved analytically from the case (a)
    figures by `solve_group_split()`, which reproduces the reported 76.9% G2
    funding share and 30.2% overall funding rate simultaneously.
    """

    n_scientists: int = 1000
    # STATED in the paper: "a community of N scientists is composed of two
    # evenly sized groups". solve_group_split() is retained as a cross-check on
    # the reported table, not as the source of this value.
    g2_fraction: float = 0.5
    n_time_units: int = 2000  # stated: "a long set of simulations (here, 2000 time steps)"
    burn_in_units: int = 100
    n_replicates: int = 5


@dataclass(frozen=True)
class RSConfig:
    """A single scenario from the paper's Table 1."""

    label: str
    reviewer_mix: dict[ReviewerType, float] = field(
        default_factory=lambda: {
            ReviewerType.CORRECT: 0.6,
            ReviewerType.HARRIED: 0.2,
            ReviewerType.SELFISH: 0.2,
        }
    )
    officer: OfficerType = OfficerType.CORRECT
    n_reviewers: int = 4  # NSF FY2009 mail-review average is "more than four"
    threshold: float = THRESHOLD_BASE
    target_funding_rate: float | None = 0.15
    require_unanimity: bool = True
    # How the officer "estimates the number of fundable proposals". The paper is
    # ambiguous; see notes/validation.md. "proposals" reads the target rate as a
    # share of submissions; "scientists" reads it as a share of the community.
    budget_base: str = "proposals"

    # Case-specific switches
    g2_quality_delta: float = 0.0  # cases (d) +5 and (e) -5
    g2_max_concurrent_grants: int | None = None  # case (f): limit to one
    cooling_off: bool = False  # case (g)

    # EPSRC-style cooling-off rule, used only when cooling_off is True.
    cooling_off_units: int = 2  # 12 months
    cooling_off_window_units: int = 4  # 24-month look-back
    cooling_off_min_failures: int = 3
    cooling_off_success_rate: float = 0.25

    inferred: InferredParams = field(default_factory=InferredParams)


def solve_group_split(
    g1_success: float = 0.213,
    g2_success: float = 0.346,
    g2_funding_share: float = 0.769,
    grant_units: int = GRANT_DURATION_UNITS,
) -> dict[str, float]:
    """Recover the G1:G2 population split from the paper's case (a) figures.

    The paper reports per-group success rates and G2's share of funding but not
    the population split. Those three numbers over-determine it, which is what
    makes the recovery trustworthy rather than a guess.

    Submissions satisfy
        F2 / (F1 + F2) = g2_funding_share,  F_g = success_g * S_g
    giving S2/S1. In steady state a G1 scientist is funded a fraction phi1 of
    the time, submits only while unfunded, and therefore submits at rate
    0.5 * (1 - phi1) per time unit, while G2 submits at rate 1. Self-consistency
    of phi1 with the award rate closes the system.

    Returns the implied ratios plus the overall funding rate, which is *not*
    used in the fit and therefore serves as an independent check against the
    paper's reported 30.2%.
    """
    # S2/S1 from the funding-share identity.
    s_ratio = (g1_success * g2_funding_share) / (
        g2_success * (1.0 - g2_funding_share)
    )

    # Steady-state funded fraction for G1: phi1 = grant_units * award_rate,
    # award_rate = 0.5 * (1 - phi1) * g1_success.
    k = grant_units * 0.5 * g1_success
    phi1 = k / (1.0 + k)

    g1_submission_rate = 0.5 * (1.0 - phi1)
    n_ratio = s_ratio * g1_submission_rate  # n2 / n1

    g2_fraction = n_ratio / (1.0 + n_ratio)

    # Independent check: overall funding rate implied by the recovered split.
    s1, s2 = 1.0, s_ratio
    overall = (g1_success * s1 + g2_success * s2) / (s1 + s2)

    return {
        "submission_ratio_s2_s1": s_ratio,
        "g1_funded_fraction": phi1,
        "population_ratio_n2_n1": n_ratio,
        "g2_fraction": g2_fraction,
        "implied_overall_funding_rate": overall,
    }


# --------------------------------------------------------------------------
# Model state
# --------------------------------------------------------------------------


@dataclass
class Scientists:
    """Vectorised scientist population. Index i identifies a scientist."""

    quality: np.ndarray  # Qs
    group: np.ndarray  # Group value, 0 or 1
    reviewer_type: np.ndarray  # ReviewerType index
    expiry: np.ndarray  # (n, GRANT_DURATION_UNITS+1) circular buffer of grant expiries
    n_concurrent_grants: np.ndarray  # count of currently held grants
    latest_end: np.ndarray  # time unit at which the last-held grant ends
    cooling_until: np.ndarray  # time unit index until which barred
    g1_slot: np.ndarray  # which half-year a G1 scientist submits in
    submitted: np.ndarray  # lifetime submission count (post burn-in)
    funded: np.ndarray  # lifetime award count (post burn-in)

    @property
    def n(self) -> int:
        return self.quality.size


def build_population(cfg: RSConfig, rng: np.random.Generator) -> Scientists:
    n = cfg.inferred.n_scientists
    quality = rng.normal(QS_MEAN, QS_SD, size=n)

    n_g2 = int(round(n * cfg.inferred.g2_fraction))
    group = np.zeros(n, dtype=np.int8)
    group[rng.permutation(n)[:n_g2]] = Group.G2.value

    probs = np.array([cfg.reviewer_mix[t] for t in ReviewerType], dtype=float)
    reviewer_type = rng.choice(len(ReviewerType), size=n, p=probs / probs.sum())

    zeros_i = np.zeros(n, dtype=np.int32)
    return Scientists(
        quality=quality,
        group=group,
        reviewer_type=reviewer_type,
        expiry=np.zeros((n, GRANT_DURATION_UNITS + 1), dtype=np.int32),
        n_concurrent_grants=zeros_i.copy(),
        latest_end=np.full(n, -1, dtype=np.int32),
        cooling_until=np.full(n, -1, dtype=np.int32),
        # "one proposal every two time units (split randomly between those
        # times)" — each G1 scientist has a fixed half-year slot.
        g1_slot=rng.integers(0, 2, size=n).astype(np.int32),
        submitted=zeros_i.copy(),
        funded=zeros_i.copy(),
    )


# --------------------------------------------------------------------------
# Review
# --------------------------------------------------------------------------


def _recommendations(
    proposal_quality: np.ndarray,
    author_idx: np.ndarray,
    sci: Scientists,
    cfg: RSConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Binary recommend/decline for every (proposal, reviewer) pair.

    Returns a boolean array of shape (n_proposals, n_reviewers). Reviewers are
    drawn at random for each proposal with the author excluded.
    """
    n_prop = proposal_quality.size
    k = cfg.n_reviewers
    if n_prop == 0:
        return np.zeros((0, k), dtype=bool)

    # Draw reviewers, resampling any that collide with the author.
    reviewers = rng.integers(0, sci.n, size=(n_prop, k))
    clash = reviewers == author_idx[:, None]
    while clash.any():
        reviewers[clash] = rng.integers(0, sci.n, size=int(clash.sum()))
        clash = reviewers == author_idx[:, None]

    qp = proposal_quality[:, None]
    rtype = sci.reviewer_type[reviewers]
    rquality = sci.quality[reviewers]

    rec = np.zeros((n_prop, k), dtype=bool)

    # Correct: recommend iff the proposal clears the threshold.
    is_correct = rtype == ReviewerType.CORRECT.value
    rec |= is_correct & (qp >= cfg.threshold)

    # Harried: same rule applied to a noisy perception of quality.
    is_harried = rtype == ReviewerType.HARRIED.value
    perceived = qp + rng.normal(0.0, HARRIED_SD, size=(n_prop, k))
    rec |= is_harried & (perceived >= cfg.threshold)

    # Selfish: decline if the proposal beats the reviewer's own quality, or
    # falls below 90% of the threshold; otherwise recommend.
    is_selfish = rtype == ReviewerType.SELFISH.value
    floor = SELFISH_FLOOR_FRACTION * cfg.threshold
    selfish_ok = (qp <= rquality) & (qp >= floor)
    rec |= is_selfish & selfish_ok

    return rec


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------


@dataclass
class RunResult:
    g1_success: float
    g2_success: float
    g1_funded_quality: float
    g2_funded_quality: float
    g2_funding_share: float
    overall_funding_rate: float
    reviews_performed: int


def _submitters(sci: Scientists, cfg: RSConfig, t: int) -> np.ndarray:
    """Boolean mask of scientists submitting a proposal this time unit."""
    is_g1 = sci.group == Group.G1.value
    is_g2 = ~is_g1

    # G1: one proposal per year, the half-year chosen at random (`g1_slot`),
    # and suppressed while funded EXCEPT in the final year of the award --
    # "G1 scientists who obtain funding do not submit new proposals until the
    # final year of the grant" (grant length 6 time units, so the final year is
    # the last 2).
    g1_turn = sci.g1_slot == (t % 2)
    blocked = (sci.latest_end - t) > 2
    submit = is_g1 & g1_turn & ~blocked

    # G2 submits every time unit regardless of funding status.
    g2_submit = is_g2.copy()
    if cfg.g2_max_concurrent_grants is not None:
        # Case (f): "limiting G2 scientists to one funded grant".
        g2_submit &= sci.n_concurrent_grants < cfg.g2_max_concurrent_grants
    submit |= g2_submit

    if cfg.cooling_off:
        submit &= sci.cooling_until <= t

    return submit


def _apply_cooling_off(
    sci: Scientists,
    cfg: RSConfig,
    t: int,
    recent_submissions: list[np.ndarray],
    recent_awards: list[np.ndarray],
    recent_strikes: list[np.ndarray],
) -> None:
    """EPSRC-style repeatedly-unsuccessful-applicant rule.

    Barred for `cooling_off_units` if, within the look-back window, the
    scientist accumulated at least `cooling_off_min_failures` failures *and*
    their personal success rate over that window is below the threshold.
    """
    window = cfg.cooling_off_window_units
    subs = np.sum(recent_submissions[-window:], axis=0)
    awards = np.sum(recent_awards[-window:], axis=0)
    # A "strike" is NOT merely an unsuccessful proposal. The EPSRC rule counts
    # proposals "ranked in the bottom half of a funding prioritization list or
    # rejected by a panel review" -- so an unfunded proposal ranked in the top
    # half does not count against the applicant.
    failures = np.sum(recent_strikes[-window:], axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(subs > 0, awards / np.maximum(subs, 1), 1.0)

    triggered = (failures >= cfg.cooling_off_min_failures) & (
        rate < cfg.cooling_off_success_rate
    )
    sci.cooling_until[triggered] = t + cfg.cooling_off_units


def run_once(cfg: RSConfig, seed: int) -> RunResult:
    rng = np.random.default_rng(seed)
    sci = build_population(cfg, rng)
    inf = cfg.inferred

    # Post-burn-in accumulators, by group.
    sub_count = np.zeros(2, dtype=np.int64)
    fund_count = np.zeros(2, dtype=np.int64)
    fund_quality_sum = np.zeros(2, dtype=np.float64)
    reviews = 0

    recent_submissions: list[np.ndarray] = []
    recent_awards: list[np.ndarray] = []
    recent_strikes: list[np.ndarray] = []

    # Year-level bookkeeping for the surplus rule.
    year_proposals = 0
    year_awards = 0

    for t in range(inf.n_time_units):
        recording = t >= inf.burn_in_units

        # Retire grants expiring in this time unit.
        slot = t % (GRANT_DURATION_UNITS + 1)
        sci.n_concurrent_grants -= sci.expiry[:, slot]
        sci.expiry[:, slot] = 0

        # --- submission -------------------------------------------------
        submit_mask = _submitters(sci, cfg, t)
        authors = np.flatnonzero(submit_mask)
        n_prop = authors.size

        qp = rng.normal(sci.quality[authors], QP_SD)
        if cfg.g2_quality_delta:
            is_g2 = sci.group[authors] == Group.G2.value
            qp = qp + is_g2 * cfg.g2_quality_delta

        # --- review -----------------------------------------------------
        rec = _recommendations(qp, authors, sci, cfg, rng)
        reviews += n_prop * cfg.n_reviewers
        if cfg.require_unanimity:
            passes = rec.all(axis=1)
        else:
            passes = rec.sum(axis=1) >= (cfg.n_reviewers - 1)

        # --- allocation -------------------------------------------------
        # The officer ranks on true proposal quality, or on the author's
        # reputation if reputation-based.
        rank_key = qp if cfg.officer is OfficerType.CORRECT else sci.quality[authors]

        year_proposals += n_prop
        if cfg.target_funding_rate is None:
            chosen = np.flatnonzero(passes)
        else:
            if cfg.budget_base == "scientists":
                budget = int(round(cfg.target_funding_rate * sci.n / 2))
            else:
                budget = int(round(cfg.target_funding_rate * n_prop))
            eligible = np.flatnonzero(passes)
            order = eligible[np.argsort(-rank_key[eligible], kind="stable")]
            chosen = order[:budget]

            # Year-end surplus: top up from the highest-rated unfunded
            # proposals of the current half-year until the annual target is met.
            if (t % 2) == 1:
                annual_target = int(round(cfg.target_funding_rate * year_proposals))
                shortfall = annual_target - (year_awards + chosen.size)
                if shortfall > 0:
                    remaining = np.setdiff1d(
                        np.arange(n_prop), chosen, assume_unique=False
                    )
                    topup_order = remaining[
                        np.argsort(-rank_key[remaining], kind="stable")
                    ]
                    chosen = np.concatenate([chosen, topup_order[:shortfall]])

        # --- bookkeeping ------------------------------------------------
        awarded_authors = authors[chosen]
        np.add.at(sci.n_concurrent_grants, awarded_authors, 1)
        end_slot = (t + GRANT_DURATION_UNITS) % (GRANT_DURATION_UNITS + 1)
        np.add.at(sci.expiry, (awarded_authors, end_slot), 1)
        np.maximum.at(sci.latest_end, awarded_authors, t + GRANT_DURATION_UNITS)

        year_awards += chosen.size
        if (t % 2) == 1:
            year_proposals = 0
            year_awards = 0

        if recording:
            g = sci.group[authors]
            for gi in (Group.G1.value, Group.G2.value):
                sub_count[gi] += int(np.count_nonzero(g == gi))
            gc = sci.group[awarded_authors]
            for gi in (Group.G1.value, Group.G2.value):
                m = gc == gi
                fund_count[gi] += int(np.count_nonzero(m))
                fund_quality_sum[gi] += float(qp[chosen][m].sum())

        if cfg.cooling_off:
            sub_vec = np.zeros(sci.n, dtype=np.int32)
            sub_vec[authors] = 1
            awd_vec = np.zeros(sci.n, dtype=np.int32)
            awd_vec[awarded_authors] = 1
            # Bottom half of the officer's prioritisation list, among unfunded.
            strike_vec = np.zeros(sci.n, dtype=np.int32)
            if n_prop:
                funded_mask = np.zeros(n_prop, dtype=bool)
                funded_mask[chosen] = True
                median_rank = np.median(rank_key)
                bottom_half = (~funded_mask) & (rank_key < median_rank)
                strike_vec[authors[bottom_half]] = 1
            recent_strikes.append(strike_vec)
            recent_submissions.append(sub_vec)
            recent_awards.append(awd_vec)
            if len(recent_submissions) >= cfg.cooling_off_window_units:
                _apply_cooling_off(
                    sci, cfg, t, recent_submissions, recent_awards, recent_strikes
                )
                w = cfg.cooling_off_window_units
                recent_submissions = recent_submissions[-w:]
                recent_awards = recent_awards[-w:]
                recent_strikes = recent_strikes[-w:]

    total_funded = fund_count.sum()
    total_sub = sub_count.sum()
    return RunResult(
        g1_success=fund_count[0] / max(sub_count[0], 1),
        g2_success=fund_count[1] / max(sub_count[1], 1),
        g1_funded_quality=fund_quality_sum[0] / max(fund_count[0], 1),
        g2_funded_quality=fund_quality_sum[1] / max(fund_count[1], 1),
        g2_funding_share=fund_count[1] / max(total_funded, 1),
        overall_funding_rate=total_funded / max(total_sub, 1),
        reviews_performed=reviews,
    )


def run(cfg: RSConfig, base_seed: int = 20110412) -> dict[str, float]:
    """Run all replicates and return the mean of each reported statistic."""
    results = [run_once(cfg, base_seed + r) for r in range(cfg.inferred.n_replicates)]
    keys = [
        "g1_success",
        "g2_success",
        "g1_funded_quality",
        "g2_funded_quality",
        "g2_funding_share",
        "overall_funding_rate",
        "reviews_performed",
    ]
    return {k: float(np.mean([getattr(r, k) for r in results])) for k in keys}


# --------------------------------------------------------------------------
# The paper's eight scenarios
# --------------------------------------------------------------------------

ALL_CORRECT = {
    ReviewerType.CORRECT: 1.0,
    ReviewerType.HARRIED: 0.0,
    ReviewerType.SELFISH: 0.0,
}


def scenarios() -> dict[str, RSConfig]:
    base = RSConfig(label="b_baseline")
    return {
        "a_perfect": replace(
            base,
            label="a_perfect",
            reviewer_mix=ALL_CORRECT,
            target_funding_rate=None,
        ),
        "b_baseline": base,
        "c_selective": replace(
            base, label="c_selective", threshold=THRESHOLD_TOP2
        ),
        "d_positive_feedback": replace(
            base, label="d_positive_feedback", g2_quality_delta=+5.0
        ),
        "e_negative_feedback": replace(
            base, label="e_negative_feedback", g2_quality_delta=-5.0
        ),
        "f_g2_one_grant": replace(
            base, label="f_g2_one_grant", g2_max_concurrent_grants=1
        ),
        "g_cooling_off": replace(base, label="g_cooling_off", cooling_off=True),
    }


# Reported values from the paper, for the acceptance test.
PUBLISHED = {
    "a_perfect": dict(g1_success=0.213, g2_success=0.346,
                      g1_funded_quality=111.6, g2_funded_quality=112.1,
                      g2_funding_share=0.769),
    "b_baseline": dict(g1_success=0.252, g2_success=0.099,
                       g1_funded_quality=115.0, g2_funded_quality=112.8,
                       g2_funding_share=0.448),
    "c_selective": dict(g1_success=0.348, g2_success=0.045,
                        g1_funded_quality=111.6, g2_funded_quality=116.4,
                        g2_funding_share=0.198),
    "d_positive_feedback": dict(g1_success=0.178, g2_success=0.135,
                                g1_funded_quality=117.6, g2_funded_quality=113.8,
                                g2_funding_share=0.606),
    "e_negative_feedback": dict(g1_success=0.339, g2_success=0.056,
                                g1_funded_quality=109.7, g2_funded_quality=112.0,
                                g2_funding_share=0.251),
    "f_g2_one_grant": dict(g1_success=0.294, g2_success=0.048,
                           g1_funded_quality=109.1, g2_funded_quality=112.5,
                           g2_funding_share=0.191),
    "g_cooling_off": dict(g1_success=0.125, g2_success=0.174,
                          g1_funded_quality=115.7, g2_funded_quality=113.0,
                          g2_funding_share=0.577),
}
