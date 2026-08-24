"""Round loop: generate proposals, triage, evaluate, allocate, measure."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dm import population
from dm.config import RunConfig
from dm.evaluation import SIGMA_EVAL, observed_score
from dm.triage import History, Proposals, TriageRule


@dataclass(frozen=True)
class SimConfig:
    n_rounds: int = 20
    burn_in: int = 5
    proposals_per_researcher: float = 1.5  # intended at full-proposal cost
    max_intended: int = 12
    sigma_idea: float = 0.5  # proposal-to-proposal variation within a researcher
    sigma_eval: float = SIGMA_EVAL  # fitted; see evaluation.SIGMA_EVAL
    # Volume responds to the cost of entering. A mechanism that makes entry
    # cheap (an EOI stage) gets more entries, which is the whole reason cheap
    # entry is not automatically a good thing: more shots at a noisy screen is
    # more chance of getting through on luck rather than quality.
    volume_elasticity: float = 0.5
    # Resubmission. A proposal that is not funded may come back next round as a
    # resubmission of the same project. Near-misses are the most likely to
    # return, which is what makes a resubmission LIMIT bite hardest exactly at
    # the funding margin -- the band where the original decision was least
    # reliable. `nearmiss_band` is a modelling choice, not a measured quantity:
    # Graves et al. establish that the margin is unreliable, not how wide it is.
    p_resubmit_base: float = 0.35
    p_resubmit_nearmiss: float = 0.75  # for proposals just below the line
    nearmiss_band: float = 0.5  # top X of the unfunded, by rank
    resubmission_gain: float = 0.10  # quality improvement on revision
    # The funder has a FIXED number of grants, not a fixed success rate. The
    # budget is set from the reference submission volume (no cap, full-cost
    # entry) and then held constant across every mechanism, so a mechanism that
    # inflates volume lowers everyone's success rate -- which is the actual
    # dynamic demand management exists to address.
    payline: float = 0.15  # grants as a share of the REFERENCE volume
    # Effort offset: fraction of the effort freed by a cap that is re-spent on
    # the proposals that survive. Barnett (2015) suggests this is close to 1.
    psi: float = 0.0


@dataclass
class RoundRecord:
    submitted: int
    intended: int
    funded: int
    # quality
    mean_quality_funded: float
    # concentration
    hhi_institution: float
    # applicant level -- how well the funded SET of people tracks quality.
    # Falls when volume rather than quality is what wins funding.
    applicant_quality_premium: float
    # Funding rate of high-volume vs low-volume researchers AT MATCHED quality.
    # >1 means submitting more pays off independently of being better.
    volume_advantage: float
    # Gini of submission counts among active researchers. Always defined, so it
    # covers the case where volume_advantage is not: an instrument can score
    # well on volume_advantage either by neutralising the effect of volume or by
    # eliminating volume variation, and only this distinguishes them.
    volume_gini: float
    n_active: int
    n_winners: int
    # cost
    wasted_effort: float
    submitted_effort: float
    review_load: float  # funder review capacity consumed, full-proposal-equivalents
    # the null metric
    success_rate: float


@dataclass
class SimResult:
    rounds: list[RoundRecord] = field(default_factory=list)

    def mean(self, attr: str) -> float:
        """Mean over rounds, skipping rounds where the metric is undefined.

        Some metrics are undefined in some rounds by design: the volume
        advantage has nothing to measure when an instrument leaves no variation
        in submission counts. If every round is undefined the mean is NaN, which
        must be produced without calling nanmean on an all-NaN slice.
        """
        v = np.asarray([getattr(r, attr) for r in self.rounds], dtype=float)
        if v.size == 0 or np.all(np.isnan(v)):
            return float("nan")
        return float(np.nanmean(v))

    def round_sd(self, attr: str) -> float:
        """Round-to-round spread within one population. A good mean with a large
        spread gives the funder a portfolio it cannot count on year to year."""
        v = np.asarray([getattr(r, attr) for r in self.rounds], dtype=float)
        if np.count_nonzero(~np.isnan(v)) < 2:
            return float("nan")
        return float(np.nanstd(v, ddof=1))


def _generate(
    res: population.Researchers,
    cfg: SimConfig,
    rng: np.random.Generator,
    entry_cost: float = 1.0,
    carried: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[Proposals, np.ndarray]:
    """Intended proposals, before any demand-management filter.

    `entry_cost` is what it costs to put one proposal in front of the funder,
    as a fraction of a full proposal. Volume responds to it with elasticity
    `volume_elasticity`, so a cheap first stage draws more entries.
    """
    rate = cfg.proposals_per_researcher * (1.0 / max(entry_cost, 1e-6)) ** cfg.volume_elasticity
    n_each = np.minimum(rng.poisson(rate, size=res.n), cfg.max_intended)
    author = np.repeat(np.arange(res.n), n_each)
    quality = res.quality[author] + rng.normal(0.0, cfg.sigma_idea, size=author.size)
    resub_count = np.zeros(author.size, dtype=np.int32)
    is_resub = np.zeros(author.size, dtype=bool)

    if carried is not None and carried[0].size:
        c_author, c_quality, c_count = carried
        author = np.concatenate([author, c_author])
        quality = np.concatenate([quality, c_quality])
        resub_count = np.concatenate([resub_count, c_count])
        is_resub = np.concatenate([is_resub, np.ones(c_author.size, dtype=bool)])

    p = Proposals(
        author=author,
        institution=res.institution[author],
        quality=quality,
        is_resubmission=is_resub,
    )
    return p, resub_count


def run(
    run_cfg: RunConfig,
    sim_cfg: SimConfig,
    rule: TriageRule,
    seed: int | None = None,
) -> SimResult:
    rng = np.random.default_rng(run_cfg.seed if seed is None else seed)
    inst, res = population.build(run_cfg, rng)
    hist = History(n_researchers=res.n, inst_size=inst.size)

    # A per-investigator cap with a multi-round window has to start in steady
    # state. If every researcher begins with an empty window they all submit in
    # round 1 and are all locked out together, producing a synchronised
    # boom-bust cycle that is an artefact of initialisation rather than a
    # property of the policy. Seed the window so researchers are spread
    # uniformly through their cycle.
    w = int(getattr(rule, "window", 1))
    if w > 1:
        occupancy = min(1.0, getattr(rule, "k", 1) / w)
        hist.sub_window = (
            rng.random(hist.sub_window.shape) < occupancy
        ).astype(np.int32)
    out = SimResult()
    # Fixed grant budget, independent of how many proposals arrive.
    budget = max(
        1,
        int(round(
            sim_cfg.payline * res.n * sim_cfg.proposals_per_researcher
        )),
    )
    carried: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    for t in range(sim_cfg.n_rounds):
        intended, resub_count = _generate(
            res, sim_cfg, rng,
            entry_cost=getattr(rule, "entry_cost", 1.0),
            carried=carried,
        )
        hist.proposal_resub_count = resub_count
        triaged = rule.apply(intended, hist, rng)
        keep = triaged.submitted

        sub_q = intended.quality[keep]
        sub_author = intended.author[keep]
        sub_inst = intended.institution[keep]

        # Effort offset: effort freed by the filter is partly re-spent on the
        # proposals that survive, which is what the field evidence suggests.
        freed = (intended.n - sub_q.size) - triaged.wasted_effort
        submitted_effort = triaged.submitted_effort + sim_cfg.psi * max(freed, 0.0)

        # Evaluation collapses everything between quality and outcome into one
        # noise term; allocation is then simply the top of the ranking.
        score = observed_score(sub_q, sim_cfg.sigma_eval, rng)
        winners = np.argsort(-score, kind="stable")[:budget]

        # --- update mechanical-rule history ------------------------------
        _update_history(hist, res.n, sub_author, winners, score, rule)

        # --- decide which rejected proposals come back next round ---------
        carried = _carry_forward(
            intended, keep, winners, score, resub_count, sim_cfg, rng
        )

        # --- metrics ------------------------------------------------------
        income = np.zeros(inst.n)
        np.add.at(income, sub_inst[winners], 1.0)

        premium, vol_adv, vol_gini, n_active, n_winners = _applicant_metrics(
            res, sub_author, winners
        )

        if t >= sim_cfg.burn_in:
            out.rounds.append(
                RoundRecord(
                    submitted=int(sub_q.size),
                    intended=int(intended.n),
                    funded=int(winners.size),
                    # A very tight cap can leave a round with nothing to fund.
                    # Quality is undefined there rather than zero, so it is
                    # recorded as NaN and skipped by the aggregator.
                    mean_quality_funded=float(sub_q[winners].mean())
                    if winners.size
                    else float("nan"),
                    hhi_institution=_hhi(income),
                    applicant_quality_premium=premium,
                    volume_advantage=vol_adv,
                    volume_gini=vol_gini,
                    n_active=n_active,
                    n_winners=n_winners,
                    wasted_effort=float(triaged.wasted_effort),
                    submitted_effort=float(submitted_effort),
                    review_load=float(triaged.review_load),
                    success_rate=float(winners.size / max(sub_q.size, 1)),
                )
            )
    return out


def _update_history(hist, n_res, sub_author, winners, score, rule) -> None:
    """Maintain the rolling record that a cooling-off rule conditions on."""
    hist.barred_until = np.maximum(hist.barred_until - 1, 0)
    z = np.zeros(n_res, dtype=np.int32)
    if sub_author.size == 0:
        hist.record_round(z, z, z)
        return

    funded_mask = np.zeros(sub_author.size, dtype=bool)
    funded_mask[winners] = True
    # A strike is a proposal ranked in the bottom half of the funder's list,
    # which is stricter than merely being unfunded.
    strike = (~funded_mask) & (score < np.median(score))

    hist.record_round(
        np.bincount(sub_author, minlength=n_res).astype(np.int32),
        np.bincount(sub_author[funded_mask], minlength=n_res).astype(np.int32),
        np.bincount(sub_author[strike], minlength=n_res).astype(np.int32),
    )

    if getattr(rule, "name", "") == "cooling_off":
        # Both conditions are evaluated over the look-back window, not over the
        # round just played: the rule bars an applicant for an accumulated
        # record, and evaluating it per round makes it fire on a single bad
        # round instead.
        subs_w, awards_w, strikes_w = hist.record_over(rule.window)
        rate = np.where(subs_w > 0, awards_w / np.maximum(subs_w, 1), 1.0)
        trip = (strikes_w >= rule.min_strikes) & (rate < rule.success_threshold)
        hist.barred_until[trip] = rule.bar_rounds


def _hhi(income: np.ndarray) -> float:
    total = income.sum()
    if total <= 0:
        return 0.0
    shares = income / total
    return float((shares**2).sum())


def _applicant_metrics(res, sub_author, winners):
    """Applicant-level measures.

    Returns (premium, volume_advantage, volume_gini, n_active, n_winners).

    `premium` is how much better, in latent quality, the funded researchers are
    than the researchers who submitted. Both of those populations are changed by
    the instrument, so this is a difference between two endogenous groups rather
    than a clean measure of selection accuracy. `n_active` and `n_winners` are
    returned with it so the populations being compared are visible.

    `volume_advantage` compares the funding rate of researchers in the top
    tercile of submission count against the bottom tercile, *within* quality
    quintiles, so quality is held fixed. Above 1 means volume buys funding on
    its own.

    It is NaN when no quality stratum has any variation in submission count,
    which is exactly what a tight per-investigator cap produces: if everyone
    submits once there is no high-volume group to compare with a low-volume one,
    and the ratio is undefined rather than 1. Returning 1 there would credit the
    instrument with neutralising an advantage when what it has done is remove
    the variation the measure needs. `volume_gini`, the Gini coefficient of
    submission counts among active researchers, is always defined and shows how
    much volume dispersion the instrument leaves behind.
    """
    n = res.n
    submitted = np.bincount(sub_author, minlength=n)
    funded_any = np.zeros(n, dtype=bool)
    if winners.size:
        funded_any[np.unique(sub_author[winners])] = True

    active = submitted > 0
    n_active = int(active.sum())
    n_winners = int(funded_any.sum())
    if not active.any() or not funded_any.any():
        return float("nan"), float("nan"), float("nan"), n_active, n_winners

    premium = float(res.quality[funded_any].mean() - res.quality[active].mean())

    q = res.quality[active]
    v = submitted[active]
    f = funded_any[active]
    gini = _gini_counts(v)

    edges = np.quantile(q, [0.2, 0.4, 0.6, 0.8])
    strata = np.digitize(q, edges)
    hi_rates, lo_rates = [], []
    for s_i in range(5):
        m = strata == s_i
        if m.sum() < 20:
            continue
        vv, ff = v[m], f[m]
        if vv.max() == vv.min():
            continue
        hi = vv >= np.quantile(vv, 2 / 3)
        lo = vv <= np.quantile(vv, 1 / 3)
        if hi.sum() and lo.sum():
            hi_rates.append(ff[hi].mean())
            lo_rates.append(ff[lo].mean())
    if not hi_rates:
        # No stratum has volume variation: the ratio does not exist.
        return premium, float("nan"), gini, n_active, n_winners
    lo_mean = float(np.mean(lo_rates))
    vol_adv = float(np.mean(hi_rates) / lo_mean) if lo_mean > 0 else float("nan")
    return premium, vol_adv, gini, n_active, n_winners


def _gini_counts(x: np.ndarray) -> float:
    """Gini coefficient of submission counts. 0 = everyone submits equally."""
    if x.size == 0:
        return float("nan")
    s = np.sort(x.astype(float))
    total = s.sum()
    if total <= 0:
        return 0.0
    i = np.arange(1, s.size + 1)
    return float((2 * (i * s).sum()) / (s.size * total) - (s.size + 1) / s.size)


def _carry_forward(intended, keep, winners, score, resub_count, cfg, rng):
    """Rejected proposals that will be resubmitted next round.

    Near-misses -- the unfunded proposals nearest the line -- return at a higher
    rate than the rest, which is what a resubmission LIMIT is really acting on.
    """
    sub_idx = np.flatnonzero(keep)
    if sub_idx.size == 0:
        return None
    funded = np.zeros(sub_idx.size, dtype=bool)
    funded[winners] = True
    rejected_local = np.flatnonzero(~funded)
    if rejected_local.size == 0:
        return None

    # Rank the rejected by score; the top `nearmiss_band` are near-misses.
    order = np.argsort(-score[rejected_local], kind="stable")
    n_near = int(round(cfg.nearmiss_band * rejected_local.size))
    p = np.full(rejected_local.size, cfg.p_resubmit_base)
    p[order[:n_near]] = cfg.p_resubmit_nearmiss

    returns = rng.random(rejected_local.size) < p
    take = sub_idx[rejected_local[returns]]
    if take.size == 0:
        return None
    return (
        intended.author[take],
        intended.quality[take] + cfg.resubmission_gain,
        resub_count[take] + 1,
    )
