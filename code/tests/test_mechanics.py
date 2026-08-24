"""Property tests for the round loop, the instruments and the measures.

Every defect found in this model so far has been here rather than in the
population layer: a budget that scaled with submissions instead of staying
fixed, a welfare term that inferred the grant count from the payline, a volume
ratio that reported "cannot measure" as "no advantage", a cooling-off rule that
ignored its own look-back window, and a proportional quota that scaled with
demand instead of headcount. Each of those is pinned here.
"""
from __future__ import annotations

import numpy as np
import pytest

from dm import triage
from dm.config import InstitutionConfig, RunConfig
from dm.runner import paired_diff, replicate, welfare
from dm.scenarios import scenarios
from dm.evaluation import SIGMA_EVAL
from dm.simulate import SimConfig, run

SIGMA = SIGMA_EVAL


@pytest.fixture
def cfg():
    return scenarios()["S4_big_and_good"]


@pytest.fixture
def sim():
    return SimConfig(sigma_eval=SIGMA, n_rounds=8, burn_in=2)


def _hist(n=50, sizes=None):
    return triage.History(n_researchers=n, inst_size=sizes)


def _proposals(author, institution, quality, is_resub=None):
    author = np.asarray(author)
    return triage.Proposals(
        author=author,
        institution=np.asarray(institution),
        quality=np.asarray(quality, dtype=float),
        is_resubmission=(
            np.zeros(author.size, dtype=bool) if is_resub is None
            else np.asarray(is_resub)
        ),
    )


# ---------------------------------------------------------------------------
# the budget
# ---------------------------------------------------------------------------


def test_grant_count_is_fixed_across_instruments(cfg, sim):
    """The funder awards the same number of grants whatever arrives.

    A budget defined as a share of submissions would pin the success rate and
    make every volume effect invisible.
    """
    expected = round(sim.payline * cfg.institutions.n_researchers
                     * sim.proposals_per_researcher)
    for rule in (
        triage.NoCap(),
        triage.IndividualCap(k=2, sigma_self=SIGMA),
        triage.InstitutionalCap(m=30, sigma_inst=SIGMA),
        triage.TwoStage(invite_ratio=0.3),
    ):
        r = run(cfg, sim, rule, seed=1)
        awarded = {rec.funded for rec in r.rounds}
        assert awarded == {expected}, f"{rule.name}: {awarded} != {expected}"


def test_success_rate_moves_when_volume_moves(cfg, sim):
    """With a fixed budget, an instrument that inflates volume must lower the
    realised success rate rather than hold it constant."""
    a = replicate(cfg, sim, triage.NoCap(), n_seeds=3)["success_rate"].mean
    b = replicate(cfg, sim, triage.IndividualCap(k=1, sigma_self=SIGMA),
                  n_seeds=3)["success_rate"].mean
    assert b > a * 1.5, (a, b)


# ---------------------------------------------------------------------------
# the measures
# ---------------------------------------------------------------------------


def test_welfare_uses_the_awarded_grant_count(cfg, sim):
    """Inferring the grant count as payline x submitted rewards volume
    inflation and punishes volume reduction."""
    est = replicate(cfg, sim, triage.NoCap(), n_seeds=3)
    expected = (est["mean_quality_funded"].mean * est["funded"].mean
                / (est["submitted_effort"].mean + est["wasted_effort"].mean))
    assert welfare(est) == pytest.approx(expected, rel=1e-9)
    # The discredited form would differ substantially for the uncapped arm.
    naive = (est["mean_quality_funded"].mean * est["submitted"].mean * sim.payline
             / (est["submitted_effort"].mean + est["wasted_effort"].mean))
    assert abs(naive - welfare(est)) > 0.05 * welfare(est)


def test_volume_advantage_is_undefined_without_volume_variation(cfg, sim):
    """A cap of one per round leaves nobody submitting more than anybody else.

    The ratio then has no high-volume group to measure, and reporting 1.0 would
    credit the instrument with neutralising an advantage it has merely made
    unmeasurable.
    """
    est = replicate(cfg, sim, triage.IndividualCap(k=1, sigma_self=SIGMA), n_seeds=3)
    assert np.isnan(est["volume_advantage"].mean)
    assert est["volume_gini"].mean == pytest.approx(0.0, abs=1e-9)

    # With room to differ, both are defined and the Gini is positive.
    est2 = replicate(cfg, sim, triage.NoCap(), n_seeds=3)
    assert not np.isnan(est2["volume_advantage"].mean)
    assert est2["volume_gini"].mean > 0.1


def test_two_stage_review_load_counts_outline_screening():
    """A two-stage scheme does not get its reduction for free."""
    n = 1000
    p = _proposals(np.arange(n), np.zeros(n, int), np.random.default_rng(0).normal(size=n))
    rule = triage.TwoStage(invite_ratio=0.2, eoi_review_cost=0.25)
    res = rule.apply(p, _hist(n), np.random.default_rng(0))
    invited = np.count_nonzero(res.submitted)
    assert res.review_load == pytest.approx(invited + 0.25 * n)
    assert res.review_load > invited


def test_effort_is_wasted_only_where_drafts_are_discarded():
    """A cap announced in advance stops the proposal being written; an internal
    competition discards work already done."""
    n, rng = 400, np.random.default_rng(0)
    q = rng.normal(size=n)
    author = np.repeat(np.arange(100), 4)
    inst = np.repeat(np.arange(10), 40)
    p = _proposals(author, inst, q)
    sizes = np.full(10, 10)

    capped = triage.IndividualCap(k=1, sigma_self=SIGMA).apply(p, _hist(100, sizes), rng)
    assert capped.wasted_effort == 0.0

    quota = triage.InstitutionalCap(m=5, sigma_inst=SIGMA).apply(p, _hist(100, sizes), rng)
    assert quota.wasted_effort == pytest.approx(n - np.count_nonzero(quota.submitted))


# ---------------------------------------------------------------------------
# the instruments
# ---------------------------------------------------------------------------


def test_individual_cap_respects_a_multi_round_window(cfg):
    """K proposals per W rounds, not per round. Restricting the instrument to a
    single round puts a false floor under how hard it can squeeze."""
    sim = SimConfig(sigma_eval=SIGMA, n_rounds=12, burn_in=4)
    loads = [
        replicate(cfg, sim, triage.IndividualCap(k=1, sigma_self=SIGMA, window=w),
                  n_seeds=3)["review_load"].mean
        for w in (1, 2, 4)
    ]
    assert loads[0] > loads[1] > loads[2], loads
    # A one-round cap of one cannot go below roughly one proposal per active
    # researcher; a multi-round window can.
    assert loads[2] < 0.5 * loads[0]


def test_cooling_off_accumulates_over_its_look_back_window(cfg):
    """The rule bars an applicant for a record accumulated over a period.

    Evaluating the trigger on a single round makes it fire only on applicants
    who submit many proposals at once, which is a different policy.
    """
    sim = SimConfig(sigma_eval=SIGMA, n_rounds=12, burn_in=4)
    loads = [
        replicate(cfg, sim,
                  triage.CoolingOff(min_strikes=3, bar_rounds=2, window=w),
                  n_seeds=3)["review_load"].mean
        for w in (1, 2, 3)
    ]
    assert loads[0] > loads[1] > loads[2], loads


def test_cooling_off_bars_nobody_when_the_threshold_cannot_be_met(cfg):
    sim = SimConfig(sigma_eval=SIGMA, n_rounds=8, burn_in=2)
    uncapped = replicate(cfg, sim, triage.NoCap(), n_seeds=3)["review_load"].mean
    mild = replicate(cfg, sim,
                     triage.CoolingOff(min_strikes=99, bar_rounds=1, window=2),
                     n_seeds=3)["review_load"].mean
    assert mild == pytest.approx(uncapped, rel=0.02)


def test_resubmission_limit_never_blocks_a_fresh_proposal():
    n = 200
    rng = np.random.default_rng(0)
    is_resub = np.zeros(n, dtype=bool)
    is_resub[:100] = True
    p = _proposals(np.arange(n), np.zeros(n, int), rng.normal(size=n), is_resub)
    h = _hist(n)
    h.proposal_resub_count = np.where(is_resub, 3, 0).astype(np.int32)

    res = triage.ResubmissionLimit(max_resubmissions=0).apply(p, h, rng)
    assert res.submitted[~is_resub].all(), "fresh proposals must never be blocked"
    assert not res.submitted[is_resub].any(), "over-limit resubmissions must be blocked"


def test_proportional_quota_scales_with_headcount_not_with_demand():
    """Two institutions of equal headcount get equal quota even when one
    happens to submit far more this round."""
    author = np.concatenate([np.repeat(np.arange(0, 20), 5),      # inst 0: 20 active
                             np.repeat(np.arange(20, 25), 5)])    # inst 1: 5 active
    inst = np.concatenate([np.zeros(100, int), np.ones(25, int)])
    rng = np.random.default_rng(0)
    p = _proposals(author, inst, rng.normal(size=author.size))
    sizes = np.array([20, 20])  # equal headcount despite unequal activity

    rule = triage.InstitutionalCap(per_researcher=0.5, sigma_inst=SIGMA)
    res = rule.apply(p, _hist(25, sizes), rng)
    per_inst = [int(res.submitted[inst == k].sum()) for k in (0, 1)]
    assert per_inst[0] == per_inst[1] == 10, per_inst


def test_random_thinning_keeps_the_requested_share():
    n = 1000
    rng = np.random.default_rng(0)
    p = _proposals(np.arange(n), np.zeros(n, int), rng.normal(size=n))
    res = triage.RandomThinning(keep_share=0.37).apply(p, _hist(n), rng)
    assert np.count_nonzero(res.submitted) == 370


def test_random_thinning_has_no_triage_value(cfg, sim):
    """The counterfactual must not select. Its funded quality should match a
    same-load draw made without reference to quality."""
    est = replicate(cfg, sim, triage.RandomThinning(keep_share=0.5), n_seeds=4)
    capped = replicate(cfg, sim, triage.InstitutionalCap(m=30, sigma_inst=SIGMA),
                       n_seeds=4)
    assert capped["mean_quality_funded"].mean > est["mean_quality_funded"].mean


# ---------------------------------------------------------------------------
# comparison machinery
# ---------------------------------------------------------------------------


def test_pairing_is_tighter_than_an_unpaired_contrast(cfg, sim):
    """Seeds fix the population, so differencing within seed removes the
    between-population variance that dominates the level."""
    a = triage.InstitutionalCap(m=30, sigma_inst=SIGMA)
    b = triage.RandomThinning(keep_share=0.5)
    n = 8
    paired = paired_diff(cfg, sim, a, b, n_seeds=n)
    ea = replicate(cfg, sim, a, n_seeds=n)["mean_quality_funded"]
    eb = replicate(cfg, sim, b, n_seeds=n)["mean_quality_funded"]
    unpaired_sem = float(np.hypot(ea.sem, eb.sem))
    assert paired.mean == pytest.approx(ea.mean - eb.mean, abs=1e-9)
    assert paired.sem < unpaired_sem


def test_tuner_finds_settings_for_both_search_directions(cfg, sim):
    """Cooling-off tightens as its parameter rises and every other instrument
    as it falls; a tuner assuming one direction returns nonsense for the rest.
    """
    from dm.sweep import Spec, tune

    target = 0.6 * replicate(cfg, sim, triage.NoCap(), n_seeds=3)["review_load"].mean
    specs = {
        "cooling off": Spec(
            lambda v: triage.CoolingOff(min_strikes=int(v[0]), bar_rounds=int(v[1])),
            grid=tuple((s, b) for s in (1, 2, 3, 4) for b in (1, 2, 4, 8)),
            tight_end="hi",
        ),
        "institutional": Spec(
            lambda m: triage.InstitutionalCap(m=max(1, int(m)), sigma_inst=SIGMA),
            lo=1, hi=150, integer=True,
        ),
    }
    for name, spec in specs.items():
        _, got, _ = tune(spec, target, cfg, sim)
        assert abs(got - target) / target < 0.15, f"{name}: {got:.0f} vs {target:.0f}"


def test_seeds_give_independent_populations(cfg, sim):
    a = run(cfg, sim, triage.NoCap(), seed=1).mean("mean_quality_funded")
    b = run(cfg, sim, triage.NoCap(), seed=2).mean("mean_quality_funded")
    c = run(cfg, sim, triage.NoCap(), seed=1).mean("mean_quality_funded")
    assert a == pytest.approx(c), "same seed must reproduce exactly"
    assert a != pytest.approx(b), "different seeds must differ"


def test_null_scenario_makes_the_unit_of_the_cap_irrelevant():
    """S0 carries no quality information in affiliation, so an institutional
    quota cannot beat the counterfactual by more than noise on concentration."""
    cfg = scenarios()["S0_null"]
    sim = SimConfig(sigma_eval=SIGMA, n_rounds=8, burn_in=2)
    quota = replicate(cfg, sim, triage.InstitutionalCap(m=30, sigma_inst=SIGMA),
                      n_seeds=4)
    assert quota["hhi_institution"].mean == pytest.approx(1 / 50, abs=0.01)


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


def test_calibrated_noise_reproduces_the_published_instability():
    """sigma_eval = 0.195 is the fitted value quoted in the write-up.

    It is fitted to one of three published bins and must reproduce the other
    two; the constant is hardcoded everywhere else, so this is what keeps it
    honest.
    """
    from dm.evaluation import GRAVES_PAYLINE, GRAVES_TARGET, sometimes_funded_share

    rng = np.random.default_rng(0)
    quality = rng.normal(size=4000)
    got = sometimes_funded_share(
        quality, SIGMA_EVAL, payline=GRAVES_PAYLINE, rng=rng, n_draws=200
    )
    assert got["sometimes"] == pytest.approx(GRAVES_TARGET["sometimes"], abs=0.03)
    assert got["always"] == pytest.approx(GRAVES_TARGET["always"], abs=0.03)
    assert got["never"] == pytest.approx(GRAVES_TARGET["never"], abs=0.03)


def test_calibration_search_recovers_a_known_noise_level():
    """The bisection must find the noise level that produced a target."""
    from dm.evaluation import (
        GRAVES_PAYLINE,
        calibrate_sigma_eval,
        sometimes_funded_share,
    )

    rng = np.random.default_rng(1)
    quality = rng.normal(size=3000)
    truth = 0.30
    target = sometimes_funded_share(
        quality, truth, GRAVES_PAYLINE, rng, n_draws=150
    )["sometimes"]
    found, _ = calibrate_sigma_eval(
        quality, GRAVES_PAYLINE, rng, target_sometimes=target, n_draws=150
    )
    assert found == pytest.approx(truth, abs=0.05), found
