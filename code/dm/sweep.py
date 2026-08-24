"""Parameter scan: does the ranking of mechanisms survive the parameters we
cannot anchor?

Several parameters in this model have no empirical anchor -- volume elasticity,
the effort offset, how noisy institutional triage is relative to funder review.
The honest response is not to guess better values but to run the whole
capacity-matched comparison across the plausible range of all of them and report
in what share of settings each conclusion holds. The write-up calls a claim
stable when that share is at least 90%; nothing here assumes a claim must hold
everywhere to be worth reporting, only that the share be stated.

Two things come out of this that a point estimate cannot give:

1. **A feasibility frontier.** Not every instrument can deliver every capacity
   cut. `min_load_share` records the tightest load each mechanism can reach at
   the most severe setting of its parameter. A funder that needs a 70% cut has a
   shorter menu than one that needs 30%, and nothing in the literature says so.
2. **Claim survival.** Each qualitative claim in the write-up is checked in every
   cell, so it can be reported as "holds in N of M worlds" rather than asserted
   from one parameter set.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from dm import triage
from dm.evaluation import SIGMA_EVAL
from dm.runner import replicate, welfare
from dm.scenarios import scenarios
from dm.simulate import SimConfig

TUNE_SEEDS = 3
FINAL_SEEDS = 12
TOLERANCE = 0.08  # a mechanism counts as hitting capacity within 8%


# ---------------------------------------------------------------------------
# the parameter point
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One world.

    Most of these are parameters with no empirical anchor, which is the reason
    for sweeping at all. Three are not: `payline` and `capacity_target` are
    choices a funder makes rather than unknowns, and `scenario` selects a
    structural case. They are swept because a conclusion should not depend on
    which funder is being described either.
    """

    scenario: str = "S4_big_and_good"
    volume_elasticity: float = 0.5
    psi: float = 0.0
    sigma_idea: float = 0.5
    payline: float = 0.15
    capacity_target: float = 0.5  # share of the uncapped review load retained
    triage_noise_mult: float = 1.0  # triage sigma as a multiple of eval sigma
    resub_intensity: float = 1.0  # multiplies both resubmission probabilities

    def sim(self) -> SimConfig:
        base = SimConfig()
        return replace(
            base,
            sigma_eval=SIGMA_EVAL,
            n_rounds=20,
            burn_in=5,
            volume_elasticity=self.volume_elasticity,
            psi=self.psi,
            sigma_idea=self.sigma_idea,
            payline=self.payline,
            p_resubmit_base=min(0.95, base.p_resubmit_base * self.resub_intensity),
            p_resubmit_nearmiss=min(
                0.98, base.p_resubmit_nearmiss * self.resub_intensity
            ),
        )

    @property
    def sigma_triage(self) -> float:
        return SIGMA_EVAL * self.triage_noise_mult


# ---------------------------------------------------------------------------
# the mechanisms, each with the range of its control parameter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Spec:
    """How a mechanism's severity is controlled.

    Either a continuous range searched by bisection, or an explicit grid of
    settings searched exhaustively. The distinction is not cosmetic: several
    instruments are controlled by a small integer (a cap of 2 proposals, a bar
    of 1 round) and therefore cannot be tuned to an arbitrary capacity target at
    all. That lumpiness is a real property of the policy instrument, not an
    artefact of the search, and is reported separately from reach.
    """

    factory: object
    grid: tuple | None = None
    lo: float = 0.0
    hi: float = 1.0
    integer: bool = False
    # Which end of the range squeezes hardest. Cooling-off bites harder as its
    # parameter RISES; every other instrument as it falls. A tuner that assumes
    # one direction for all of them silently returns garbage for the rest.
    tight_end: str = "lo"
    # True when severity is set by a small integer the funder cannot subdivide.
    lumpy: bool = False


def candidates(cell: Cell) -> dict[str, Spec]:
    st = cell.sigma_triage

    def cooling(v):
        strikes, bar = v
        return triage.CoolingOff(bar_rounds=int(bar), min_strikes=int(strikes))

    # Cooling-off has two levers in the real EPSRC rule -- how many strikes
    # trigger it and how long the bar lasts -- so both are searched. Pinning one
    # of them makes the instrument look unable to hit targets it can hit.
    cool_grid = tuple(
        (s, b) for s in (1, 2, 3, 4, 5) for b in (1, 2, 3, 4, 6, 8)
    )

    def individual(v):
        k, w = v
        return triage.IndividualCap(k=int(k), sigma_self=st, window=int(w))

    # K proposals per W rounds. Restricting this to W=1 puts a false floor under
    # the instrument: it would then be unable to admit fewer than roughly one
    # proposal per active researcher per round, however tight K is set.
    indiv_grid = tuple((k, w) for w in (1, 2, 3, 4, 6) for k in (1, 2, 3, 4, 6))

    return {
        "individual cap": Spec(individual, grid=indiv_grid, lumpy=True),
        "institutional cap": Spec(
            lambda m: triage.InstitutionalCap(m=max(1, int(m)), sigma_inst=st),
            lo=1, hi=150, integer=True,
        ),
        "institutional prop": Spec(
            lambda r: triage.InstitutionalCap(per_researcher=max(0.02, r), sigma_inst=st),
            lo=0.02, hi=3.0,
        ),
        "cooling off": Spec(cooling, grid=cool_grid, tight_end="hi", lumpy=True),
        "resubmission limit": Spec(
            lambda m: triage.ResubmissionLimit(max_resubmissions=max(0, int(m))),
            grid=tuple(range(0, 6)), lumpy=True,
        ),
        "two-stage EOI": Spec(
            lambda r: triage.TwoStage(invite_ratio=float(np.clip(r, 0.02, 1.0))),
            lo=0.02, hi=1.0,
        ),
        "random (placebo)": Spec(
            lambda s: triage.RandomThinning(keep_share=float(np.clip(s, 0.02, 1.0))),
            lo=0.02, hi=1.0,
        ),
    }


# ---------------------------------------------------------------------------
# capacity tuning
# ---------------------------------------------------------------------------


def _load(rule, cfg, sim, seeds=TUNE_SEEDS) -> float:
    return replicate(cfg, sim, rule, n_seeds=seeds)["review_load"].mean


def tune(spec: Spec, target, cfg, sim, iters=12):
    """Find the setting whose review load is closest to `target`.

    Returns (rule, achieved_load, min_load). `min_load` is the smallest load the
    instrument can produce anywhere in its range -- the hardest squeeze it can
    deliver at all. That is a different question from whether it can land ON the
    target, and the two are reported separately.
    """
    if spec.grid is not None:
        best, best_load, best_err = None, None, float("inf")
        min_load = float("inf")
        for v in spec.grid:
            got = _load(spec.factory(v), cfg, sim)
            min_load = min(min_load, got)
            if abs(got - target) < best_err:
                best, best_load, best_err = spec.factory(v), got, abs(got - target)
        return best, best_load, min_load

    tight_val = spec.lo if spec.tight_end == "lo" else spec.hi
    min_load = _load(spec.factory(tight_val), cfg, sim)

    # Direction-aware bisection on a monotone-in-load parameter.
    increasing = spec.tight_end == "lo"  # load rises with the parameter
    best, best_load, best_err = None, None, float("inf")
    a, b = spec.lo, spec.hi
    for _ in range(iters):
        mid = (a + b) / 2
        val = int(round(mid)) if spec.integer else mid
        rule = spec.factory(val)
        got = _load(rule, cfg, sim)
        if abs(got - target) < best_err:
            best, best_load, best_err = rule, got, abs(got - target)
        if (got > target) == increasing:
            b = mid
        else:
            a = mid
        if spec.integer and abs(b - a) < 1:
            break
    return best, best_load, min_load


# ---------------------------------------------------------------------------
# one cell
# ---------------------------------------------------------------------------

METRIC_KEYS = (
    "mean_quality_funded",
    "applicant_quality_premium",
    "volume_advantage",
    "volume_gini",
    "n_active",
    "n_winners",
    "hhi_institution",
    "success_rate",
    "review_load",
    "submitted",
    "intended",
    "funded",
    "wasted_effort",
    "submitted_effort",
)


def evaluate_cell(cell: Cell) -> list[dict]:
    """Run the full capacity-matched comparison at one parameter point.

    Returns one row per mechanism, plus a row for the infeasible `no cap`
    reference that shows what the capacity constraint itself costs.
    """
    cfg = scenarios()[cell.scenario]
    sim = cell.sim()

    ref = replicate(cfg, sim, triage.NoCap(), n_seeds=FINAL_SEEDS)
    ref_load = ref["review_load"].mean
    target = cell.capacity_target * ref_load

    rows: list[dict] = [
        _row(cell, "no cap (infeasible)", ref, sim, ref_load, ref_load,
             reaches=False, lands=False, lumpy=False,
             min_load_share=1.0, ref_load=ref_load)
    ]

    for name, spec in candidates(cell).items():
        rule, _tuned_load, min_load = tune(spec, target, cfg, sim)
        est = replicate(cfg, sim, rule, n_seeds=FINAL_SEEDS)
        # Judge the achieved load on the same replicate the metrics come from.
        # The tuner searches on few seeds for speed, and using its noisier
        # estimate here would report a landing rate for a load nothing else in
        # the row was measured at.
        got = est["review_load"].mean
        rows.append(
            _row(
                cell, name, est, sim, got, target,
                # Two distinct failure modes, kept apart on purpose:
                #   reaches -- can squeeze at least this hard at all
                #   lands   -- can hit the target within tolerance
                # An instrument set by a small integer often reaches far but
                # lands badly, and calling that "infeasible" hides the reason.
                reaches=min_load <= target * (1 + TOLERANCE),
                lands=abs(got - target) / max(target, 1e-9) < TOLERANCE,
                lumpy=spec.lumpy,
                min_load_share=min_load / max(ref_load, 1e-9),
                ref_load=ref_load,
            )
        )
    return rows


def _row(cell, name, est, sim, got, target, *, reaches, lands, lumpy,
         min_load_share, ref_load):
    q = est["mean_quality_funded"]
    row = {
        **asdict(cell),
        "mechanism": name,
        "reaches": reaches,
        "lands": lands,
        "lumpy": lumpy,
        # Kept for the comparison tables: a mechanism only enters a
        # capacity-matched comparison if it actually sits at the capacity.
        "feasible": bool(reaches and lands),
        "achieved_load": got,
        "target_load": target,
        "ref_load": ref_load,
        # The tightest cut this instrument can deliver, as a share of the
        # uncapped load. Above the funder's target means it is not on the menu.
        "min_load_share": min_load_share,
        "welfare": welfare(est, sim.payline),
        # Spread, not just level: how much the funded portfolio varies between
        # worlds and from round to round within one.
        "q_sd_between": q.sd,
        "q_sd_within": q.volatility,
    }
    for k in METRIC_KEYS:
        row[k] = est[k].mean
        row[k + "_sem"] = est[k].sem
    return row
