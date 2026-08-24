"""exp10 -- where the two-stage advantage comes from, and what its floor rests on.

Both results appear in the paper and neither had a script behind it.

Part 1 tests the competing explanation for two-stage screening funding a better
portfolio than reviewing everything. Either the scheme enlarges the field it
selects from, because cheap entry draws more entries, or it reads each proposal
twice on independent noise. The second is isolated by stripping the first away:
the outline costs nothing to write, nothing to screen, and entry is priced at a
full proposal, so no volume response is possible and only the extra independent
reading remains.

Part 2 varies what an outline costs to screen. A two-stage scheme must read every
outline it attracts, so its review load cannot fall below the cost of doing so.
That floor is proportional to the screening cost, which makes the size of the
floor a parameter choice but its existence a property of the design.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from dm import triage
from dm.runner import replicate
from dm.scenarios import scenarios
from dm.simulate import SimConfig
from dm.sweep import SIGMA_EVAL, Spec, tune

SEEDS = 12


def pre_screen(sigma: float, invite: float = 0.5) -> triage.TwoStage:
    """Two-stage with every advantage except the second reading removed."""
    return triage.TwoStage(
        invite_ratio=invite,
        eoi_cost=0.0,
        eoi_review_cost=0.0,
        sigma_eoi=sigma,
        # Entry priced at a full proposal, breaking the usual link to
        # eoi_cost, so no volume response is possible and only the extra
        # independent reading is left to measure.
        entry_cost=1.0,
    )


def main() -> None:
    cfg = scenarios()["S4_big_and_good"]

    print("=" * 72)
    print("1. IS THE TWO-STAGE GAIN A SECOND INDEPENDENT READING?")
    print("=" * 72)
    print("Outline free to write and free to screen, entry at full price,")
    print("elasticity zero. Nothing left but the extra reading.\n")

    sim0 = SimConfig(sigma_eval=SIGMA_EVAL, n_rounds=20, burn_in=5,
                     volume_elasticity=0.0)
    ref = replicate(cfg, sim0, triage.NoCap(), n_seeds=SEEDS)
    print(f"{'arm':34s} {'load':>6s} {'Qfund':>7s}")
    print(f"{'no cap':34s} {ref['review_load'].mean:6.0f} "
          f"{ref['mean_quality_funded'].mean:7.3f}")
    ballot = replicate(cfg, sim0, triage.RandomThinning(keep_share=0.5), n_seeds=SEEDS)
    print(f"{'ballot at the same load':34s} {ballot['review_load'].mean:6.0f} "
          f"{ballot['mean_quality_funded'].mean:7.3f}")
    for s in (SIGMA_EVAL, 0.4, 0.9, 1.5, 3.0):
        e = replicate(cfg, sim0, pre_screen(s), n_seeds=SEEDS)
        print(f"{'pre-screen, sigma=' + format(s, '.3g'):34s} "
              f"{e['review_load'].mean:6.0f} {e['mean_quality_funded'].mean:7.3f}")
    print("\n  A second reading beats a ballot at the same load, which is ordinary")
    print("  triage value. If it never exceeds the uncapped reference, the")
    print("  advantage in the headline table must come from pool inflation.")

    print("\n" + "=" * 72)
    print("2. WHAT THE TWO-STAGE LOAD FLOOR RESTS ON")
    print("=" * 72)
    sim = SimConfig(sigma_eval=SIGMA_EVAL, n_rounds=20, burn_in=5)
    uncapped = replicate(cfg, sim, triage.NoCap(), n_seeds=SEEDS)["review_load"].mean
    target = 0.5 * uncapped
    print(f"Uncapped load {uncapped:.0f}; target {target:.0f}.\n")
    print(f"{'screening cost':>15s} {'floor share':>12s} {'tuned v':>8s} "
          f"{'load':>7s} {'Qfund':>7s}")
    for c in (0.05, 0.10, 0.20, 0.30, 0.40):
        floor = replicate(
            cfg, sim, triage.TwoStage(invite_ratio=0.02, eoi_review_cost=c),
            n_seeds=SEEDS,
        )["review_load"].mean
        spec = Spec(
            lambda r, c=c: triage.TwoStage(
                invite_ratio=float(np.clip(r, 0.02, 1.0)), eoi_review_cost=c
            ),
            lo=0.02, hi=1.0,
        )
        rule, _, _ = tune(spec, target, cfg, sim)
        e = replicate(cfg, sim, rule, n_seeds=SEEDS)
        print(f"{c:15.2f} {floor / uncapped:12.3f} {rule.invite_ratio:8.3f} "
              f"{e['review_load'].mean:7.0f} {e['mean_quality_funded'].mean:7.3f}")
    print("\n  The floor scales with the screening cost and never reaches zero.")


if __name__ == "__main__":
    main()
