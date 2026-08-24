"""exp08 -- the headline capacity-matched table, with the confounds controlled.

Three changes over exp06, each answering a specific objection to that table.

1. **A load-matched placebo per instrument, not one shared placebo.** Instruments
   set by small integers cannot land exactly on a capacity target, so the
   headline loads differ by several per cent. Portfolio quality is sensitive
   enough to pool size that this matters: random thinning moves 0.05 in Qfund
   across a +/-4% band. Tuning a separate placebo to each instrument's *achieved*
   load makes the reported triage value immune to the mismatch, because both
   arms then sit at the same load by construction.

2. **The populations behind the applicant premium are reported.** appPrem is a
   difference between two groups the instrument itself defines: researchers who
   submitted, and researchers who won. Both change size dramatically between
   instruments (610 to 1877 submitters in early runs), so the premium alone
   cannot be read as selection accuracy. `n_active`, `n_winners` and grants per
   winner are shown next to it.

3. **Concentration is reported against benchmarks.** An HHI is uninterpretable
   on its own when institutions differ in size. Three reference points are
   computed: the HHI of the institution size distribution, of an equal split,
   and of a perfect merit allocation that funds the 450 best researchers
   outright.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from dm import population, triage
from dm.runner import paired_diff, replicate, welfare
from dm.scenarios import scenarios
from dm.simulate import SimConfig
from dm.sweep import Cell, candidates, tune

SEEDS = 16


def placebo_rule(load: float, cfg, sim):
    """Tune a random-thinning arm to a specific review load."""
    lo, hi = 0.02, 1.0
    best, best_err = None, float("inf")
    for _ in range(14):
        mid = (lo + hi) / 2
        rule = triage.RandomThinning(keep_share=mid)
        got = replicate(cfg, sim, rule, n_seeds=4)["review_load"].mean
        if abs(got - load) < best_err:
            best, best_err = rule, abs(got - load)
        if got > load:
            hi = mid
        else:
            lo = mid
    return best


def benchmarks(cfg, sim) -> dict[str, float]:
    """Reference points for the institutional concentration index.

    Averaged over the same populations the comparison uses. Institution sizes
    are redrawn per seed, so a benchmark computed from a single draw is not
    comparable with a 16-seed mean.
    """
    size_hhi, merit_hhi = [], []
    for s in range(SEEDS):
        rng = np.random.default_rng(cfg.seed + 1000 * s)
        inst, res = population.build(cfg, rng)
        shares = inst.size / inst.size.sum()
        size_hhi.append(float((shares**2).sum()))
        budget = int(round(sim.payline * res.n * sim.proposals_per_researcher))
        top = np.argsort(-res.quality)[:budget]
        w = np.bincount(res.institution[top], minlength=inst.n).astype(float)
        merit_hhi.append(float(((w / w.sum()) ** 2).sum()))
    return {
        "institution size distribution": float(np.mean(size_hhi)),
        "equal split across institutions": 1.0 / cfg.institutions.n_institutions,
        "perfect merit (fund the 450 best researchers)": float(np.mean(merit_hhi)),
    }


def describe(rule) -> str:
    """The parameter setting the tuner chose, so the table is reproducible."""
    n = getattr(rule, "name", "")
    if n == "individual_cap":
        return f"K={rule.k}, W={rule.window}"
    if n == "institutional_cap":
        return (f"r={rule.per_researcher:.2f}/head" if rule.per_researcher
                else f"M={rule.m}")
    if n == "cooling_off":
        return f"S={rule.min_strikes}, B={rule.bar_rounds}"
    if n == "resubmission_limit":
        return f"R={rule.max_resubmissions}"
    if n == "two_stage":
        return f"v={rule.invite_ratio:.3f}"
    return "---"


def main() -> None:
    cell = Cell()
    cfg = scenarios()[cell.scenario]
    sim = cell.sim()

    ref = replicate(cfg, sim, triage.NoCap(), n_seeds=SEEDS)
    ref_load = ref["review_load"].mean
    target = 0.5 * ref_load

    print(f"Capacity-matched comparison, scenario {cell.scenario}, {SEEDS} seeds.")
    print(f"Uncapped load {ref_load:.0f}; target {target:.0f}; budget fixed at "
          f"{ref['funded'].mean:.0f} grants.\n")

    rows: list[tuple[str, dict, dict | None]] = [("no cap (infeasible)", ref, None)]
    tuned: dict[str, str] = {}
    for name, spec in candidates(cell).items():
        if name == "random (placebo)":
            continue
        rule, got, _ = tune(spec, target, cfg, sim)
        tuned[name] = describe(rule)
        est = replicate(cfg, sim, rule, n_seeds=SEEDS)
        pl_rule = placebo_rule(est["review_load"].mean, cfg, sim)
        # Paired within seed: both arms see the same population draws.
        tv = paired_diff(cfg, sim, rule, pl_rule, n_seeds=SEEDS)
        rows.append((name, est, tv))

    def setting_of(name, _e):
        return tuned.get(name, "---")

    hdr = (f"{'instrument':20s} {'setting':>14s} {'load':>6s} {'Qfund':>14s} "
           f"{'triage':>15s} {'appPrem':>8s} {'active':>7s} {'winners':>8s} "
           f"{'g/win':>6s} {'volAdv':>7s} {'vGini':>6s} {'HHI':>7s} {'Q/E':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for name, e, pl in rows:
        va = e["volume_advantage"].mean
        va_s = "  n/a" if np.isnan(va) else f"{va:5.3f}"
        q = e["mean_quality_funded"]
        tv = ("            --" if pl is None
              else f"{pl.mean:+.3f}+/-{1.96 * pl.sem:.3f}")
        gpw = e["funded"].mean / max(e["n_winners"].mean, 1e-9)
        print(f"{name:20s} {setting_of(name, e):>14s} {e['review_load'].mean:6.0f} "
              f"{q.mean:.3f}+/-{1.96 * q.sem:.3f} {tv:>15s} "
              f"{e['applicant_quality_premium'].mean:8.3f} "
              f"{e['n_active'].mean:7.0f} {e['n_winners'].mean:8.0f} "
              f"{gpw:6.2f} {va_s:>7s} {e['volume_gini'].mean:6.3f} "
              f"{e['hhi_institution'].mean:7.4f} {welfare(e):6.3f}")

    print("\n  volAdv is n/a where the instrument leaves no variation in")
    print("  submission counts, which makes the ratio undefined rather than 1.")
    print("  vGini is the Gini of submission counts: 0 means everyone submits")
    print("  equally often, which is how a tight cap removes volume advantage.")
    print("  g/win is grants per distinct winner: repeat winning at the person level.")

    print("\n  Concentration benchmarks (HHI):")
    for k, v in benchmarks(cfg, sim).items():
        print(f"    {k:20s} {v:.4f}")


if __name__ == "__main__":
    main()
