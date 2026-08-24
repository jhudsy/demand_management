"""exp07 -- robustness of the mechanism ranking across the unanchorable parameters.

Several parameters here cannot be anchored to data: volume elasticity, the
effort offset psi, how noisy institutional triage is relative to funder review,
how persistently rejected projects come back. Rather than defend a guess, this
runs the entire capacity-matched comparison across the plausible range of all of
them and asks which conclusions hold everywhere.

Two designs, because they answer different questions:

  OAT   one parameter moved at a time from the base case (the default values in
        `Cell`). Attributes any change to a named cause.
  JOINT a Latin-hypercube sample of the whole space. Catches interactions the
        OAT design cannot see, and is what the survival rates are computed on.

Outputs `results/exp07_cells.csv` and prints the feasibility frontier, the
claim-survival table, and the reliability (spread) comparison.

    python experiments/exp07_robustness.py [--joint N] [--jobs N] [--quick]
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from dm.sweep import Cell, evaluate_cell

RESULTS = Path(__file__).resolve().parents[1] / "results"

# Every parameter that has no empirical anchor, with the range we are willing to
# defend as plausible. The OAT levels include the base-case value, and an
# assertion below holds them to exactly these endpoints, since the write-up
# quotes the ranges as the space actually explored.
RANGES: dict[str, tuple[float, float]] = {
    "volume_elasticity": (0.0, 1.5),
    "psi": (0.0, 1.0),
    "sigma_idea": (0.2, 1.0),
    "payline": (0.05, 0.30),
    "capacity_target": (0.25, 0.85),
    "triage_noise_mult": (1.0, 4.0),
    "resub_intensity": (0.3, 1.7),
}

OAT_LEVELS: dict[str, list] = {
    "scenario": [
        "S0_null", "S1_size_only", "S2_quality_only",
        "S3_size_and_quality", "S4_big_and_good", "S5_heavy_tail",
    ],
    "volume_elasticity": [0.0, 0.25, 0.5, 1.0, 1.5],
    "psi": [0.0, 0.5, 1.0],
    "sigma_idea": [0.2, 0.5, 1.0],
    "payline": [0.05, 0.15, 0.30],
    "capacity_target": [0.25, 0.4, 0.5, 0.7, 0.85],
    "triage_noise_mult": [1.0, 2.0, 4.0],
    "resub_intensity": [0.3, 1.0, 1.7],
}


for _p, _lv in OAT_LEVELS.items():
    if _p in RANGES:
        _lo, _hi = RANGES[_p]
        assert min(_lv) == _lo and max(_lv) == _hi, (
            f"OAT levels for {_p} must span exactly the range reported in the "
            f"write-up: got {min(_lv)}..{max(_lv)}, expected {_lo}..{_hi}"
        )


def design(n_joint: int, seed: int = 20260804) -> list[Cell]:
    """OAT cells plus a Latin-hypercube sample of the joint space."""
    base = Cell()
    cells = {base}

    for param, levels in OAT_LEVELS.items():
        for v in levels:
            cells.add(replace(base, **{param: v}))

    rng = np.random.default_rng(seed)
    names = list(RANGES)
    # Latin hypercube: one draw per stratum per dimension, independently
    # shuffled, so the sample covers each margin evenly without a full grid.
    cuts = np.stack([
        rng.permutation(n_joint) for _ in names
    ], axis=1)
    u = (cuts + rng.random((n_joint, len(names)))) / n_joint
    scen = OAT_LEVELS["scenario"]
    for i in range(n_joint):
        kw = {}
        for j, name in enumerate(names):
            lo, hi = RANGES[name]
            kw[name] = float(lo + u[i, j] * (hi - lo))
        kw["scenario"] = scen[rng.integers(len(scen))]
        cells.add(replace(base, **kw))

    return sorted(cells, key=lambda c: (c.scenario, c.volume_elasticity))


def is_oat(row, base: Cell) -> bool:
    fields = list(OAT_LEVELS)
    diffs = sum(1 for f in fields if row[f] != getattr(base, f))
    return diffs <= 1


# ---------------------------------------------------------------------------
# claims to test -- each is a statement the write-up currently makes
# ---------------------------------------------------------------------------


def claims(g: pd.DataFrame) -> tuple[dict[str, bool | None], dict[str, str]]:
    """Evaluate each qualitative claim within one cell.

    `g` holds every mechanism at one parameter point. A claim returns None when
    the cell cannot speak to it (e.g. the mechanisms it names are infeasible),
    so "no opinion" is never counted as support.

    Also returns, for each "X is the extreme" claim, which mechanism actually
    took the extreme -- knowing a claim fails 40% of the time is much less
    useful than knowing what beats it when it does.
    """
    f = g[g["feasible"] & (g["mechanism"] != "no cap (infeasible)")]
    if f.empty:
        return {}, {}
    m = f.set_index("mechanism")
    holder: dict[str, str] = {}

    # The two institutional variants are the same instrument under different
    # quota rules, so they are one family: a claim about "institutional caps"
    # must not be tested by pitting them against each other.
    INSTITUTIONAL = {"institutional cap", "institutional prop"}

    def best(col, family, minimise=False, key=None):
        """Is the extreme value of `col`, among feasible non-placebo
        mechanisms, held by a member of `family`?"""
        fam = {family} if isinstance(family, str) else family
        real = m.drop(index=["random (placebo)"], errors="ignore")
        if not (fam & set(real.index)) or len(real) < 2:
            return None
        s = real[col].dropna()
        # A metric that is undefined for the candidate cannot support a claim
        # about it. Volume advantage is undefined whenever an instrument leaves
        # no variation in submission counts, and counting that as a win would
        # credit the instrument for making the measure inapplicable.
        if not (fam & set(s.index)) or len(s) < 2:
            return None
        who = s.idxmin() if minimise else s.idxmax()
        if key is not None:
            holder[key] = who
        return bool(who in fam)

    def beats(a, b, col, higher=True):
        """Does mechanism `a` beat `b` on `col`?"""
        if a not in m.index or b not in m.index:
            return None
        d = m.loc[a, col] - m.loc[b, col]
        return bool(d > 0 if higher else d < 0)

    out: dict[str, bool | None] = {}

    def extreme(key, col, family, minimise=False):
        out[key] = best(col, family, minimise=minimise, key=key)

    extreme("individual caps minimise volume advantage",
            "volume_advantage", "individual cap", minimise=True)
    extreme("two-stage maximises portfolio quality",
            "mean_quality_funded", "two-stage EOI")
    extreme("two-stage minimises applicant selectivity",
            "applicant_quality_premium", "two-stage EOI", minimise=True)
    extreme("institutional caps minimise welfare",
            "welfare", INSTITUTIONAL, minimise=True)
    extreme("institutional caps deconcentrate most",
            "hhi_institution", INSTITUTIONAL, minimise=True)
    extreme("resubmission limits maximise applicant selectivity",
            "applicant_quality_premium", "resubmission limit")
    # H1, both readings. The original intuition was that institutional caps lose
    # good proposals that individual caps would keep; at the proposal level the
    # opposite held, because a quota can be reallocated within an institution
    # but not within a person.
    out["H1 proposal level: institutional > individual on quality"] = beats(
        "institutional cap", "individual cap", "mean_quality_funded"
    )
    out["H1 applicant level: individual > institutional on selectivity"] = beats(
        "individual cap", "institutional cap", "applicant_quality_premium"
    )

    # Every real mechanism beats the volume-matched random placebo: triage is
    # doing selection, not merely shrinking the pile.
    if "random (placebo)" in m.index:
        pl = m.loc["random (placebo)", "mean_quality_funded"]
        real = m.drop(index=["random (placebo)"])
        out["all mechanisms beat the random placebo"] = (
            bool((real["mean_quality_funded"] > pl).all()) if len(real) else None
        )

    # The capacity constraint costs portfolio quality: no feasible mechanism
    # matches the (infeasible) uncapped reference.
    # The capacity constraint costs portfolio quality. Two-stage is separated
    # out because it is the one instrument that can exceed the uncapped
    # reference: cheap entry enlarges the pool it screens from. exp10 rules out
    # the competing explanation (that screening twice reads on two independent
    # noise draws) -- that alone never beats the uncapped reference.
    ref = g[g["mechanism"] == "no cap (infeasible)"]
    if not ref.empty:
        rq = float(ref["mean_quality_funded"].iloc[0])
        real = m.drop(index=["random (placebo)"], errors="ignore")
        single = real.drop(index=["two-stage EOI"], errors="ignore")
        out["single-stage instruments all cost portfolio quality"] = (
            bool((single["mean_quality_funded"] < rq).all()) if len(single) else None
        )
        out["two-stage exceeds the uncapped reference on quality"] = (
            bool(m.loc["two-stage EOI", "mean_quality_funded"] > rq)
            if "two-stage EOI" in m.index else None
        )
    return out, holder


def main() -> None:
    ap = argparse.ArgumentParser()
    # 400 hypercube points plus the 24 distinct one-at-a-time cells give the
    # 424 settings the write-up reports, so the default reproduces the paper.
    ap.add_argument("--joint", type=int, default=400)
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.joint = 12

    cells = design(args.joint)
    jobs = args.jobs or None
    print(f"exp07 -- {len(cells)} parameter cells, 7 mechanisms each.")
    t0 = time.time()

    rows: list[dict] = []
    done = 0
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for res in ex.map(evaluate_cell, cells, chunksize=1):
            rows.extend(res)
            done += 1
            if done % 20 == 0 or done == len(cells):
                el = time.time() - t0
                print(f"  {done}/{len(cells)} cells  {el:.0f}s "
                      f"(eta {el / done * (len(cells) - done):.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "exp07_cells.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(df)} rows, {time.time() - t0:.0f}s)\n")

    report(df, args.joint)


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def report(df: pd.DataFrame, n_joint: int) -> None:
    base = Cell()
    real = df[df["mechanism"] != "no cap (infeasible)"]

    print("=" * 78)
    print("1. FEASIBILITY FRONTIER -- the tightest cut each instrument can deliver")
    print("=" * 78)
    print("min_load_share is the review load at the instrument's most severe")
    print("setting, as a share of the uncapped load. A funder needing to cut to")
    print("40% cannot use an instrument whose floor is 60%.\n")
    fr = (
        real.groupby("mechanism")["min_load_share"]
        .agg(["median", "min", "max"])
        .sort_values("median")
    )
    print(f"{'mechanism':22s} {'median':>8s} {'best':>8s} {'worst':>8s}")
    for name, r in fr.iterrows():
        print(f"{name:22s} {r['median']:8.2f} {r['min']:8.2f} {r['max']:8.2f}")

    print("\n  Two distinct ways an instrument can fail the funder:")
    print("    reaches -- can it squeeze this hard at all?")
    print("    lands   -- can it be tuned to sit AT the target, or only near it?")
    print("  Instruments set by a small integer (a cap of 2, a bar of 1 round)")
    print("  routinely reach far but land badly. That lumpiness is a property of")
    print("  the policy, not of the search.\n")
    tab = real.groupby("mechanism")[["reaches", "lands", "lumpy"]].mean()
    tab = tab.sort_values("lands", ascending=False)
    print(f"{'mechanism':22s} {'reaches':>8s} {'lands':>8s} {'integer-set':>12s}")
    for name, r in tab.iterrows():
        print(f"{name:22s} {r['reaches']:8.0%} {r['lands']:8.0%} "
              f"{'yes' if r['lumpy'] > 0.5 else 'no':>12s}")

    print("\n  reach, by how deep the required cut is (share of load retained):")
    b = pd.cut(real["capacity_target"], [0, 0.35, 0.5, 0.65, 1.0])
    piv = real.pivot_table(
        index="mechanism", columns=b, values="reaches", aggfunc="mean", observed=True
    )
    with pd.option_context("display.width", 200, "display.float_format", "{:.0%}".format):
        print(piv.to_string())

    print("\n" + "=" * 78)
    print("2. CLAIM SURVIVAL -- does each stated conclusion hold across the space?")
    print("=" * 78)
    rec: dict[str, list] = {}
    ctx: dict[str, list] = {}
    won: dict[str, list] = {}
    for _, g in df.groupby([f for f in OAT_LEVELS]):
        c, holder = claims(g)
        for k, v in c.items():
            if v is None:
                continue
            rec.setdefault(k, []).append(v)
            ctx.setdefault(k, []).append(g.iloc[0])
            if k in holder:
                won.setdefault(k, []).append(holder[k])
    print(f"{'claim':52s} {'holds':>7s} {'cells':>7s}")
    print("-" * 70)
    surv = {k: (np.mean(v), len(v)) for k, v in rec.items()}
    for k, (p, n) in sorted(surv.items(), key=lambda kv: -kv[1][0]):
        print(f"{k:52s} {p:6.0%} {n:7d}")

    print("\n  where the shaky claims break down:")
    for k, (p, n) in surv.items():
        if p >= 0.95 or p == 0:
            continue
        fails = pd.DataFrame([c for c, v in zip(ctx[k], rec[k]) if not v])
        holds = pd.DataFrame([c for c, v in zip(ctx[k], rec[k]) if v])
        drivers = []
        for f in RANGES:
            if fails.empty or holds.empty:
                continue
            d = fails[f].mean() - holds[f].mean()
            spread = df[f].std()
            if spread > 0 and abs(d) / spread > 0.5:
                drivers.append(f"{f} {'higher' if d > 0 else 'lower'} ({d:+.2f})")
        print(f"    {k} ({p:.0%})")
        print(f"      fails when: {', '.join(drivers) if drivers else 'no single parameter dominates'}")
        if k in won:
            beat = pd.Series([w for w, v in zip(won[k], rec[k]) if not v])
            if not beat.empty:
                shares = beat.value_counts(normalize=True)
                took = ", ".join(f"{n} {s:.0%}" for n, s in shares.head(3).items())
                print(f"      taken instead by: {took}")

    print("\n" + "=" * 78)
    print("3. RELIABILITY -- spread, not just level")
    print("=" * 78)
    print("q_sd_within is the round-to-round standard deviation of funded-portfolio")
    print("quality inside one world: what a funder actually experiences year to")
    print("year. A good mean with a large spread is not a good instrument.\n")
    rel = (
        real[real["feasible"]]
        .groupby("mechanism")[["mean_quality_funded", "q_sd_within", "q_sd_between"]]
        .median()
        .sort_values("q_sd_within")
    )
    rel["cv_within"] = rel["q_sd_within"] / rel["mean_quality_funded"]
    print(f"{'mechanism':22s} {'Qfund':>8s} {'sd(round)':>10s} {'sd(world)':>10s} {'cv':>7s}")
    for name, r in rel.iterrows():
        print(f"{name:22s} {r['mean_quality_funded']:8.3f} {r['q_sd_within']:10.4f} "
              f"{r['q_sd_between']:10.4f} {r['cv_within']:7.3f}")

    print("\n" + "=" * 78)
    print("4. ONE-AT-A-TIME SENSITIVITY -- what actually moves the answer")
    print("=" * 78)
    print("Range of median Qfund across the levels of each parameter, holding the")
    print("rest at the base case. Large range = the conclusion depends on a number")
    print("we cannot anchor.\n")
    oat = df[df.apply(lambda r: is_oat(r, base), axis=1)]
    oat = oat[oat["feasible"] & (oat["mechanism"] != "no cap (infeasible)")]
    allc = df[df.apply(lambda r: is_oat(r, base), axis=1)]
    allc = allc[allc["mechanism"] != "no cap (infeasible)"]
    print(f"{'parameter':22s} {'Qfund':>9s} {'volAdv':>9s} {'welfare':>9s} {'feasible%':>12s}")
    for f in OAT_LEVELS:
        sub = oat[oat.apply(
            lambda r: all(r[o] == getattr(base, o) for o in OAT_LEVELS if o != f),
            axis=1,
        )]
        if sub[f].nunique() < 2:
            continue

        def rng_(col: str, sub=sub, f=f) -> float:
            """Spread of the median of `col` across this parameter's levels."""
            med = sub.groupby(f)[col].median()
            return float(med.max() - med.min())

        fe = allc.groupby(f)["feasible"].mean()
        print(f"{f:22s} {rng_('mean_quality_funded'):9.3f} "
              f"{rng_('volume_advantage'):9.3f} {rng_('welfare'):9.3f} "
              f"{fe.min():8.0%}-{fe.max():.0%}")


if __name__ == "__main__":
    main()
