"""Replication: run a configuration over many seeds and report uncertainty.

Every number that reaches the paper goes through here. Single-seed runs are for
debugging only -- the differences between mechanisms are often smaller than the
between-seed spread, and reporting a point estimate without that spread is how
simulation papers manufacture findings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dm.config import RunConfig
from dm.simulate import SimConfig, SimResult, run

# Metrics reported for every mechanism.
METRICS = (
    "intended",
    "submitted",
    "funded",
    "mean_quality_funded",
    "applicant_quality_premium",
    "volume_advantage",
    "volume_gini",
    "n_active",
    "n_winners",
    "hhi_institution",
    "wasted_effort",
    "review_load",
    "submitted_effort",
    "success_rate",
)


@dataclass
class Estimate:
    mean: float
    sem: float
    n: int
    # Spread across replicate worlds. Distinct from `sem`, which is the
    # precision of our estimate of the mean: `sd` is how much the outcome
    # itself varies between worlds, which is what a funder actually bears.
    sd: float = 0.0
    # Mean within-run round-to-round standard deviation. A mechanism can have a
    # good average portfolio and still deliver an unreliable one year to year.
    volatility: float = 0.0

    @property
    def ci95(self) -> tuple[float, float]:
        h = 1.96 * self.sem
        return (self.mean - h, self.mean + h)

    def __str__(self) -> str:
        return f"{self.mean:.3f}±{1.96 * self.sem:.3f}"


def replicate(
    run_cfg: RunConfig,
    sim_cfg: SimConfig,
    rule,
    n_seeds: int = 20,
) -> dict[str, Estimate]:
    """Run `n_seeds` independent populations and summarise each metric.

    The seed governs both the population draw and every stochastic step, so
    replicates are independent worlds rather than repeated draws in one world.
    """
    per_seed: dict[str, list[float]] = {m: [] for m in METRICS}
    # Within-run spread: how much the metric moves from round to round inside a
    # single world, averaged over worlds.
    per_seed_vol: dict[str, list[float]] = {m: [] for m in METRICS}
    for s in range(n_seeds):
        r: SimResult = run(run_cfg, sim_cfg, rule, seed=run_cfg.seed + 1000 * s)
        for m in METRICS:
            per_seed[m].append(r.mean(m))
            per_seed_vol[m].append(r.round_sd(m))

    out = {}
    for m, vals in per_seed.items():
        a = np.asarray(vals, dtype=float)
        ok = a[~np.isnan(a)]
        if ok.size == 0:
            out[m] = Estimate(mean=float("nan"), sem=0.0, n=0)
            continue
        vol = np.asarray(per_seed_vol[m], dtype=float)
        vol_ok = vol[~np.isnan(vol)]
        out[m] = Estimate(
            mean=float(ok.mean()),
            sem=float(ok.std(ddof=1) / np.sqrt(ok.size)) if ok.size > 1 else 0.0,
            n=ok.size,
            sd=float(ok.std(ddof=1)) if ok.size > 1 else 0.0,
            volatility=float(vol_ok.mean()) if vol_ok.size else float("nan"),
        )
    return out


def paired_diff(
    run_cfg: RunConfig,
    sim_cfg: SimConfig,
    rule_a,
    rule_b,
    metric: str = "mean_quality_funded",
    n_seeds: int = 16,
) -> Estimate:
    """Mean and SE of the per-seed difference between two rules.

    Seeds govern the population draw as well as the stochastic steps, so every
    rule sees the same sequence of worlds. Differencing within a seed removes
    the between-world variance, which dominates the unpaired interval: an
    unpaired comparison here reports the spread of institutional landscapes,
    not the precision of the contrast being made.
    """
    diffs = []
    for s in range(n_seeds):
        seed = run_cfg.seed + 1000 * s
        a = run(run_cfg, sim_cfg, rule_a, seed=seed).mean(metric)
        b = run(run_cfg, sim_cfg, rule_b, seed=seed).mean(metric)
        diffs.append(a - b)
    d = np.asarray(diffs, dtype=float)
    return Estimate(
        mean=float(d.mean()),
        sem=float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1 else 0.0,
        n=d.size,
        sd=float(d.std(ddof=1)) if d.size > 1 else 0.0,
    )


def welfare(est: dict[str, Estimate], payline: float | None = None) -> float:
    """Quality delivered per unit of applicant effort consumed.

    The grant budget is FIXED, so the number funded is the same under every
    mechanism and must be read off the run rather than inferred from the payline
    and the submission count. Inferring it (`submitted * payline`) silently
    rewards mechanisms that inflate volume and punishes those that shrink it:
    it credited the uncapped reference with twice the grants it awards and
    two-stage screening with less than half.
    """
    effort = est["submitted_effort"].mean + est["wasted_effort"].mean
    return est["mean_quality_funded"].mean * est["funded"].mean / max(effort, 1e-9)


def differs(a: Estimate, b: Estimate) -> bool:
    """Do two estimates differ at roughly 95%? Used to stop the write-up
    claiming a difference the replicates do not support."""
    se = np.hypot(a.sem, b.sem)
    return abs(a.mean - b.mean) > 1.96 * se if se > 0 else a.mean != b.mean
