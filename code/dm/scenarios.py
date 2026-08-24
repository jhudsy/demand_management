"""Institutional scenarios — the cases to investigate.

These are to this paper what cases (a)-(g) are to Roebber & Schultz: worlds that
differ in one structural assumption at a time, so any result can be attributed to
a named cause.

Institutions differ in exactly two ways: how many researchers they have, and how
good those researchers are on average. Nothing else about institutions is
modelled -- see `config.py`.
"""

from __future__ import annotations

from dataclasses import replace

from dm.config import InstitutionConfig, RunConfig

_BASE = InstitutionConfig(n_institutions=50, n_researchers=2000)


def scenarios() -> dict[str, RunConfig]:
    s: dict[str, InstitutionConfig] = {}

    # S0 -- the null. Equal sizes, quality independent of affiliation. This is
    # the assumption made by the anchor paper's ABM, and the case in which the
    # unit of the cap must make no difference. If the effect does not vanish
    # here, the model is wrong.
    s["S0_null"] = _BASE

    # S1 -- size only. Institutions differ in headcount but not in quality.
    # Isolates the pure arithmetic of a flat quota over unequal group sizes.
    s["S1_size_only"] = replace(_BASE, size_dist="lognormal", size_spread=0.8)

    # S2 -- quality only. Equal headcount; good researchers concentrate.
    # Isolates the information loss from a quota that binds on a group whose
    # members are unevenly good.
    s["S2_quality_only"] = replace(_BASE, rho_quality=0.5)

    # S3 -- both, uncorrelated. Large institutions are not systematically better.
    s["S3_size_and_quality"] = replace(
        _BASE, size_dist="lognormal", size_spread=0.8, rho_quality=0.5
    )

    # S4 -- both, correlated: big institutions are also better. The usual
    # real-world claim, and the case where a flat quota is most contentious.
    s["S4_big_and_good"] = replace(
        _BASE,
        size_dist="lognormal",
        size_spread=0.8,
        rho_quality=0.5,
        corr_size_quality=0.6,
    )

    # S5 -- heavy-tailed sizes. A few very large institutions, many small ones:
    # what a flat per-institution quota is most redistributive against.
    s["S5_heavy_tail"] = replace(
        _BASE,
        size_dist="powerlaw",
        size_spread=1.2,
        rho_quality=0.5,
        corr_size_quality=0.5,
    )

    return {
        name: RunConfig(institutions=cfg, label=name, seed=20260804)
        for name, cfg in s.items()
    }
