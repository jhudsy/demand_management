"""Institutions and researchers.

This is the layer that does not exist in any prior model in this area. The
anchor paper's ABM assigns affiliation *uniformly at random and independently of
quality*, and uses it only as a reporting label — which is precisely the null
case in which institution-level demand management has no differential effect.
Here institutions differ in size and in average member quality (see `config.py`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dm.config import InstitutionConfig, RunConfig


@dataclass
class Institutions:
    """Institution-level state. Index k identifies an institution."""

    size: np.ndarray  # number of researchers
    quality: np.ndarray  # institution effect on member latent quality

    @property
    def n(self) -> int:
        return self.size.size


@dataclass
class Researchers:
    """Researcher-level state. Index i identifies a researcher."""

    institution: np.ndarray  # index into Institutions
    quality: np.ndarray  # latent research quality Q, ~N(0,1) across population

    @property
    def n(self) -> int:
        return self.quality.size


def _draw_sizes(cfg: InstitutionConfig, rng: np.random.Generator) -> np.ndarray:
    """Institution sizes summing (approximately) to n_researchers."""
    k, total = cfg.n_institutions, cfg.n_researchers

    if cfg.size_dist == "equal":
        raw = np.ones(k)
    elif cfg.size_dist == "lognormal":
        raw = rng.lognormal(mean=0.0, sigma=cfg.size_spread, size=k)
    elif cfg.size_dist == "powerlaw":
        # Pareto with shape `size_spread`; smaller shape = heavier tail.
        raw = rng.pareto(a=max(cfg.size_spread, 0.1), size=k) + 1.0
    else:
        raise ValueError(f"unsupported size_dist {cfg.size_dist!r}")

    # Scale to the researcher budget, enforce a floor, then repair the total so
    # the population size is exactly as configured (comparability across
    # scenarios depends on this).
    sizes = np.maximum(np.round(raw / raw.sum() * total), cfg.min_size).astype(int)
    while sizes.sum() != total:
        diff = total - sizes.sum()
        if diff > 0:
            sizes[rng.integers(0, k)] += min(diff, max(1, diff // k + 1))
        else:
            shrinkable = np.flatnonzero(sizes > cfg.min_size)
            if shrinkable.size == 0:
                raise ValueError(
                    "min_size too large for n_researchers / n_institutions"
                )
            sizes[rng.choice(shrinkable)] -= 1
    return sizes


def _correlated_effect(
    log_size: np.ndarray, sigma: float, corr: float, rng: np.random.Generator
) -> np.ndarray:
    """A zero-mean institution-level effect with SD `sigma`, correlated `corr`
    with log institution size."""
    if sigma == 0.0:
        return np.zeros_like(log_size)
    z = (log_size - log_size.mean()) / (log_size.std() or 1.0)
    noise = rng.normal(size=log_size.size)
    noise = (noise - noise.mean()) / (noise.std() or 1.0)
    combined = corr * z + np.sqrt(max(1.0 - corr**2, 0.0)) * noise
    return sigma * combined


def build(cfg: RunConfig, rng: np.random.Generator) -> tuple[Institutions, Researchers]:
    cfg.validate()
    icfg = cfg.institutions

    sizes = _draw_sizes(icfg, rng)
    log_size = np.log(sizes.astype(float))

    inst_quality = _correlated_effect(
        log_size, icfg.rho_quality, icfg.corr_size_quality, rng
    )
    institutions = Institutions(size=sizes, quality=inst_quality)

    # Researchers.
    membership = np.repeat(np.arange(icfg.n_institutions), sizes)
    n = membership.size

    # Latent quality = institution effect + individual deviation, with total
    # variance held at 1 regardless of the split. rho_quality is therefore
    # exactly corr(Q_i, institution effect).
    within_sd = np.sqrt(max(1.0 - icfg.rho_quality**2, 0.0))
    quality = inst_quality[membership] + rng.normal(0.0, within_sd, size=n)

    researchers = Researchers(institution=membership, quality=quality)
    return institutions, researchers


def summarise(inst: Institutions, res: Researchers) -> dict[str, float]:
    """Diagnostics used by the scenario report and the tests."""
    mean_q = np.array(
        [res.quality[res.institution == k].mean() for k in range(inst.n)]
    )
    return {
        "n_researchers": float(res.n),
        "n_institutions": float(inst.n),
        "size_min": float(inst.size.min()),
        "size_max": float(inst.size.max()),
        "size_gini": _gini(inst.size.astype(float)),
        "quality_sd_total": float(res.quality.std()),
        # Share of quality variance sitting between institutions -- the proper
        # size-weighted between-group sum of squares over the total, which
        # equals rho_quality^2 in expectation. (A naive np.var over the
        # institution means is wrong under skewed sizes and can exceed 1.)
        "quality_var_between_share": _between_share(inst.size, mean_q, res.quality),
        "inst_quality_sd": float(inst.quality.std()),
        "corr_size_meanq": float(
            np.corrcoef(np.log(inst.size.astype(float)), mean_q)[0, 1]
        )
        if inst.n > 1 and mean_q.std() > 0 and inst.size.std() > 0
        else float("nan"),
    }


def _gini(x: np.ndarray) -> float:
    if x.size == 0 or x.sum() == 0:
        return 0.0
    s = np.sort(x)
    i = np.arange(1, s.size + 1)
    return float((2 * (i * s).sum()) / (s.size * s.sum()) - (s.size + 1) / s.size)


def _between_share(sizes: np.ndarray, mean_q: np.ndarray, quality: np.ndarray) -> float:
    """Size-weighted between-institution share of total quality variance."""
    grand = quality.mean()
    ss_between = float((sizes * (mean_q - grand) ** 2).sum())
    ss_total = float(((quality - grand) ** 2).sum())
    return ss_between / ss_total if ss_total > 0 else 0.0
