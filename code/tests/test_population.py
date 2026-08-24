"""Property tests: the population layer must mean what the parameters say."""
import numpy as np
import pytest

from dm import population
from dm.config import InstitutionConfig, RunConfig
from dm.scenarios import scenarios


def _build(cfg, seed=0):
    return population.build(cfg, np.random.default_rng(seed))


@pytest.mark.parametrize("name,cfg", list(scenarios().items()))
def test_population_size_exact(name, cfg):
    inst, res = _build(cfg)
    assert res.n == cfg.institutions.n_researchers
    assert inst.size.sum() == cfg.institutions.n_researchers
    assert inst.size.min() >= cfg.institutions.min_size


def test_rho_quality_is_the_between_institution_correlation():
    """rho_quality must literally be corr(researcher quality, institution effect)."""
    for rho in (0.0, 0.3, 0.6, 0.85):
        cfg = RunConfig(
            institutions=InstitutionConfig(
                n_institutions=200, n_researchers=40000, rho_quality=rho
            )
        )
        inst, res = _build(cfg, seed=7)
        if rho == 0.0:
            # Degenerate: no talent variance at all, so the correlation is
            # undefined rather than zero. The substantive claim is that
            # affiliation carries no quality information.
            assert np.allclose(inst.quality, 0.0)
            continue
        got = np.corrcoef(res.quality, inst.quality[res.institution])[0, 1]
        assert got == pytest.approx(rho, abs=0.03), f"rho={rho} got {got:.3f}"


def test_total_quality_variance_is_invariant_to_the_split():
    """Changing rho_quality must move variance between levels, not add any."""
    sds = []
    for rho in (0.0, 0.4, 0.8):
        cfg = RunConfig(
            institutions=InstitutionConfig(
                n_institutions=200, n_researchers=40000, rho_quality=rho
            )
        )
        _, res = _build(cfg, seed=11)
        sds.append(res.quality.std())
    assert max(sds) - min(sds) < 0.03, f"total quality SD drifted: {sds}"


def test_null_scenario_has_no_institutional_structure():
    """S0 is the case in which the unit of the cap must not matter."""
    inst, res = _build(scenarios()["S0_null"])
    # Equal headcount everywhere.
    assert inst.size.std() == 0.0
    # Affiliation carries no quality information at all.
    assert np.allclose(inst.quality, 0.0)
    # Institution means should differ only by sampling noise, whose scale is
    # 1/sqrt(size) -- not by any structural effect.
    mean_q = np.array([res.quality[res.institution == k].mean() for k in range(inst.n)])
    expected_se = 1.0 / np.sqrt(inst.size.mean())
    assert mean_q.std() < 1.5 * expected_se, (
        f"institution means vary more than sampling noise allows: "
        f"{mean_q.std():.3f} vs SE {expected_se:.3f}"
    )


def test_size_dispersion_increases_with_spread():
    ginis = []
    for spread in (0.3, 0.8, 1.5):
        cfg = RunConfig(
            institutions=InstitutionConfig(size_dist="lognormal", size_spread=spread)
        )
        inst, res = _build(cfg, seed=3)
        ginis.append(population.summarise(inst, res)["size_gini"])
    assert ginis[0] < ginis[1] < ginis[2], ginis


def test_size_advantage_correlation_is_respected():
    cfg = RunConfig(
        institutions=InstitutionConfig(
            n_institutions=300,
            n_researchers=60000,
            size_dist="lognormal",
            rho_quality=0.5,
            corr_size_quality=0.7,
        )
    )
    inst, res = _build(cfg, seed=5)
    got = np.corrcoef(np.log(inst.size.astype(float)), inst.quality)[0, 1]
    assert got == pytest.approx(0.7, abs=0.08), got


