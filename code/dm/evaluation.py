"""Evaluation: how a proposal's quality turns into a funding outcome.

Deliberately minimal. We do not model reviewers, panels, reviewer types, or
score aggregation. A proposal's chance of being funded is a function of its
quality, and everything between the two is collapsed into a single noise term:

    observed score  S_j = Q_j + eps_j,   eps ~ N(0, sigma_eval)

Allocation then ranks on S and funds down to the payline, which makes
P(funded | Q) a smooth increasing function of quality without any of the
machinery. `sigma_eval` is the *only* evaluation parameter, and it is
calibrated rather than chosen: `calibrate_sigma_eval` performs the fit and
`SIGMA_EVAL` below is the fitted value every experiment runs at.

The same collapsed form is reused for the two triage signals, with their own
noise levels (`InstitutionalCap.sigma_inst` and `IndividualCap.sigma_self` in
`triage.py`), so "who triages better" is one number per triager, on the same
scale.
"""

from __future__ import annotations

import numpy as np


def observed_score(
    quality: np.ndarray, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """Noisy observation of latent quality."""
    if sigma <= 0.0:
        return quality.copy()
    return quality + rng.normal(0.0, sigma, size=quality.shape)


def sometimes_funded_share(
    quality: np.ndarray,
    sigma: float,
    payline: float,
    rng: np.random.Generator,
    n_draws: int = 200,
) -> dict[str, float]:
    """Reproduce the Graves et al. (2011) bootstrap classification.

    Draw the evaluation noise repeatedly, re-rank, re-apply the funding line,
    and classify each proposal by how often it is funded:

        always     funded in every draw
        sometimes  funded in some draws and not others
        never      funded in no draw

    Graves et al. report 9% always / 29% sometimes / 61% never for NHMRC
    Project Grants, equivalently that 59% of funded grants are sometimes not
    funded. This is the target our evaluation noise is calibrated against.
    """
    n = quality.size
    budget = max(1, int(round(payline * n)))
    funded_count = np.zeros(n, dtype=np.int32)

    for _ in range(n_draws):
        s = observed_score(quality, sigma, rng)
        winners = np.argpartition(-s, budget - 1)[:budget]
        funded_count[winners] += 1

    always = float(np.mean(funded_count == n_draws))
    never = float(np.mean(funded_count == 0))
    sometimes = 1.0 - always - never

    # Of the proposals funded in a typical draw, what share are not funded in
    # every draw? This is the "59% of funded grants" restatement.
    typical_funded = funded_count >= (n_draws // 2)
    unstable_funded = typical_funded & (funded_count < n_draws)
    share_unstable = (
        float(unstable_funded.sum() / typical_funded.sum())
        if typical_funded.any()
        else 0.0
    )

    return {
        "always": always,
        "sometimes": sometimes,
        "never": never,
        "funded_sometimes_not": share_unstable,
    }


# Published targets, Graves, Barnett & Clarke (2011), BMJ 343:d4797.
GRAVES_TARGET = {"always": 0.09, "sometimes": 0.29, "never": 0.61}
# The payline those figures were measured at, and therefore the one the fit is
# performed at. Not the payline the model is run at, which is a free parameter.
GRAVES_PAYLINE = 0.229

# The fitted value. `calibrate_sigma_eval` is what produces it, but it is held
# as a constant so that every experiment shares one calibration rather than
# re-fitting per run on a different population draw. Every entry point uses
# this; nothing in the model should carry its own evaluation-noise default.
# tests/test_mechanics.py::test_sigma_eval_reproduces_graves is what keeps the
# constant honest against the three published bins.
SIGMA_EVAL = 0.195


def calibrate_sigma_eval(
    quality: np.ndarray,
    payline: float,
    rng: np.random.Generator,
    target_sometimes: float = GRAVES_TARGET["sometimes"],
    bounds: tuple[float, float] = (0.01, 5.0),
    tol: float = 0.002,
    max_iter: int = 40,
    n_draws: int = 200,
) -> tuple[float, dict[str, float]]:
    """Solve for the evaluation noise reproducing the observed instability.

    The share of proposals in the "sometimes funded" band increases
    monotonically with noise (at zero noise the ranking is deterministic and
    nothing is sometimes-funded; at very high noise almost everything is), so a
    bisection is sound.

    This replaces an arbitrary noise parameter with a measured one, and is the
    methodological answer to the calibration weakness in this project's anchor
    paper, whose noise level was chosen "to make mechanism differences visible".
    """
    lo, hi = bounds
    best = (hi, {})
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        got = sometimes_funded_share(quality, mid, payline, rng, n_draws=n_draws)
        best = (mid, got)
        err = got["sometimes"] - target_sometimes
        if abs(err) < tol:
            break
        if err > 0:
            hi = mid
        else:
            lo = mid
    return best
