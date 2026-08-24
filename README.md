# Demand management in research funding

An agent-based model comparing the instruments funders use to restrict what may
be submitted to a scheme — institutional quotas, per-investigator caps,
cooling-off rules, resubmission limits and two-stage screening — when each is
tuned to deliver the **same review load**.

The write-up is `paper/findings.tex`. This file describes the code.

---

## Quick start

```bash
cd code
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q          # ~2 s
.venv/bin/python experiments/exp08_headline.py # the main comparison, ~5 min
```

Every experiment prints to stdout and writes nothing except `exp07`, which also
saves a CSV. Redirect to `results/` to keep the output:

```bash
.venv/bin/python experiments/exp08_headline.py > results/exp08_headline.txt
```

---

## Reproducing the paper

Everything in `paper/findings.pdf` traces to `code/results/`, which is committed
so the numbers can be checked without a re-run. To regenerate them from scratch:

```bash
cd code
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q
.venv/bin/python experiments/exp08_headline.py           > results/exp08_headline.txt
.venv/bin/python experiments/exp07_robustness.py         > results/exp07_report.txt
.venv/bin/python experiments/exp05_payline.py            > results/exp05_payline.txt
.venv/bin/python experiments/exp10_two_stage_mechanism.py > results/exp10_two_stage.txt
.venv/bin/python -m validation.report                    > results/validation.txt
```

Total ~20 min on 14 cores. `exp07` also writes `results/exp07_cells.csv`
(424 parameter settings x 8 mechanisms), which is the input to every
claim-survival percentage in §4.3.

The results are deterministic given the seeds, so a correct re-run reproduces the
committed files byte for byte. If they differ, the RNG stream has moved --- see
the first note under *Things worth knowing* below.

The manuscript is a single self-contained file with an inline `thebibliography`,
so no `.bib` or BibTeX pass is needed:

```bash
cd paper && latexmk -pdf findings.tex
```

---

## Layout

### `code/dm/` — the model

| file | contains |
|---|---|
| `config.py` | `InstitutionConfig` and `RunConfig`: how institutions differ (size distribution, `rho_quality`, size–quality correlation). Validation lives here. |
| `population.py` | Builds institutions and researchers from a config. Latent quality is standardised to N(0,1) with `rho_quality` splitting variance between the institution and the individual, so total variance is invariant to the split. `summarise()` gives the diagnostics the tests assert on. |
| `scenarios.py` | The six named institutional scenarios S0–S5 used throughout. S0 is the null case in which affiliation carries no quality information. |
| `evaluation.py` | The whole evaluation model: an observed score is latent quality plus Gaussian noise. `calibrate_sigma_eval()` fits that noise to the Graves et al. (2011) bootstrap instability; `GRAVES_TARGET` holds the published bins. |
| `triage.py` | The instruments. Each takes the intended proposals and returns a mask of those submitted plus an effort and review-load accounting. `History` carries the rolling per-researcher record that cooling-off and windowed caps condition on. |
| `simulate.py` | `SimConfig` and the round loop: generate, filter, score, allocate, carry resubmissions forward, record. Also the applicant-level measures. |
| `runner.py` | `replicate()` runs one configuration over N independent populations; `paired_diff()` differences two rules **within** population; `welfare()` is quality per unit applicant effort. |
| `sweep.py` | `Cell` (one parameter point), `candidates()` (each instrument with the range of its severity control), and `tune()` (finds the setting hitting a target review load). Direction-aware: cooling-off tightens as its parameter rises, everything else as it falls. |

### `code/experiments/`

| script | produces | feeds |
|---|---|---|
| `exp08_headline.py` | The capacity-matched comparison. Each instrument is tuned to half the uncapped load, then compared against a random-thinning counterfactual tuned to *its own* achieved load, differenced within population. Also the HHI benchmarks. | Paper §4.1, §4.2 |
| `exp07_robustness.py` | 424 parameter settings (one-at-a-time sweep plus a 400-point Latin hypercube), re-tuning every instrument in each. Reports the feasibility frontier, claim-survival rates, elasticity sensitivity and the one-at-a-time ranges. Writes `results/exp07_cells.csv`. | Paper §4.3 |
| `exp05_payline.py` | Decomposition of triage value against deletion loss across paylines. | Paper §4.4 |
| `exp10_two_stage_mechanism.py` | Two things. (1) Whether two-stage's quality advantage is a second independent reading, tested by stripping away the cost and elasticity advantages. (2) How the two-stage load floor depends on what an outline costs to screen. | Paper §4.5, §4.3.3 |

Approximate runtimes on 14 cores: `exp07` ~10 min (parallel), `exp08` ~5 min,
`exp10` ~3 min, `exp05` ~1 min.

`exp07` takes `--joint N` (hypercube points, default 400), `--jobs N` (worker
processes) and `--quick` (12 points, for a smoke test).

### `code/validation/`

Reimplementation of Roebber & Schultz (2011), the only prior agent-based
treatment of proposal-limiting policy. `roebber_schultz.py` is the model,
`report.py` prints the comparison against their published figures. Their
case (a) reproduces out of sample; the later cases could not be reproduced from
the published description. The two `[FAIL]` lines at the end of
`results/validation.txt` are that non-reproduction, and are expected: they are
left failing on purpose rather than tuned away. See `notes/validation.md`.

### `code/tests/`

`test_population.py` covers the population layer: that `rho_quality` really is
the correlation it claims to be, that total quality variance does not drift with
the split, that S0 has no institutional structure.

`test_mechanics.py` covers the round loop, the instruments and the measures.
Every bug found in this model has been here rather than in the population layer,
and each is pinned by a test — see the module docstring for the list.

---

## Things worth knowing before changing anything

**The RNG stream is load-bearing for reproducibility.** One seed drives the
population draw *and* every stochastic step. Adding or removing any `rng` call
shifts every subsequent draw, so results change in the third decimal even when
the change is semantically irrelevant. Removing two unused draws during a
cleanup did exactly that. If a number in the paper moves and you cannot see why,
check whether the number of RNG calls changed.

**Comparisons must be paired.** The set of institutions varies more between
populations than the instruments do within one, so an unpaired interval on a
difference is several times too wide. Use `paired_diff()`, not the difference of
two `replicate()` means.

**Instruments must be compared at equal review load**, never at equal parameter
values, and review load includes everything the funder reads — under two-stage
screening that is every outline as well as every invited proposal.

**`volume_advantage` is NaN when it cannot be measured**, which is what a tight
per-investigator cap produces: if everyone submits once there is no high-volume
group to compare against. Do not coerce it to 1; use `volume_gini`, which is
always defined.

**There is one evaluation-noise constant.** `evaluation.SIGMA_EVAL` is the
fitted value, and every default that needs a review-accuracy number — the round
loop's `sigma_eval`, the individual cap's `sigma_self`, the institutional
quota's `sigma_inst` — refers to it rather than carrying its own. Do not
reintroduce a local literal; `test_sigma_eval_reproduces_graves` checks the
constant against all three published bins, and a second copy would escape that
check.

**Two-stage's `entry_cost` tracks `eoi_cost`.** Entering the scheme costs an
outline, so the two are one number by construction. Set `entry_cost` explicitly
only to break that link deliberately, as `exp10`'s `pre_screen()` does to
suppress the volume response while keeping the screen.

**The grant budget is fixed**, not a share of submissions. An instrument that
inflates volume lowers the realised success rate while the number of grants
stays put — that is the dynamic the model exists to capture.

---

## Notes

`notes/design.md` records the modelling decisions and why alternatives were
rejected; parts of it predate the current experiment set and refer to scripts
that have since been removed. `notes/validation.md` covers the Roebber &
Schultz replication. `notes/findings-2026-08-04.md` is a working log, superseded
by the paper where the two disagree.
