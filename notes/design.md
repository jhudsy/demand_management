# Demand Management ABM — Design Notes

Working notes for the model behind
[[institutional-vs-individual-demand-management]].

**Status: partly superseded.** These are design notes written before and during
implementation, kept for the record of *why* choices were made and what was
rejected. Where they disagree with `paper/findings.tex`, the paper is current.
Section references to `exp01`/`exp02` describe exploratory scripts that have
since been removed; the surviving experiments are listed in the repository
README.

## What this model is for

Compare demand-management regimes that differ in **the unit the quota binds on**,
and therefore in **who performs the triage the quota forces**:

- individual caps (self-triage),
- institutional caps (delegated triage by the organisation),
- performance-conditioned caps and cooling-off periods (mechanical triage on history),
- no cap (baseline).

Evaluated against funded-portfolio quality, Matthew-effect dynamics, institutional
income incidence, equity, and total welfare — **not** against success rate, which
rises under every regime by construction.

This is not an AI paper. No `f`, `rho`, `kappa`, or AI-adoption parameter.

## Relationship to prior code — settled

**There is no prior ABM code in this repository.** `raw/papers/` contains only the
LaTeX and PDFs of the anchor paper; no simulation source exists anywhere in the
tree. The "reuse the allocation kernel" plan from the idea page therefore means
*reimplement a comparable kernel*, not import one. This model is built from
scratch, and the question of how much old code to carry is moot.

Consequence: we are free to choose a clean architecture, and we are obliged to
validate against something external — which is why replication matters.

## AutoSci conventions — checked

AutoSci has **no convention for code directories**. `wiki/experiments/` holds
metadata pages only (hypothesis, setup, metrics, outcome); the `methods` entity
has a `code_repo` string field that points at code living elsewhere. So
`demand_management/{code,paper,notes,results}/` conflicts with nothing. The wiki
side of this work will be an `experiments/` page linked to the idea, with
`code_repo` pointing here.

## Architecture

```
demand_management/
  code/
    dm/
      config.py       frozen dataclasses for every parameter block
      population.py   institutions, PIs, quality, sorting, size distribution
      proposals.py    proposal generation, effort allocation, effort offset
      review.py       reviewer signal + noise; panel aggregation
      triage.py       THE NEW LAYER: self / institutional / conditioned filters
      allocation.py   SPR, pure lottery, threshold lottery, weighted lottery
      dynamics.py     multi-round loop, capacity feedback, funding history
      metrics.py      the five metric families
      runner.py       sweep driver, seeding, parallelism, result serialisation
    validation/       replication of published models (see validation.md)
    experiments/      scripts that produce the paper's figures and tables
    tests/            unit + property tests
  paper/              LaTeX
  notes/              these
  results/            generated artefacts (gitignored except manifests)
```

Design rules:

- **Deterministic given a seed.** Every stochastic draw goes through an explicit
  `numpy.random.Generator` passed down; no global RNG state.
- **Config is data.** Every run is fully described by a serialisable config object
  written next to its results, so any figure can be regenerated from its manifest.
- **Triage and allocation are separate interfaces.** A regime is a
  `(triage_rule, allocation_rule)` pair. This is the whole architectural point:
  the paper's claim is that demand management is a *pre-allocation filter*, and the
  code should make that separation structural rather than tangled.
- **Metrics operate on a recorded event log**, not on the fly, so new metrics can
  be computed from stored runs without re-simulating.

## Core parameters (to be pinned in `config.py`)

**Population / institutional structure — the load-bearing block**

| Parameter | Meaning | Default / sweep |
|---|---|---|
| `n_institutions` | number of organisations | 50 |
| `size_dist` | institution size distribution | `uniform`, `lognormal`, `powerlaw` |
| `rho_sort` | correlation between institution effect and researcher quality | **0 → 0.8 sweep**; this is what H1 lives on |
| `sigma_within` | within-institution quality SD | sweep |
| `size_quality_corr` | do larger institutions have higher mean quality | sweep |

**Triage**

| Parameter | Meaning |
|---|---|
| `alpha_I`, `beta_I`, `rho_I`, `sigma_I` | institutional triage signal: loading on quality, prestige, presentation; noise |
| `alpha_self`, `sigma_self` | applicant self-triage signal |
| `inst_objective` | `awards` / `income` / `prestige` / `fair` |
| `psi` | effort offset — fraction of freed effort re-spent on surviving proposals |
| `triage_timing` | fraction of full effort sunk before triage runs |
| `lambda_leak` | role-substitution circumvention rate |

**Evaluation / system**

| Parameter | Meaning |
|---|---|
| `sigma_reviewer` | reviewer noise — **calibrated, not chosen** (see below) |
| `n_reviewers` | reviews per proposal |
| `payline` | budget / submissions; sweep across the 10–15% threshold |
| `n_rounds`, `burn_in` | round loop |
| `feedback_gain` | funding → future quality (cumulative advantage) |

## Calibration targets, not free choices

Two parameters are pinned to published measurements rather than chosen for
visibility. This is the main methodological upgrade over the anchor paper, whose
parameters were explicitly chosen "to make mechanism differences visible."

1. **Reviewer noise** → calibrate `sigma_reviewer` so the model reproduces
   **29% of proposals in the "sometimes funded" band** under bootstrap
   resampling of panel scores (Graves et al. 2011). Implemented as an inverse
   problem in `validation/calibrate_noise.py`: run the bootstrap procedure on
   simulated panels and solve for the noise level matching 9% always / 29%
   sometimes / 61% never.
2. **Applicant effort** → anchor the cost scale to **34–38 working days per
   proposal** (Herbert 2013; Barnett 2015), so welfare numbers are in
   interpretable units (person-years, and a fraction of scheme budget comparable
   to Herbert's 14%).

## Open design questions

Recorded here as they arise; resolved ones move to the answered list with a date.

1. Should the model be a **superset** of Roebber & Schultz (so switching
   institutions off and using two behavioural groups reproduces their table
   exactly), or should R&S be a **standalone replication module** sitting beside a
   cleanly-designed model of our own? See `validation.md`.
2. Should institution sizes be **synthetic** (swept distributions) or
   **calibrated** to a real system (UK HESA/REF, or NHMRC applicant counts)?
3. Is `feedback_gain` in v1, or deferred? Without it the Matthew-effect metrics
   measure static inequality rather than cumulative advantage; with it, run time
   and parameter count both grow.
4. Do we model **researcher mobility** between institutions? A binding
   institutional cap creates a real relocation incentive. Currently listed as a
   declared simplification.

---

## Decision 2026-08-04 — institutions differ along FOUR separable channels

"Some institutions are stronger" is not one claim. Conflating these is the main
conceptual error available in this literature, and separating them is a result in
itself.

| Channel | What varies | Acts on | If a flat institutional cap binds here... |
|---|---|---|---|
| **talent** | latent quality `Q` of members | the science itself | ...it **destroys good work** |
| **support** | presentation at fixed `Q`, *and* internal triage accuracy | research office, grant-writing help, editing | ...it **corrects a distortion**, but discards the system's most accurate triage |
| **prestige** | reviewer score at fixed `Q` | tradition, reputation | ...it **corrects a distortion**, with no offsetting cost |
| **capacity** | number of staff | shots on goal | ...it is a **pure size transfer** |

A research office raises support, not talent. Tradition raises prestige, not
talent. Only talent means the institution genuinely does better science.

**The headline claim this sets up:** the welfare verdict on institutional caps is
not a single number, it is conditional on which channel dominates in the real
system. That converts an unresolvable policy argument into an empirical question a
funder can actually investigate.

Implemented in `dm/config.py::InstitutionConfig`. `rho_talent` is parameterised so
it is *literally* `corr(researcher quality, institution talent effect)`, with total
latent-quality variance held at 1 regardless of the split — so changing it moves
variance between levels rather than adding any. Both properties are enforced by
tests.

## Scenario set (`dm/scenarios.py`)

These are to this paper what cases (a)-(g) are to Roebber & Schultz: worlds
differing in one structural assumption at a time.

| | Scenario | Sizes | Quality structure |
|---|---|---|---|
| S0 | null | equal | none — the case where the unit of the cap must NOT matter |
| S1 | size only | lognormal | none |
| S2 | talent only | equal | talent clusters (`rho=0.5`) |
| S3 | size and talent | lognormal | talent clusters, correlated with size |
| S4 | support only | equal | none — research offices differ |
| S5 | prestige only | equal | none — reputation differs |
| S6 | composite | lognormal | all channels on, all correlated with size |
| S7 | heavy tail | power law | talent clusters; stress case for the redistribution result |

**S0 is the falsification case.** It is the assumption made by the anchor paper's
ABM (affiliation uniform and independent of quality). If the institution-vs-
individual difference does not vanish in S0, the model is wrong.

Verified diagnostics:

```
scenario             insts  small  large sizeGini Qvar btwn corr(sz,Q)  supp prest
S0_null                 50     40     40    0.000     0.032         --  0.00  0.00
S1_size_only            50      5    143    0.411     0.029       0.02  0.00  0.00
S2_talent_only          50     40     40    0.000     0.279         --  0.00  0.00
S3_size_and_talent      50      5    143    0.411     0.276       0.49  0.00  0.00
S4_support_only         50     40     40    0.000     0.027         --  0.50  0.00
S5_prestige_only        50     40     40    0.000     0.027         --  0.00  0.50
S6_composite            50      5    162    0.451     0.198       0.42  0.39  0.45
S7_heavy_tail           50      5   1048    0.754     0.210       0.44  0.00  0.00
```

`Qvar btwn` recovers `rho_talent^2` as designed (S2: 0.279 vs 0.25 target).

## Replication — CLOSED 2026-08-04

Scope-limited and closed deliberately. Case (a) validated the engine
out-of-sample and caught a real error in the published paper (threshold 105, not
110). Cases (b)-(e) turn on **reviewer-type heterogeneity** (honest / sloppy /
selfish), which is not a demand-management mechanism and is not in our model --
we calibrate reviewer noise against Graves instead. Chasing them was work on
someone else's research question.

Case (g)'s cooling-off inversion is not reproduced, so **H6 must be tested in our
own model rather than cited from theirs**. That is the stronger position anyway:
H6 is a mechanism claim (a rule that removes a group's weakest writers thins
competition for its strongest) and our model can generate it or fail to on its own
terms.

## Correction 2026-08-04 — four channels collapsed to one parameter

The four-channel decomposition above (talent / support / prestige / capacity) was
over-modelled and is superseded.

**Why it was wrong.** For the mechanics of triage, the source of an institution's
advantage is unobservable and irrelevant: whether its proposals score higher
because its researchers are better, its research office is better, or its name
carries weight, the ranking is the same and the same proposals survive the cap.
Modelling three sources separately (plus three size-correlations) tripled the
parameter space to distinguish things the mechanism cannot tell apart.

**What survives, and why it is a demand-management result rather than a quality
one.** Selection efficiency is measured against *latent* quality. So the same
triage decisions carry opposite welfare verdicts depending on whether the
advantage is real:

- advantage real -> capping strong institutions **destroys good science**
- advantage apparent -> capping strong institutions **corrects a distortion**

Only the *institutional* mechanism is exposed to this, because only it conditions
on the institution. An individual cap is blind to it. So this is precisely a
statement about the comparison the paper is making.

**Implementation.** Two parameters replace six:

| Parameter | Meaning |
|---|---|
| `rho_advantage` | how concentrated institutional advantage is (0 = affiliation carries no information) |
| `advantage_real_share` | fraction of that advantage reflected in latent quality; **0 -> 1 sweep is the headline experiment** |

Invariant enforced by test: changing `advantage_real_share` moves the *same*
institutional advantage between latent quality and score bias. Triage sees the
sum and is unchanged; welfare sees the split and is not. S2 and S3 differ in this
parameter alone and are the paper's central contrast.

Internal triage accuracy is now its own independent parameter rather than being
derived from a "support" channel.

### Revised scenario set

| | Scenario | Sizes | Advantage |
|---|---|---|---|
| S0 | null | equal | none — the falsification case |
| S1 | size only | lognormal | none |
| S2 | advantage real | equal | concentrated, 100% real |
| S3 | advantage apparent | equal | concentrated, 0% real — identical triage to S2 |
| S4 | size and advantage | lognormal | concentrated, 50% real, size-correlated |
| S5 | heavy tail | power law | concentrated, 50% real |

Plus `real_share_sweep()`: hold triage fixed, vary only the realness, and locate
where the institutional cap's effect on selection efficiency changes sign.

Verified:

```
scenario                  small  large sizeGini  advSD  realSD  biasSD Qvar btwn
S0_null                      40     40    0.000   0.00    0.00    0.00     0.032
S1_size_only                  5    143    0.411   0.00    0.00    0.00     0.029
S2_advantage_real            40     40    0.000   0.50    0.50    0.00     0.279
S3_advantage_apparent        40     40    0.000   0.50    0.00    0.50     0.027
S4_size_and_advantage         5    143    0.411   0.47    0.24    0.24     0.105
S5_heavy_tail                 5   1048    0.754   0.37    0.19    0.19     0.087
```

S2 and S3 carry identical advantage (SD 0.50) but S2 routes it entirely into
quality (between-institution quality variance 0.279) and S3 entirely into score
bias (0.027, i.e. none). 13 property tests passing.

## Correction 2026-08-04 (second) — institutions differ in size and quality, nothing else

The `advantage_real_share` parameter is deleted. The four-channel version before
it is deleted. Institutions now differ in exactly two ways:

| Parameter | Meaning |
|---|---|
| `size_dist` / `size_spread` | headcount distribution |
| `rho_quality` | literally corr(researcher latent quality, institution effect); 0 = affiliation carries no information |
| `corr_size_quality` | are larger institutions also better? |

**Why the real/apparent split was dropped.** It described a property of the
*evaluation system*, not of demand management. If reviewers over-reward a
prestigious name, that bias is present identically with and without a cap, is not
caused by demand management, and is corrected by a different instrument
(structured forms, blinding). Carrying it here would have (a) confounded two
policies, (b) forced every result to be reported along an axis nobody can measure
in a real system, and (c) doubled the presentation burden of the paper for no
decision-relevant gain. Individual track-record prestige remains in
`ResearcherConfig`, where it feeds the reviewer signal as a reviewer-side effect
rather than an institutional one.

### Final scenario set

| | Scenario | Sizes | Quality |
|---|---|---|---|
| S0 | null | equal | independent of affiliation — **falsification case** |
| S1 | size only | lognormal | independent of affiliation |
| S2 | quality only | equal | concentrated |
| S3 | size and quality | lognormal | concentrated, uncorrelated with size |
| S4 | big and good | lognormal | concentrated, correlated with size |
| S5 | heavy tail | power law | concentrated, correlated with size |

Plus `quality_concentration_sweep()`: vary `rho_quality` from 0 to 0.8. **H1
predicts the institutional-vs-individual gap grows along this axis and vanishes
at 0.** That sweep is the primary test of the paper's main claim, and S0 is its
falsification case.

Verified:

```
scenario               smallest largest sizeGini instQ SD Qvar btwn corr(sz,Q)
S0_null                      40      40    0.000     0.00     0.032         --
S1_size_only                  5     143    0.411     0.00     0.029       0.02
S2_quality_only              40      40    0.000     0.50     0.279         --
S3_size_and_quality           5     143    0.411     0.50     0.290      -0.14
S4_big_and_good               5     143    0.411     0.47     0.276       0.49
S5_heavy_tail                 5    1048    0.754     0.47     0.303       0.45

sweep:  rho  0.00  0.20  0.40  0.50  0.60  0.80
   Qvar btwn  0.029 0.069 0.195 0.290 0.404 0.680
```

The sweep recovers rho^2 plus sampling noise, as designed.

---

## First end-to-end run — 2026-08-04 (PRELIMINARY, single seed, no replicates)

Layers built: `evaluation.py` (quality -> outcome, one noise parameter),
`triage.py` (the mechanisms), `simulate.py` (round loop + metrics).

### Evaluation noise is now calibrated, not chosen

Fitting **one** parameter to **one** target reproduces all three of Graves'
published bins, at Graves' actual payline of 22.9% (620 funded of 2705):

| | model | Graves |
|---|---|---|
| always funded | 10.9% | 9% |
| sometimes funded | **28.9%** | **29%** (fitted) |
| never funded | 60.1% | 61% |

`sigma_eval = 0.195` against a proposal-quality SD of ~1.12. The other two bins
falling out is a validation, not a fit.

### H1 IS CONTRADICTED BY THE FIRST RUN

At matched volume (~50% of intended proposals admitted under every mechanism),
the institutional cap produces a **higher**-quality funded portfolio than the
individual cap in every scenario, including the null:

```
S0_null              kept   Qfund  top10 killed  HHI     wasted
no cap               1.00   1.694     0.0%       0.0244      0
individual cap K=1   0.52   1.804    36.0%       0.0244      0
institutional flat   0.50   2.029     0.9%       0.0290   1507
two-stage EOI        0.50   2.048     2.6%       0.0289    225
```

**Mechanism:** an institution pools across ~40 researchers, so it selects the
best 31 of 60 proposals. An individual cap cannot reallocate across a person --
a researcher with three good proposals must drop two regardless of how good they
are. The individual cap's own loss term, `sum_i max(0, r_i - K)`, is *larger*
than the institutional one at matched volume, and it lands on top-decile work:
**36% of top-decile proposals are destroyed by K=1**, against ~1% by the
institutional cap.

H1's argument was `sum_k max(0, R_k - M)` grows with the variance of the capped
unit -- true, but it assumed perfect triage and ignored that the individual cap
has a bigger loss of the same kind. The pooling advantage dominates.

**This may still flip.** The run assumes `sigma_inst == sigma_self == 0.6`, i.e.
institutions triage exactly as well as researchers judge their own work. The
field evidence is that self-triage is poor (Herbert's kappa = -0.06), but nothing
says institutions are better. The phase boundary in
(sigma_inst, sigma_self) is now the decisive experiment, and H1 should be
restated as conditional on it rather than asserted.

### My S0 falsification test was mis-specified

I claimed institutional and individual caps must give identical results when
institutions are equal-sized and quality is independent of affiliation. They do
not, and should not: they bind at different *pooling levels*, which matters even
with no institutional structure at all. The correct null for S0 is that the
institutional cap creates no systematic *between-institution* inequity (HHI 0.0290
vs 0.0244 no-cap -- small, as expected), not that it matches the individual cap.

### Two-stage EOI looks dominant on the welfare dimension

Same or better funded quality than the institutional cap, at **225 vs 1507**
proposal-equivalents of wasted effort -- because what it discards is the cheap
stage. If this survives replication it is the paper's clearest policy result.

### H2 visible already (S4, big-and-good)

| mechanism | Qfund | HHI |
|---|---|---|
| no cap | 1.931 | 0.0820 |
| institutional **flat** | 2.225 | **0.0678** |
| institutional **proportional** | **2.328** | 0.1080 |

Flat caps deconcentrate (HHI down) at a quality cost; proportional caps do the
opposite. That is the efficiency-deconcentration frontier, showing up
immediately.

### The null metric behaves as predicted

Success rate is **15.0% in every single row of every scenario**. It discriminates
between nothing, which is exactly the argument for not evaluating demand
management on it.

### Deferred, as agreed

Track record / cumulative advantage (funding raises future funding probability) is
noted as future work and not implemented. `simulate.py` has the round loop it
would hang off.

### Caveats on the above

Single seed per configuration, no replicates, no confidence intervals. Treat every
number as directional until the runner does proper replication.

---

## 2026-08-04 — three design changes, and the EOI result reversed

### 1. Triage accuracy = review accuracy (removes a free parameter)

An institutional triage panel is doing the same job as a funder panel, so it is
given the same reliability: `sigma_inst = sigma_self = sigma_eval = 0.195`,
calibrated against Graves. This is a principled choice rather than an assumption
that institutions are better or worse judges, and it removes the free parameter
the earlier phase-boundary experiment was going to sweep.

### 2. Volume now responds to entry cost — and this reverses the EOI finding

Previously `TwoStage` was handed a fixed set of intended proposals and invited a
fraction of them, which quietly assumed a cheap first stage draws no extra
entries. That was wrong. Volume now scales as
`(1 / entry_cost) ** volume_elasticity`, and an EOI costs 0.15 of a full
proposal.

Consequence (S0, elasticity 0.5):

| mechanism | intended | wasted effort | volume advantage |
|---|---|---|---|
| no cap | 3013 | 0 | 1.59 |
| individual cap K=1 | 3008 | 0 | **1.00** |
| institutional flat | 3010 | 1510 | 1.66 |
| **two-stage EOI** | **7745** | **581** | **1.66** |

The EOI stage draws **2.6x the volume**, and its wasted effort rises from 225 to
581. Its earlier "dominance" was an artefact of holding volume fixed.

**More importantly, it does nothing about winning by luck.** `volume_advantage`
is the funding rate of high-submission researchers against low-submission ones
*within quality quintiles*, so quality is held constant. Two-stage sits at 1.66,
statistically the same as no cap at 1.59. Cheap entry means more shots at a noisy
screen, which is precisely the objection.

**Only the individual cap eliminates it, at exactly 1.00** — because with K=1
everyone gets the same number of shots, by construction.

### 3. Proposal level and applicant level DISAGREE, and it is systematic

| S0 | Qfund (proposal) | appPrem (applicant) |
|---|---|---|
| individual cap K=1 | 1.860 | **1.358** |
| institutional flat | **2.042** | 0.937 |

Institutional caps produce a better funded *portfolio* and a worse-selected set of
funded *people*. The institution concentrates its quota on its strongest few
researchers, who then win repeatedly; the individual cap spreads entries across
people. This is the same proposal-level vs applicant-level estimand distinction
as in the anchor paper, arriving here for an entirely different reason, and it
means the institutional-vs-individual question has **no answer without naming the
estimand**.

## Optimal cap design — the objectives disagree

An exploratory script (since removed) swept each mechanism's quota against
four objectives; `exp07_robustness.py` now does this across the whole parameter
space.

| objective | S0 winner | S4 winner |
|---|---|---|
| quality (funded portfolio) | institutional M=5 (2.840) | institutional M=5 (2.952) |
| people (applicant selectivity) | individual K=1 (1.365) | individual K=1 (1.361) |
| welfare (quality per unit effort) | individual K=1 (0.278) | individual K=1 (0.308) |
| spread (de-concentration) | two-stage inv=0.8 | institutional M=45 |
| volume advantage (lower better) | individual K=1 (1.00) | individual K=1 (1.00) |

**No single cap is best, and the disagreement is not marginal.** The tightest
institutional cap maximises funded-portfolio quality (2.840) while having the
*worst* welfare of any mechanism tested (0.036 against 0.278 for individual K=1 --
a factor of ~8), because nearly every proposal is written and then discarded.
That is H4 demonstrated: tight institutional caps buy proposal quality by burning
applicant effort.

So "what should the cap be" has no answer until the funder says what it is
optimising. That is a result, and it is probably the paper's central practical
message.

### Still to do

- Replication/CIs. *Resolved:* every reported figure is now replicated over
  16 independent populations, with instrument contrasts differenced within
  population; see `dm/runner.py`.
- `volume_elasticity` is assumed at 0.5 and should be swept; the EOI conclusion
  depends on it.
- Cooling-off and resubmission-limit mechanisms are implemented but not yet in
  the comparison tables.
- Track record / cumulative advantage remains deferred as agreed.

---

## 2026-08-04 (later) — a bug of mine, resubmission modelled, replicated results

### Correction: the budget was a share, not a fixed sum

`budget = payline * submitted` made success rate **identically 15% by
construction**. My earlier claim that "success rate discriminates between
nothing" was therefore vacuous -- it was hard-coded, not observed. Funders have a
fixed number of grants, so the budget is now set once from the reference volume
and held constant across mechanisms. Success rate is now endogenous and ranges
7.4%-30% across mechanisms.

This changed several earlier conclusions. Most importantly, with a fixed budget a
cap **reduces the funder's choice set**, so restricting volume now *lowers* funded
quality rather than raising it. Individual K=1 went from 1.860 (apparently better
than no-cap) to 1.494 (clearly worse than no-cap 1.791).

### Resubmission is now modelled

Previously `is_resubmission` was always False, so the resubmission-limit
mechanism could not bind at all. Rejected proposals now return next round with a
quality gain, and **near-misses return at a higher rate than the rest** -- which
is what makes a resubmission limit act on the band where the original decision
was least reliable.

### Replicated results, S0 null, 20 seeds, 95% CI

| mechanism | intended | Qfund | appPrem | volAdv | succ% | welfare |
|---|---|---|---|---|---|---|
| no cap | 6077 | 1.791±0.015 | 1.183±0.012 | 1.555±0.026 | 7.4% | **0.269** |
| individual K=1 | 3694 | 1.494±0.014 | 1.023±0.009 | **1.000±0.000** | 26.3% | 0.224 |
| individual K=2 | 4491 | 1.696±0.015 | 1.129±0.010 | 1.140±0.009 | 14.2% | 0.254 |
| institutional M=30 | 3576 | 1.760±0.017 | **0.480±0.010** | 1.604±0.017 | 30.0% | **0.111** |
| institutional M=45 | 3990 | 1.773±0.015 | 0.636±0.011 | 1.632±0.014 | 20.0% | 0.150 |
| cooling off | 5076 | 1.770±0.016 | 1.146±0.012 | 1.593±0.021 | 10.6% | 0.265 |
| resub limit 1 | 5167 | 1.731±0.016 | **1.260±0.012** | 1.526±0.018 | 10.3% | 0.260 |
| resub limit 0 | 4392 | 1.705±0.016 | 1.251±0.012 | 1.551±0.014 | 15.0% | 0.256 |
| two-stage inv=0.5 | 10348 | **2.274±0.019** | 1.100±0.015 | 1.590±0.019 | 8.7% | 0.262 |

**Findings, all significant at 95% unless noted:**

1. **Every volume-reducing cap lowers funded quality.** With a fixed budget,
   fewer submissions means a smaller choice set and the funder digs deeper into
   the distribution. This is the opposite of the naive expectation and it is the
   cleanest result so far.
2. **Institutional caps wreck applicant-level selectivity while doing nothing for
   portfolio quality.** appPrem falls to 0.480 against 1.183; Qfund at M=45
   (1.773) is *not distinguishable* from no cap (1.791). They concentrate funding
   on fewer people without improving what gets funded, and they have the worst
   welfare of anything tested (0.111 vs 0.269) because of discarded effort.
3. **Only an individual cap of 1 removes the volume advantage** -- exactly 1.000,
   by construction. K=2 already leaks (1.140).
4. **Resubmission limits are the only mechanism that IMPROVES applicant
   selectivity** (1.260 vs 1.183), by stopping the same project being ground
   through repeatedly. Novel, and not something the literature discusses.
5. **Doing nothing has the best welfare** (0.269). Every cap costs welfare; the
   institutional ones cost the most.

### The two-stage result is entirely an artefact of volume inflation

Sweeping `volume_elasticity`, which the earlier conclusion depended on:

| elasticity | 2-stage intended | 2-stage Qfund | 2-stage volAdv |
|---|---|---|---|
| **0.00** | 3799 | **1.741±0.032** | 1.697 |
| 0.25 | 6308 | 2.017±0.035 | 1.678 |
| 0.50 | 10354 | 2.273±0.043 | 1.592 |
| 0.75 | 16780 | 2.511±0.044 | 1.702 |
| 1.00 | 25806 | 2.708±0.047 | **2.363±0.093** |

**At zero elasticity the two-stage quality advantage vanishes** (1.741 against
no-cap's 1.791). Its entire apparent superiority comes from drawing more entries,
not from filtering better. And at elasticity 1.0 it becomes a volume disaster:
25,806 intended proposals -- 4x the no-cap level -- with volume advantage 2.363,
far worse than any other mechanism.

So an EOI stage does not manage demand. It *subsidises* demand, and hands the
advantage to whoever can enter most often. This is the objection raised in
discussion, and the model confirms it in the strongest possible form.

### Still open

- All results are S0 (null institutional structure). The scenario sweep across
  S1-S5 has not been re-run since the budget fix.
- `volume_elasticity` has no empirical anchor. Barnett (2015) is the closest
  evidence and suggests effort is roughly invariant to task size, which if
  anything implies a *high* elasticity for entry count.
- Track record / cumulative advantage still deferred.
