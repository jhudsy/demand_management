# Validation Targets

The model must reproduce published results before any new claim is made. Three
targets, in descending order of usefulness.

## Target 1 — Roebber & Schultz (2011), PLoS ONE 6(4):e18680

The only prior ABM that simulates proposal-limiting policies, and therefore the
primary sanity check. Their published table is our acceptance test.

### Their model, as specified in the paper

| Component | Specification |
|---|---|
| Scientist quality | `Qs ~ N(100, 10)` |
| Proposal quality | `Qp ~ N(Qs, 5)` |
| Group G1 ("few") | 1 proposal every 2 time units; **stops submitting while holding a funded grant** |
| Group G2 ("many") | 1 proposal every time unit regardless of funding |
| Time unit | 6 months; grants are 3-year awards, equal value |
| Correct reviewer | recommends funding only for top 16% (≥ 1 SD above mean, i.e. `Qp ≥ 110`) |
| Harried reviewer | perceives quality as `N(Qp, 5)`, then applies the same rule |
| Selfish reviewer | declines if `Qp > Qs_reviewer`, **or** if `Qp < 0.9 × threshold` |
| Program officer, correct | perfect knowledge of `Qp`; ranks precisely |
| Reputation-based officer | substitutes `Qs` for `Qp` when ranking |
| Review assignment | `K` independent reviewers drawn at random, author excluded |
| Funding rule | each half-year, fund **unanimously positive** proposals up to half the annual target; year-end surplus reconsiders highest-rated unfunded proposals from the previous six months until the target rate is reached |
| Baseline mix | 60% correct / 20% harried / 20% selfish, 15% target funding rate |

### Acceptance test — reproduce this table

G1 success / G2 success / mean funded quality G1 / mean funded quality G2 /
G2 share of funding:

| Case | G1 | G2 | Q̄ G1 | Q̄ G2 | G2 share |
|---|---|---|---|---|---|
| (a) Perfect: all correct, no limits | 21.3% | 34.6% | 111.6 | 112.1 | 76.9% |
| (b) Baseline: 60/20/20, 15% target | 25.2% | 9.9% | 115.0 | 112.8 | 44.8% |
| (c) More selective: top 2% | 34.8% | 4.5% | 111.6 | 116.4 | 19.8% |
| (d) Positive feedback: +5 for G2 | 17.8% | 13.5% | 117.6 | 113.8 | 60.6% |
| (e) Negative feedback: −5 for G2 | 33.9% | 5.6% | 109.7 | 112.0 | 25.1% |
| (f) **G2 limited to one funded grant** | 29.4% | 4.8% | 109.1 | 112.5 | 19.1% |
| (g) **Cooling-off period, 12 months** | 12.5% | 17.4% | 115.7 | 113.0 | 57.7% |

Overall funding rate in case (a): 30.2%. Reviewer load: +13% from (a) to (b);
−22% in (f); −34% in (g).

Cases (f) and (g) are the demand-management cases and the two we most need to
reproduce. The **qualitative** signatures matter more than the third decimal:

- **(f)** the cap suppresses the targeted group *and* **lowers the untargeted
  group's mean funded quality** (115.0 → 109.1) via reduced competition;
- **(g)** the cooling-off period **inverts its own distributional intent** —
  G2's funding share *rises* to 57.7% while G1's success rate halves.

If our implementation does not reproduce those two sign patterns, something is
wrong with the model, not with the paper.

> **Note added 2026-08-24.** The paragraph above was written *before* the
> replication was attempted, and is left unedited as the pre-registered
> criterion. The outcome was mixed: signature (f) reproduces, signature (g) does
> not. The two `[FAIL]` lines in `results/validation.txt` are those (g) checks,
> and they are expected rather than a regression. The criterion's conclusion —
> that a failure implicates our model rather than the source — was not sustained;
> see *Assessment* at the end of this file for why the published description is
> judged insufficient to reproduce, and note the standing constraint that nothing
> in our model may be tuned to match it.

### Parameters the paper does not state — must be inferred and documented

These are genuine gaps. Each needs a documented choice plus a sensitivity check
showing the acceptance test is robust to it:

1. **Number of scientists** `N`, and the **G1:G2 split**. Not given in the text
   we extracted. Case (a)'s overall funding rate of 30.2% together with the group
   success rates constrains the ratio — solve for it rather than guess.
2. **Simulation length and burn-in.** Grants last 3 years (6 time units), and G1
   withdraws while funded, so the system needs enough rounds to reach a stationary
   state. No horizon is stated.
3. **Number of replicates** behind the reported means, and whether the reported
   figures are per-replicate means or pooled.
4. **Year-end "highest-rated"** — rated by whom? Presumably the program officer's
   ranking (`Qp` for correct, `Qs` for reputation-based), but unstated.
5. **Cooling-off trigger condition** in case (g). Described only as "12-month
   prohibition following specific underperformance metrics", modelled on EPSRC.
   The EPSRC RUA rule is: 3+ proposals in the bottom half or rejected before
   panel within 24 months, **and** personal success rate < 25%. Use that, and
   document it.
6. **Selfish reviewer threshold**: "below 90% of minimum threshold" — 90% of 110
   is 99, which is *below* the population mean of 100. Confirm this reading; it
   makes selfish reviewers decline almost everything above their own quality and
   accept a band below it, which is a strange but reproducible rule.

### Decision required — see design.md open question 1

Either implement R&S as a **standalone module** (`validation/roebber_schultz.py`,
its own agent loop, our shared RNG and metrics utilities only), or make our model
a **superset** whose degenerate configuration reproduces theirs.

Trade-off: the superset gives a much stronger sanity check ("our engine, their
parameters, their numbers") but drags R&S's idiosyncrasies — unanimity voting,
two fixed behavioural groups, no continuous ranking — into our design. The
standalone module is cleaner but only validates the shared utilities.

## Target 2 — Graves, Barnett & Clarke (2011), BMJ 343:d4797

Not a replication — a **calibration target**. Bootstrap-resample simulated panel
member scores, re-rank, re-apply the funding line, and classify proposals as
always / sometimes / never funded. Solve for the reviewer-noise level that
reproduces:

- **9%** always funded
- **29%** sometimes funded
- **61%** never funded
- equivalently: **59%** of funded grants are sometimes not funded

This replaces an arbitrary noise parameter with a measured one, and it is the
methodological answer to review comment W1 on the anchor paper.

## Target 3 — Gross & Bergstrom (2019), contest model

Cheap analytic cross-check rather than a simulation replication: at a given
payline, the equilibrium proposal-preparation effort predicted by the contest
model should match what our applicants do when the effort-offset parameter is
configured to make them best-responders. Confirms the effort/welfare accounting
is not mis-scaled.

Also worth reproducing in passing: Herbert (2013)'s system-level figure —
**550 working years, ≈14% of scheme budget** — should fall out of our welfare
accounting when calibrated to 34–38 days per proposal at a comparable scheme size.

## Non-targets

- The anchor paper's own ABM. **No code exists** and its parameters were chosen
  for visibility rather than calibration, so it is not a validation target. Where
  its qualitative findings overlap (caps concentrate effort rather than removing
  it) we note agreement, but we do not tune to match it.

---

## Replication status — 2026-08-04

### Resolved by reading the paper directly (not the summarised extraction)

1. **Threshold is 105, not 110.** The prose says the correct reviewer funds "the
   top 16% of all proposals or at least one standard deviation above the mean",
   which reads as 110. **Figure 2 shows the implemented test as `Qp >= 105`**, and
   only 105 reproduces the case (a) funding rate: `Qp ~ N(100, 11.18)`, so
   `P(Qp >= 105) = 0.33` against `P(Qp >= 110) = 0.19` versus a reported 30.2%.
   The figures are authoritative where they disagree with the text.
2. **Groups are evenly sized** — stated explicitly ("two evenly sized groups"),
   so `g2_fraction = 0.5`. `solve_group_split()` is retained as a consistency
   check on the reported table, not as the source of the value.
3. **G1 submits once per year, half-year chosen at random** (Fig. 1: "already
   submitted one proposal this year?"), not deterministically on even steps.
4. **G1 resumes in the final year of an award**, not at expiry ("do not submit
   new proposals until the final year of the grant").
5. **Horizon is 2000 time steps**, stated.
6. Selfish rule confirmed by Fig. 2 as `95 <= Qp <= Qs(R)`, consistent with
   "90% of the defined minimum threshold" at a threshold of 105.

### Case (a) — replicates

| metric | paper | model |
|---|---|---|
| overall funding rate | 30.2% | **30.25%** |
| G2 share of funding | 76.9% | **76.7%** |
| mean funded quality G1 | 111.6 | **111.8** |
| mean funded quality G2 | 112.1 | **112.3** |
| G1 success | 21.3% | 24.5% |
| G2 success | 32.6% | 32.6% |

The funding rate and funding share were not used to fit anything, so this is a
genuine out-of-sample match and it confirms the threshold inference.

### Case (g) — success rates replicate

G1 11.9% (paper 12.5%), G2 17.0% (paper 17.4%). Funding share is off
(69.0% vs 57.7%).

### Cases (b)–(f) — DO NOT replicate

The central discrepancy: in the paper, introducing non-correct reviewers collapses
G2's funding share from 76.9% to 44.8%. In our implementation it barely moves
(76.7% -> 73.4%). Every downstream case inherits the error, because all of
(b)–(g) build on the baseline reviewer mix.

**Leading hypothesis — the budget rule.** The paper says the officer "uses that
rate to estimate the number of fundable proposals and then funds proposals with
unanimously positive recommendations up to half that limit". We currently read the
limit as `target_rate x proposals_this_half_year`, which at baseline is ~112 per
half-year and is never binding (only ~5-15% of proposals clear unanimity). If the
limit is instead `target_rate x N_scientists` (150/year, 75/half-year), the budget
binds hard, the officer's ranking starts to matter, and competition among the
high-quality persistent G2 submitters could produce the reported collapse.

**Candidates to test, in order:**

1. Budget base: proposals vs scientists vs a fixed grant count.
2. `n_reviewers`: Fig. 4 shows the correct+unanimous G2 share at ~50% for K=4 and
   ~42.5% for K=5, bracketing the reported 44.8%. Our K=4 gives 73.4%, so K alone
   does not explain the gap, but it must be swept jointly with the budget rule.
3. Whether "success" in Table 1 is per-proposal or per-scientist-per-year. Our
   per-proposal reading reproduces case (a)'s funding share exactly, which is
   evidence for it, but it should be checked against the baseline once the budget
   rule is settled.
4. Whether the year-end surplus rule reconsiders only the second half-year's
   proposals (as implemented) or the whole year's.

**Not yet a concern:** the qualitative signature for case (f) already holds — the
cap lowers the untargeted group's funded quality (113.1 -> 111.7) and cuts the
targeted group's share (73.4% -> 50.4%). The case (g) inversion does **not** yet
reproduce, and that one matters most for our H6, so it must be resolved before
any claim rests on it.

## Replication status update — 2026-08-04 (second pass)

### Corrected

**Cooling-off trigger.** The rule was implemented as "any unsuccessful proposal
counts as a strike". Page 6 of the paper gives EPSRC's actual rule: barred 12
months if, within 24 months, the applicant had **at least three proposals ranked
in the bottom half of a funding prioritisation list** (or rejected before panel)
**and** an overall success rate below 25%. An unfunded proposal ranked in the top
half does **not** count. Now implemented as a median split on the officer's
ranking among unfunded proposals. Consequence: the trigger is much weaker, and
reviewer-burden reduction falls from ~20% to ~13% — further from the paper's
reported 34%, which is informative rather than a regression (see below).

### Ruled out

Neither candidate hypothesis explains the baseline failure:

| budget_base | K=2 | K=3 | K=4 | K=5 |
|---|---|---|---|---|
| proposals | 74.8% | 73.9% | 74.2% | 74.8% |
| scientists | 75.8% | 75.6% | 75.2% | 75.0% |

(G2 funding share; paper's case (b) reports **44.8%**.)

`budget_base="scientists"` *does* fix one thing — it makes the budget bind, and
G1's mean funded quality rises to 115.2 against the paper's 115.0. So that
interpretation is probably correct and should be kept. But it does not move the
share.

**The decisive clue is that `n_reviewers` has essentially no effect in our
implementation**, while the paper's Figure 4 shows the correct+unanimous G2 share
falling steeply with K (≈67% at K=2 to ≈42.5% at K=5). A parameter that drives
their headline figure and does nothing in ours means the review stage is
structurally different, not mis-tuned.

### The unresolved asymmetry

For G2's per-proposal success to fall below G1's, G2's *submitted* proposals must
be systematically lower quality than G1's. Our model produces the opposite
ordering by construction: G1 pauses while funded, so G1's submitting pool is
**depleted of its high-quality members**, while G2's is not. That correctly
reproduces case (a), where G2 outperforms G1 (32.6% vs 24.5%, paper 34.6% vs
21.3%). It cannot produce case (b), where the ordering reverses.

The paper's own explanation — "the presence of non-correct reviewers retards the
effectiveness of the G2 strategy" — does not identify a channel that treats the
two groups differently, since both draw reviewers from the same pool and both have
identical quality distributions ("these characteristics are identical in both
groups").

Remaining candidates, none yet tested:

1. **Order of consideration within a half-year.** Figure 3 shows ranking only on
   the deferred/year-end branch; the in-year branch reads as first-come rather
   than ranked. If proposals are considered in arrival order and G1/G2 arrival
   patterns differ (G1 submits once a year in a randomly chosen half, G2 every
   half), a systematic ordering advantage could exist.
2. **Per-scientist review load feeding back into reviewer type.** Page 5 says the
   effect of increasing reviewer load "can be discerned from these results — as
   correct reviewers are converted into harried ones by this load", described as
   *not explicitly modelled*. If it was in fact implemented, G2's volume would
   degrade the reviewer pool that judges G2.
3. **Inheritance from Thurner & Hanel (2010)**, which the paper says the model was
   "modified from". That model may carry structure the paper does not restate.

**Assessment.** Case (a) replicates out-of-sample and the demand-management
signature for case (f) reproduces qualitatively. The baseline reversal does not,
and may not be recoverable from the published description alone. This is worth
reporting in the paper as a replication finding in its own right: the only prior
ABM of proposal-limiting policy cannot be fully reproduced from its own methods
section. **Nothing in our own model should be tuned to match it, and H6 must not
lean on case (g) until this resolves.**

---

## Status of the automated checks — 2026-08-24

`python -m validation.report` ends with four qualitative assertions, committed as
`results/validation.txt`. Two pass and two fail, and the split is the finding:

| check | status | reading |
|---|---|---|
| (f) cap lowers untargeted group's funded quality | PASS | 113.1 -> 111.7 |
| (f) cap cuts targeted group's funding share | PASS | 73.4% -> 50.4% |
| (g) cooling-off raises high-volume group's share | FAIL | 71.9%, direction not reproduced |
| (g) cooling-off roughly halves G1 success | FAIL | 11.8% vs 13.0%, magnitude not reproduced |

The (f) pair is the demand-management signature this project actually depends on,
and it holds. The (g) pair rests on the baseline reversal analysed above, which
is not recoverable from the published methods section. These two failures are
therefore left standing deliberately: they are the replication finding, not a bug
to be fixed, and suppressing them would misrepresent the result. They must not be
made to pass by tuning.
