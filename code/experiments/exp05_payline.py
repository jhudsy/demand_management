"""Payline sensitivity, and the decomposition that explains every result:

    net effect of a cap  =  triage value  -  deletion loss

where triage value is the gap to a volume-matched random placebo, and deletion
loss is what the placebo loses against no cap. Both are measured, not assumed,
and both are reported here as signed quantities: the deletion-loss column is
negative because it is a loss. The write-up states the same identity with both
terms as positive magnitudes."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataclasses import replace
from dm import triage
from dm.runner import replicate
from dm.scenarios import scenarios
from dm.evaluation import SIGMA_EVAL
from dm.simulate import SimConfig

SIGMA, SEEDS = SIGMA_EVAL, 12
MECHS = {
    "individual K=1":     triage.IndividualCap(k=1, sigma_self=SIGMA),
    "institutional M=30": triage.InstitutionalCap(m=30, sigma_inst=SIGMA),
    "cooling off":        triage.CoolingOff(),
    "resub limit 1":      triage.ResubmissionLimit(max_resubmissions=1),
}

cfg = scenarios()["S4_big_and_good"]
print("S4_big_and_good — decomposition by payline (grants per reference proposal)\n")
for payline in (0.05, 0.10, 0.15, 0.25):
    sim = replace(SimConfig(sigma_eval=SIGMA, n_rounds=20, burn_in=5), payline=payline)
    base = replicate(cfg, sim, triage.NoCap(), n_seeds=SEEDS)
    b = base["mean_quality_funded"].mean
    print(f"--- payline {payline:.0%}  (no cap: Qfund {b:.3f}, success {base['success_rate'].mean:.1%}) ---")
    print(f"    {'mechanism':20s} {'Qfund':>7s} {'deletion loss':>14s} {'triage value':>13s} {'NET':>8s} {'succ%':>7s}")
    for name, rule in MECHS.items():
        e = replicate(cfg, sim, rule, n_seeds=SEEDS)
        keep = e["submitted"].mean / max(e["intended"].mean, 1)
        pl = replicate(cfg, sim, triage.RandomThinning(keep_share=keep), n_seeds=SEEDS)
        q, p = e["mean_quality_funded"].mean, pl["mean_quality_funded"].mean
        print(f"    {name:20s} {q:7.3f} {p-b:14.3f} {q-p:+13.3f} {q-b:+8.3f} "
              f"{e['success_rate'].mean:6.1%}")
    print()
