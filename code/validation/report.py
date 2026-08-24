"""Print the Roebber & Schultz acceptance-test comparison table."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validation.roebber_schultz import PUBLISHED, run, scenarios  # noqa: E402

FIELDS = [
    ("g1_success", "G1 succ", True),
    ("g2_success", "G2 succ", True),
    ("g1_funded_quality", "Qbar G1", False),
    ("g2_funded_quality", "Qbar G2", False),
    ("g2_funding_share", "G2 share", True),
]


def main() -> None:
    print(f"{'case':22s} {'metric':10s} {'paper':>8s} {'model':>8s} {'diff':>8s}")
    print("-" * 60)
    baseline = None
    for name, cfg in scenarios().items():
        got = run(cfg)
        want = PUBLISHED[name]
        if name == "b_baseline":
            baseline = got
        for key, label, pct in FIELDS:
            p, m = want[key], got[key]
            if pct:
                print(
                    f"{name:22s} {label:10s} {p*100:7.1f}% {m*100:7.1f}% "
                    f"{(m-p)*100:+7.1f}"
                )
            else:
                print(
                    f"{name:22s} {label:10s} {p:8.1f} {m:8.1f} {m-p:+8.1f}"
                )
        print(
            f"{name:22s} {'reviews':10s} {'':>8s} "
            f"{got['reviews_performed']:8.0f}"
            + (
                f" {100*(got['reviews_performed']/baseline['reviews_performed']-1):+7.1f}%"
                if baseline and name != "b_baseline"
                else ""
            )
        )
        print("-" * 60)

    print("\nQualitative signatures that must hold (see notes/validation.md):")
    res = {n: run(c) for n, c in scenarios().items()}
    b, f, g = res["b_baseline"], res["f_g2_one_grant"], res["g_cooling_off"]
    checks = [
        (
            "(f) cap lowers UNTARGETED group's funded quality",
            f["g1_funded_quality"] < b["g1_funded_quality"],
            f"{b['g1_funded_quality']:.1f} -> {f['g1_funded_quality']:.1f}",
        ),
        (
            "(f) cap cuts targeted group's funding share",
            f["g2_funding_share"] < b["g2_funding_share"],
            f"{b['g2_funding_share']*100:.1f}% -> {f['g2_funding_share']*100:.1f}%",
        ),
        (
            "(g) cooling-off RAISES high-volume group's share",
            g["g2_funding_share"] > b["g2_funding_share"],
            f"{b['g2_funding_share']*100:.1f}% -> {g['g2_funding_share']*100:.1f}%",
        ),
        (
            "(g) cooling-off roughly halves G1 success",
            g["g1_success"] < 0.75 * b["g1_success"],
            f"{b['g1_success']*100:.1f}% -> {g['g1_success']*100:.1f}%",
        ),
    ]
    for desc, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc:52s} {detail}")


if __name__ == "__main__":
    main()
