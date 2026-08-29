"""A6 — what the no-CoT bias actually looks like in magnitude terms.

The write-up currently calls the thinking-off effect "a prior on the answer". That is a placeholder
for a mechanism, and this block replaces it with a description the data can support: with no chain
of thought the model appears to emit the THRESHOLD and choose a SIDE, rather than producing an
independent estimate that happens to land favourably.

Measures, per arm, on the existing thinking-off rollouts:
  - signed log10(estimate / T), signed so that POSITIVE = toward the favoured side for that arm
    (above_good favours >T, below_good favours <=T). Baseline has no favoured side, so it is
    reported unsigned as log10(estimate / T).
  - fraction landing within +-10% and +-25% of T
  - the sign split, P(estimate > T)

UNFILTERED throughout. The [T/10, 10T] outlier filter is calibrated to the thinking-ON baseline,
and the no-think baseline sits 5.6x above T; filtering removes 68/150 of it (see hypotheses.md)
and would manufacture agreement between arms. Filtered numbers are printed alongside so the
difference is visible rather than assumed.

Run with: python3 runs/a6_nothink_magnitude.py   (no GPU)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from src.qual import parse                                     # noqa: E402

RUN = HERE / "thinking_off"
ARMS = {"above_good": "nothink_above_good", "below_good": "nothink_below_good",
        "baseline": "nothink_baseline"}


def main():
    T = json.load(open(HERE / "hour0" / "threshold.json"))["threshold"]
    res = {"threshold": T, "arms": {}}

    print(f"threshold T = {T:.3e}\n")
    print(f"{'arm':12s}{'n':>5s}{'med signed':>12s}{'|.|<=10%':>10s}{'|.|<=25%':>10s}"
          f"{'P(>T)':>8s}{'P(fav)':>9s}{'med est':>12s}")
    for arm, stem in ARMS.items():
        recs = list(parse.parse_jsonl_file(RUN / "raw" / f"{stem}.jsonl"))
        vals = []
        for r in recs:
            v, _ = parse.parse_estimate(r["raw_completion"])
            if v is not None and v > 0:
                vals.append(v)
        v = np.array(vals, dtype=float)
        ratio = v / T
        lg = np.log10(ratio)
        signed = lg if arm != "below_good" else -lg
        within10 = float(np.mean(np.abs(ratio - 1) <= 0.10))
        within25 = float(np.mean(np.abs(ratio - 1) <= 0.25))
        p_above = float(np.mean(v > T))
        p_fav = p_above if arm == "above_good" else (float(np.mean(v <= T))
                                                     if arm == "below_good" else float("nan"))
        keep = (v >= T / 10) & (v <= 10 * T)
        res["arms"][arm] = {
            "n": len(v), "median_signed_log10": float(np.median(signed)),
            "median_estimate": float(np.median(v)),
            "frac_within_10pct": within10, "frac_within_25pct": within25,
            "p_above_T": p_above, "p_favoured": p_fav,
            "iqr_signed_log10": float(np.subtract(*np.percentile(signed, [75, 25]))),
            "filtered": {"n": int(keep.sum()),
                         "frac_within_10pct": float(np.mean(np.abs(ratio[keep] - 1) <= 0.10)),
                         "median_estimate": float(np.median(v[keep]))},
            "deciles_signed_log10": [float(x) for x in np.percentile(signed, np.arange(0, 101, 10))]}
        print(f"{arm:12s}{len(v):5d}{np.median(signed):+12.3f}{within10:10.3f}{within25:10.3f}"
              f"{p_above:8.3f}{p_fav:9.3f}{np.median(v):12.3e}")

    print("\n  (baseline's 'signed' column is unsigned log10(est/T); it has no favoured side)")

    print("\n=== how concentrated is the no-CoT answer on T itself? ===")
    for arm in ARMS:
        a = res["arms"][arm]
        print(f"  {arm:12s} within 10% of T: {a['frac_within_10pct']:.3f}   "
              f"within 25%: {a['frac_within_25pct']:.3f}   "
              f"median {a['median_estimate']:.3e}   IQR(signed log10) {a['iqr_signed_log10']:.3f}")
    print(f"\n  reference: the framing-controls coin arm put 54% within +-10% of T with no stake "
          f"at all,\n  and its median estimate was 75,000,001 — one above the threshold.")

    print("\n=== reading ===")
    ag, bg = res["arms"]["above_good"], res["arms"]["below_good"]
    print(f"  above_good P(>T) = {ag['p_above_T']:.3f}, below_good P(>T) = {bg['p_above_T']:.3f}")
    print(f"  both arms' estimates sit within 25% of T in {ag['frac_within_25pct']:.0%} / "
          f"{bg['frac_within_25pct']:.0%} of rollouts, against the no-bet baseline's "
          f"{res['arms']['baseline']['frac_within_25pct']:.0%}")

    out = RUN / "a6_magnitude.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
