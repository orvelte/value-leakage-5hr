"""A13 — the null for the ratchet: how often does an UNBIASED trace cross the threshold?

A12.1 reported 41/41 (first favoured -> ended favoured, no reversals) and 0.130 recovery for
covert rollouts that started unfavoured. Neither is interpretable on its own: if baseline traces
essentially never cross T after their first number, then 41/41 is just "traces don't cross" and
0.130 is normal, not suppressed.

Construction follows the stopping-rule null exactly: the 30 baseline trajectories are scored
under each FIXED framing separately. Scoring them under both at once counts every rollout twice
— once as favoured, once as unfavoured — and forces the rate to 0.5 by arithmetic. That is the
vacuous null recorded in hypotheses.md.

One thing to be explicit about, because it changes how the two framings should be read: they are
complementary by construction. "Favoured" under above_good is >T and under below_good is <=T, so
the two framings partition the SAME 2x2 crossing structure two ways. The framing-free transition
matrix is reported first for that reason; the framed views are relabelings of it, not two
independent measurements.

Run with: python3 runs/a13_baseline_crossing_null.py   (no GPU)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from src.qual import judge                                     # noqa: E402

RUN = HERE / "hour0"
# A12.1 reference numbers, for the comparisons at the end
REF = {"all_unfav": (14, 39), "overt_unfav": (11, 16), "covert_unfav": (3, 23),
       "all_fav": (41, 41)}


def wilson(k, n):
    if n == 0:
        return float("nan"), (float("nan"), float("nan"))
    lo, hi = sps.binomtest(int(k), int(n), 0.5).proportion_ci(0.95)
    return k / n, (float(lo), float(hi))


def main():
    T = json.load(open(RUN / "threshold.json"))["threshold"]
    man = {m["id"]: m for m in json.load(open(RUN / "trajectory_manifest.json"))}

    trajs = []
    for iid, m in man.items():
        if m["direction"] != "baseline":
            continue
        f = RUN / "trajectory_outputs" / f"{iid}.txt"
        if not f.exists():
            continue
        t = judge.parse_trajectory_answer(f.read_text())
        t = [v for v in t or [] if v != 0]          # A12.4 rule
        if t:
            trajs.append({"id": iid, "traj": t})

    print(f"baseline trajectories: {len(trajs)}  (threshold T = {T:,.0f})\n")

    # --- framing-free transition matrix on raw side ---
    above = lambda v: v > T
    cells = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}
    for r in trajs:
        cells[(above(r["traj"][0]), above(r["traj"][-1]))] += 1
    n_start_above = cells[(True, True)] + cells[(True, False)]
    n_start_below = cells[(False, True)] + cells[(False, False)]
    print("=== framing-free transition matrix (no bet, so no favoured side) ===")
    print(f"  {'':18s}{'ends >T':>10s}{'ends <=T':>10s}")
    print(f"  {'starts >T':18s}{cells[(True,True)]:10d}{cells[(True,False)]:10d}")
    print(f"  {'starts <=T':18s}{cells[(False,True)]:10d}{cells[(False,False)]:10d}")
    p_stay_above, ci_sa = wilson(cells[(True, True)], n_start_above)
    p_cross_up, ci_cu = wilson(cells[(False, True)], n_start_below)
    print(f"\n  P(ends >T | starts >T)  = {p_stay_above:.3f}  [{ci_sa[0]:.3f}, {ci_sa[1]:.3f}]"
          f"   n={n_start_above}")
    print(f"  P(ends >T | starts <=T) = {p_cross_up:.3f}  [{ci_cu[0]:.3f}, {ci_cu[1]:.3f}]"
          f"   n={n_start_below}")

    # --- how much crossing happens at all, anywhere in the trace ---
    n_cross, any_cross = [], 0
    for r in trajs:
        sides = [above(v) for v in r["traj"]]
        c = sum(1 for a, b in zip(sides, sides[1:]) if a != b)
        n_cross.append(c)
        any_cross += c > 0
    print(f"\n=== crossing anywhere in the trace ===")
    print(f"  trajectories with >=1 crossing: {any_cross}/{len(trajs)} "
          f"({any_cross/len(trajs):.3f});  median crossings {np.median(n_cross):.0f}, "
          f"max {max(n_cross)}")
    print(f"  so an unbiased trace DOES move across T during reasoning; the question is where "
          f"it settles")

    res = {"threshold": T, "n": len(trajs),
           "transition": {f"{a}->{b}": v for (a, b), v in cells.items()},
           "p_end_above_given_start_above": p_stay_above,
           "p_end_above_given_start_below": p_cross_up,
           "frac_with_any_crossing": any_cross / len(trajs),
           "median_crossings": float(np.median(n_cross)), "framings": {}}

    # --- the two fixed framings ---
    print(f"\n=== scored under each FIXED framing (never both at once) ===")
    print(f"  {'framing':14s}{'first estimate':18s}{'n':>4s}{'P(final favoured)':>20s}{'95% CI':>20s}")
    for framing in ("above_good", "below_good"):
        fav = (lambda v: v > T) if framing == "above_good" else (lambda v: v <= T)
        row = {}
        for label, want in (("favoured", True), ("unfavoured", False)):
            sub = [r for r in trajs if fav(r["traj"][0]) is want]
            k = sum(fav(r["traj"][-1]) for r in sub)
            p, ci = wilson(k, len(sub))
            row[label] = {"n": len(sub), "k": int(k), "p": p, "ci": list(ci)}
            print(f"  {framing:14s}{label:18s}{len(sub):4d}{p:20.3f}"
                  f"{f'[{ci[0]:.3f}, {ci[1]:.3f}]':>20s}")
        res["framings"][framing] = row
    print("\n  (the two framings are complementary relabelings of the matrix above, not two "
          "independent tests)")

    # --- the comparisons the headline rests on ---
    print(f"\n=== against the intervention numbers (A12.1) ===")
    base_unfav = [(res['framings'][f]['unfavoured']['k'],
                   res['framings'][f]['unfavoured']['n']) for f in ("above_good", "below_good")]
    base_fav = [(res['framings'][f]['favoured']['k'],
                 res['framings'][f]['favoured']['n']) for f in ("above_good", "below_good")]
    # pool the two framings for the comparison: each rollout contributes once per framing, and
    # the framings partition it differently, so report BOTH rather than a pooled single number
    res["comparisons"] = {}
    for name, (k, n) in REF.items():
        print(f"  {name:14s} {k}/{n} = {k/n:.3f}")
    for i, framing in enumerate(("above_good", "below_good")):
        ku, nu = base_unfav[i]
        kf, nf = base_fav[i]
        for ref_name in ("covert_unfav", "overt_unfav"):
            rk, rn = REF[ref_name]
            odds, p = sps.fisher_exact([[rk, rn - rk], [ku, nu - ku]])
            res["comparisons"][f"{ref_name} vs baseline[{framing}] (started unfavoured)"] = {
                "ref": rk / rn, "baseline": ku / nu if nu else float("nan"),
                "fisher_or": float(odds), "fisher_p": float(p)}
            print(f"  {ref_name} {rk/rn:.3f} vs baseline[{framing}] unfavoured "
                  f"{ku}/{nu}={ku/nu if nu else float('nan'):.3f}  Fisher p={p:.4f}")
        rk, rn = REF["all_fav"]
        odds, p = sps.fisher_exact([[rk, rn - rk], [kf, nf - kf]])
        res["comparisons"][f"all_fav vs baseline[{framing}] (started favoured)"] = {
            "ref": rk / rn, "baseline": kf / nf if nf else float("nan"),
            "fisher_or": float(odds), "fisher_p": float(p)}
        print(f"  no-reversal 41/41=1.000 vs baseline[{framing}] favoured "
              f"{kf}/{nf}={kf/nf if nf else float('nan'):.3f}  Fisher p={p:.4f}")

    out = RUN / "a13_baseline_crossing_null.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
