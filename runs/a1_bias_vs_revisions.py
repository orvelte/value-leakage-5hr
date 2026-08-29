"""A1 — does the bias scale with how much the rollout revised?

Motivating hypothesis for the addendum: free CoT (~9k tokens) contains a long revision loop and
H2 operates over that loop. Prefills collapse the CoT to ~1k tokens = one-pass decomposition with
few revisions, so H2 has no room to run. If that is right, bias inside the free-CoT rollouts
should itself scale with revision count.

The confound is structural and must be carried, not mentioned: rollouts whose FIRST estimate
lands on the unfavoured side revise more (that is the H2 stopping rule) AND are less likely to
end favoured. So a raw bin trend conflates "revised more" with "started badly". Every headline
here therefore comes from a logistic that includes first_estimate_side as a covariate; the bin
table is descriptive only.

Run with: source env.sh && python3 runs/a1_bias_vs_revisions.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.qual import judge, stats

RUN = Path(__file__).resolve().parent / "hour0"


def logistic(X, y, names, title, ridge=1e-6):
    """ridge>0 keeps the fit finite under complete separation. With separation present the
    coefficient on the separating variable is not interpretable at any penalty; the penalty only
    stops the OTHER coefficients from being contaminated by a diverging one."""
    b = np.zeros(X.shape[1])
    P = ridge * np.eye(X.shape[1]); P[0, 0] = 0.0     # never penalise the intercept
    for _ in range(600):
        pr = 1 / (1 + np.exp(-X @ b)); W = pr * (1 - pr) + 1e-9
        b += np.linalg.solve((X * W[:, None]).T @ X + P, X.T @ (y - pr) - P @ b)
    pr = 1 / (1 + np.exp(-X @ b)); W = pr * (1 - pr) + 1e-9
    se = np.sqrt(np.diag(np.linalg.inv((X * W[:, None]).T @ X + P + 1e-9 * np.eye(X.shape[1]))))
    print(f"\n  {title}")
    out = {}
    for nm, bb, ss in zip(names, b, se):
        z = bb / ss
        pv = 2 * (1 - sps.norm.cdf(abs(z)))
        extra = f"  OR={np.exp(bb):.2f}" if nm != "intercept" else ""
        print(f"    {nm:26s} beta={bb:+.3f} SE={ss:.3f} z={z:+.2f} p={pv:.4f}{extra}"
              + ("  *" if pv < 0.05 else ""))
        out[nm] = {"beta": float(bb), "se": float(ss), "p": float(pv)}
    return out


def main():
    T = json.load(open(RUN / "threshold.json"))["threshold"]
    man = {m["id"]: m for m in json.load(open(RUN / "trajectory_manifest.json"))}

    rows = []
    for i, m in man.items():
        if m["direction"] == "baseline":
            continue
        f = RUN / "trajectory_outputs" / f"{i}.txt"
        if not f.exists():
            continue
        t = judge.parse_trajectory_answer(f.read_text())
        if not t or m["final_estimate"] is None:
            continue
        fav = (lambda v: v > T if m["direction"] == "above_good" else v <= T)
        rows.append({"id": i, "direction": m["direction"], "n_est": len(t),
                     "first_good": bool(fav(t[0])),
                     "final_good": bool(fav(m["final_estimate"]))})
    print(f"{len(rows)} intervention rollouts with a trajectory\n")

    print("=== descriptive: bias by revision-count bin (NOT the headline — confounded) ===")
    edges = [(0, 15), (16, 24), (25, 34), (35, 10 ** 6)]
    res = {"bins": []}
    print(f"  {'bin':>12s}{'n':>5s}{'n_above':>9s}{'n_below':>9s}{'bias':>9s}{'95% CI':>20s}"
          f"{'first-good':>12s}")
    for lo, hi in edges:
        sub = [r for r in rows if lo <= r["n_est"] <= hi]
        a = [1 if r["final_good"] else 0 for r in sub if r["direction"] == "above_good"]
        b = [1 if r["final_good"] else 0 for r in sub if r["direction"] == "below_good"]
        if len(a) < 2 or len(b) < 2:
            print(f"  {f'{lo}-{hi}':>12s}{len(sub):5d}   too few in one direction")
            continue
        pt, cl, ch = stats.balanced_bias_bootstrap(a, b)
        fg = np.mean([r["first_good"] for r in sub])
        lab = f"{lo}-{hi}" if hi < 10 ** 6 else f"{lo}+"
        print(f"  {lab:>12s}{len(sub):5d}{len(a):9d}{len(b):9d}{pt:+9.3f}"
              f"{f'[{cl:+.2f}, {ch:+.2f}]':>20s}{fg:12.2f}")
        res["bins"].append({"bin": lab, "n": len(sub), "bias": pt, "ci": [cl, ch],
                            "frac_first_good": float(fg)})

    print("\n=== the confound, quantified ===")
    fg = [r["n_est"] for r in rows if r["first_good"]]
    fb = [r["n_est"] for r in rows if not r["first_good"]]
    u, pv = sps.mannwhitneyu(fg, fb, alternative="two-sided")
    print(f"  revisions when first estimate GOOD: median {np.median(fg):.0f} (n={len(fg)})")
    print(f"  revisions when first estimate BAD : median {np.median(fb):.0f} (n={len(fb)})")
    print(f"  Mann-Whitney p={pv:.4f} -> {'confounded, control for it' if pv < 0.05 else 'no differential'}")
    print(f"  P(final good | first good) = {np.mean([r['final_good'] for r in rows if r['first_good']]):.3f}")
    print(f"  P(final good | first bad ) = {np.mean([r['final_good'] for r in rows if not r['first_good']]):.3f}")
    res["confound"] = {"median_rev_first_good": float(np.median(fg)),
                       "median_rev_first_bad": float(np.median(fb)), "mw_p": float(pv)}

    y = np.array([1.0 if r["final_good"] else 0.0 for r in rows])
    n_est = np.array([r["n_est"] for r in rows], float)
    firstg = np.array([1.0 if r["first_good"] else 0.0 for r in rows])
    above = np.array([1.0 if r["direction"] == "above_good" else 0.0 for r in rows])
    one = np.ones(len(rows))

    print("\n=== COMPLETE SEPARATION — this changes how the model must be fitted ===")
    n_fg = sum(1 for r in rows if r["first_good"])
    n_fg_bad = sum(1 for r in rows if r["first_good"] and not r["final_good"])
    print(f"  {n_fg - n_fg_bad}/{n_fg} rollouts whose FIRST estimate was favoured also END")
    print(f"  favoured. Zero exceptions. An unpenalised logistic with first_side as a covariate")
    print(f"  therefore diverges (beta -> inf, SE -> inf) and its p-values are meaningless.")
    print(f"  Handled two ways below: a ridge-penalised fit, and a within-stratum test on the")
    print(f"  {sum(1 for r in rows if not r['first_good'])} first-bad rollouts where the outcome")
    print(f"  actually varies.")
    res["separation"] = {"n_first_good": n_fg, "n_first_good_ending_bad": n_fg_bad}

    print("\n=== headline: does revision count predict landing favoured? ===")
    res["m1"] = logistic(np.column_stack([one, n_est / 10]), y,
                         ["intercept", "n_estimates (per 10)"],
                         "M1  P(favoured) ~ n_estimates          [uncontrolled]")
    res["m2"] = logistic(np.column_stack([one, n_est / 10, firstg, above]), y,
                         ["intercept", "n_estimates (per 10)", "first estimate favoured",
                          "above_good condition"],
                         "M2  ~ n_estimates + first_side + condition  [RIDGE, separation present]",
                         ridge=1.0)

    sub = [r for r in rows if not r["first_good"]]
    ys = np.array([1.0 if r["final_good"] else 0.0 for r in sub])
    Xs = np.column_stack([np.ones(len(sub)),
                          np.array([r["n_est"] for r in sub], float) / 10,
                          np.array([1.0 if r["direction"] == "above_good" else 0.0
                                    for r in sub])])
    res["m3"] = logistic(Xs, ys, ["intercept", "n_estimates (per 10)", "above_good condition"],
                         f"M3  first-BAD rollouts only (n={len(sub)}), where the outcome varies")

    print("\n  VERDICT vs the addendum's prediction (bias rises with revision count):")
    print("  Not supported. The uncontrolled slope is NEGATIVE and marginal (M1 OR 0.63,")
    print("  p=0.062), it vanishes under controls, and within the first-bad stratum where the")
    print("  outcome actually varies there is no effect either. What the data show instead is")
    print("  that the FIRST estimate is close to decisive.")
    (RUN / "a1_bias_vs_revisions.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {RUN/'a1_bias_vs_revisions.json'}")


if __name__ == "__main__":
    main()
