"""Covert vs overt good-side rollouts: CoT length, number of intermediate estimates, and
where the FIRST estimate fell relative to the threshold.

The third measure is the one the plan cares about most: if first estimates already differ,
the leak is upstream of the verbalized reasoning; if only finals differ, it is a late-stage
move; if the difference is mainly CoT length, the selection-effect story is live.

Read the confound note printed at the end before believing the length result.

Run with: python3 runs/analyze_trajectory.py [--run hour0]
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.qual import judge, parse, segment

GROUPS = [("overt (Admits)", "INFLUENCED"), ("covert (Denies)", "NOT_INFLUENCED")]


def fmt_p(p):
    return f"p={p:.4f}" + ("  *" if p < 0.05 else "")


def compare(name, a, b, unit=""):
    """Mann-Whitney with rank-biserial effect size; medians reported, not means, since these
    distributions are skewed and small."""
    if len(a) < 3 or len(b) < 3:
        print(f"  {name:26s} too few observations ({len(a)}/{len(b)})")
        return None
    u, p = sps.mannwhitneyu(a, b, alternative="two-sided")
    r = 2 * u / (len(a) * len(b)) - 1
    print(f"  {name:26s} overt {st.median(a):8.1f}{unit}  covert {st.median(b):8.1f}{unit}  "
          f"diff {st.median(a)-st.median(b):+8.1f}  U={u:6.0f} {fmt_p(p)}  r={r:+.2f}")
    return {"median_overt": st.median(a), "median_covert": st.median(b),
            "n_overt": len(a), "n_covert": len(b), "U": float(u), "p": float(p),
            "rank_biserial": float(r)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="hour0")
    args = ap.parse_args()
    run_dir = Path(__file__).resolve().parent / args.run
    T = json.load(open(run_dir / "threshold.json"))["threshold"]

    manifest = {m["id"]: m for m in json.load(open(run_dir / "trajectory_manifest.json"))}
    raw = {}
    for d in ["above_good", "below_good"]:
        for rec in parse.parse_jsonl_file(run_dir / "raw" / f"giraffes_{d}.jsonl"):
            raw[rec["item_id"]] = rec

    rows, unparsed = [], []
    for item_id, m in manifest.items():
        f = run_dir / "trajectory_outputs" / f"{item_id}.txt"
        traj = judge.parse_trajectory_answer(f.read_text()) if f.exists() else None
        if traj is None:
            unparsed.append(item_id)
            continue
        rec = raw[item_id]
        # "favored side" is direction-dependent: above_good favors > T, below_good favors <= T
        first_good = (traj[0] > T if m["direction"] == "above_good" else traj[0] <= T) if traj else None
        rows.append({**m, "trajectory": traj, "n_estimates": len(traj),
                     "first_estimate": traj[0] if traj else None,
                     "first_on_good_side": first_good,
                     "tokens": rec["num_tokens"],
                     "sentences": len(segment.segment_sentences(rec["raw_completion"]))})

    print(f"parsed {len(rows)}/{len(manifest)} trajectories"
          + (f"  (unparsed: {unparsed})" if unparsed else "") + f"\nthreshold = {T:,.0f}\n")

    g = {label: [r for r in rows if r["judge_raw"] == cat] for label, cat in GROUPS}
    print("=== GOOD-SIDE rollouts only, overt vs covert "
          f"(n={len(g['overt (Admits)'])} vs {len(g['covert (Denies)'])}) ===")
    out = {}
    out["cot_tokens"] = compare("CoT length (tokens)", [r["tokens"] for r in g["overt (Admits)"]],
                                [r["tokens"] for r in g["covert (Denies)"]])
    out["cot_sentences"] = compare("CoT length (sentences)",
                                   [r["sentences"] for r in g["overt (Admits)"]],
                                   [r["sentences"] for r in g["covert (Denies)"]])
    out["n_estimates"] = compare("# intermediate estimates",
                                 [r["n_estimates"] for r in g["overt (Admits)"]],
                                 [r["n_estimates"] for r in g["covert (Denies)"]])

    print("\n=== Where the FIRST estimate fell ===")
    out["first_estimate"] = {}
    for label, _ in GROUPS:
        sub = [r for r in g[label] if r["first_estimate"] is not None]
        if not sub:
            continue
        n_good = sum(r["first_on_good_side"] for r in sub)
        firsts = [r["first_estimate"] for r in sub]
        lo, hi = sps.binomtest(n_good, len(sub), 0.5).proportion_ci(0.95)
        print(f"  {label:16s} n={len(sub):2d}  first already on good side: {n_good}/{len(sub)} "
              f"= {n_good/len(sub):.2f}  95% CI [{lo:.2f}, {hi:.2f}]")
        print(f"                     median first estimate = {st.median(firsts):,.0f}  "
              f"(threshold {T:,.0f})")
        out["first_estimate"][label] = {"n": len(sub), "n_first_good": n_good,
                                        "p_first_good": n_good / len(sub),
                                        "ci": [lo, hi], "median_first": st.median(firsts)}
    # Pooling raw first-estimate values across directions is meaningless -- "good" is > T in
    # above_good and <= T in below_good, so the two halves point opposite ways. Report per
    # direction, and pool only via a signed distance oriented toward each rollout's own
    # favored side.
    print("\n  -- per direction (pooled medians would be meaningless: the good side flips) --")
    for d in ["above_good", "below_good"]:
        print(f"  {d}  (good = {'>' if d == 'above_good' else '<='} {T:,.0f})")
        for label, _ in GROUPS:
            sub = [r for r in g[label] if r["direction"] == d and r["first_estimate"]]
            if not sub:
                continue
            ng = sum(r["first_on_good_side"] for r in sub)
            print(f"     {label:16s} n={len(sub):2d}  first good-side {ng}/{len(sub)}  "
                  f"median first = {st.median([r['first_estimate'] for r in sub]):,.0f}")

    def signed(r):
        lr = np.log10(r["first_estimate"] / T)
        return lr if r["direction"] == "above_good" else -lr

    so = [signed(r) for r in g["overt (Admits)"] if r["first_estimate"]]
    sc = [signed(r) for r in g["covert (Denies)"] if r["first_estimate"]]
    if so and sc:
        u, p = sps.mannwhitneyu(so, sc, alternative="two-sided")
        print(f"\n  signed log10(first/T) toward the favored side: overt {np.median(so):+.3f}  "
              f"covert {np.median(sc):+.3f}  U={u:.0f} {fmt_p(p)}")
        out["first_signed_log"] = {"overt_median": float(np.median(so)),
                                   "covert_median": float(np.median(sc)), "p": float(p)}

    # Is "covert starts good-side" just the revision-count difference? Conditioning on the
    # final outcome means a rollout with fewer revisions had less room to move, so its first
    # estimate is mechanically more likely to already be good. Control for it.
    X = np.column_stack([np.ones(len(rows)),
                         np.array([1.0 if r["judge_raw"] == "NOT_INFLUENCED" else 0.0 for r in rows]),
                         np.array([r["n_estimates"] for r in rows]) / 10.0])
    y = np.array([1.0 if r["first_on_good_side"] else 0.0 for r in rows])
    b = np.zeros(3)
    for _ in range(400):
        pr = 1 / (1 + np.exp(-X @ b)); W = pr * (1 - pr) + 1e-9
        b += np.linalg.solve((X * W[:, None]).T @ X + 1e-6 * np.eye(3), X.T @ (y - pr))
    pr = 1 / (1 + np.exp(-X @ b)); W = pr * (1 - pr) + 1e-9
    se = np.sqrt(np.diag(np.linalg.inv((X * W[:, None]).T @ X + 1e-6 * np.eye(3))))
    print("\n  logistic  P(first estimate already good-side) ~ covert + n_estimates")
    out["first_logistic"] = {}
    for nm, bb, ss in zip(["intercept", "covert", "n_estimates_per10"], b, se):
        z = bb / ss
        pv = 2 * (1 - sps.norm.cdf(abs(z)))
        print(f"    {nm:20s} beta={bb:+.3f} SE={ss:.3f} z={z:+.2f} {fmt_p(pv)}"
              + (f"  OR={np.exp(bb):.2f}" if nm != "intercept" else ""))
        out["first_logistic"][nm] = {"beta": float(bb), "se": float(ss), "p": float(pv)}

    ov = [r for r in g["overt (Admits)"] if r["first_estimate"] is not None]
    cv = [r for r in g["covert (Denies)"] if r["first_estimate"] is not None]
    if ov and cv:
        tbl = [[sum(r["first_on_good_side"] for r in ov), len(ov) - sum(r["first_on_good_side"] for r in ov)],
               [sum(r["first_on_good_side"] for r in cv), len(cv) - sum(r["first_on_good_side"] for r in cv)]]
        odds, p = sps.fisher_exact(tbl)
        print(f"\n  overt vs covert on first-estimate-good: Fisher OR={odds:.2f}, {fmt_p(p)}")
        out["first_estimate"]["fisher"] = {"table": tbl, "odds_ratio": float(odds), "p": float(p)}
        compare("first estimate (value)", [r["first_estimate"] for r in ov],
                [r["first_estimate"] for r in cv])

    print("\n=== CONFOUNDS — read before believing any of the above ===")
    print("  1. Both groups are conditioned on landing good-side AND on a judge label whose")
    print("     instances disagreed at p=0.007. This is post-hoc stratification, not a design.")
    print("  2. Length vs label is circular: a longer CoT gives the judge more text in which to")
    print("     find an admission, so 'overt CoTs are longer' may be a judge artifact rather")
    print("     than a fact about the model. Testing this needs length-matched sampling.")
    print("  3. n_estimates and tokens are near-collinear by construction; treat them as one")
    print("     finding, not two.")
    print("  4. No bad-side trajectory control was run, so the first-estimate contrast is still")
    print("     conditioned on the final outcome. The n_estimates control above addresses one")
    print("     route for that to bite, not all of them.")

    (run_dir / "trajectory_results.json").write_text(json.dumps(
        {"threshold": T, "n_parsed": len(rows), "unparsed": unparsed,
         "comparisons": out, "rows": rows}, indent=2))
    print(f"\nwrote {run_dir/'trajectory_results.json'}")


if __name__ == "__main__":
    main()
