"""H2: is there asymmetric scrutiny? Revision, direction, and stopping by which side of the
threshold the current estimate sits on.

The Kunda signature is: keep revising while the current estimate is on the BAD side, stop once
it reaches the GOOD side, and let revisions drift toward the favoured side.

Three traps this script is built around:

1. A raw "P(stop | good side)" is near-meaningless. The last estimate is by construction where
   the rollout stopped, and finals are biased good, so the raw rate is guaranteed to favour the
   good side. The honest quantity is a HAZARD: at each step, given you are here and have not yet
   stopped, what is the probability you stop now? Position must be controlled for, because late
   steps stop more regardless of side. Done here with a logistic on (side, step index).

2. The baseline condition is the null. Baseline prompts mention no threshold, so any
   side-dependence there is an artifact of the estimate distribution, not an incentive.

   BUT the obvious way to build it is wrong. Scoring each baseline rollout under BOTH framings
   counts every step twice -- once as "good", once as "bad" -- which forces
   P(stop | good) == P(stop | bad) identically. That null returns OR = 1.000, p = 1.0000 by
   arithmetic and can never detect anything. Instead the baseline is scored under each FIXED
   framing separately ("good" = above T for all rollouts, then "good" = below T for all), which
   asks the real question: absent any incentive, does the model stop more when its current
   estimate happens to be large (or small)? A permutation null over per-rollout framing
   assignments is reported alongside.

3. Anchoring is not asymmetry. The threshold pulls both directions toward the same number, so
   drift toward T is not evidence for H2; only drift that differs by which side is FAVOURED is.

Run with: source env.sh && python3 runs/analyze_revision.py [--run hour0]
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.qual import judge


def logistic(X, y, names, label):
    b = np.zeros(X.shape[1])
    for _ in range(500):
        pr = 1 / (1 + np.exp(-X @ b))
        W = pr * (1 - pr) + 1e-9
        b += np.linalg.solve((X * W[:, None]).T @ X + 1e-6 * np.eye(X.shape[1]), X.T @ (y - pr))
    pr = 1 / (1 + np.exp(-X @ b))
    W = pr * (1 - pr) + 1e-9
    se = np.sqrt(np.diag(np.linalg.inv((X * W[:, None]).T @ X + 1e-6 * np.eye(X.shape[1]))))
    print(f"  {label}")
    out = {}
    for nm, bb, ss in zip(names, b, se):
        z = bb / ss
        p = 2 * (1 - sps.norm.cdf(abs(z)))
        star = " *" if p < 0.05 else ""
        extra = f"  OR={np.exp(bb):.2f}" if nm != "intercept" else ""
        print(f"    {nm:22s} beta={bb:+.3f} SE={ss:.3f} z={z:+.2f} p={p:.4f}{extra}{star}")
        out[nm] = {"beta": float(bb), "se": float(ss), "p": float(p)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="hour0")
    ap.add_argument("--min-len", type=int, default=3,
                    help="skip trajectories shorter than this; too few steps to model stopping")
    args = ap.parse_args()
    run_dir = Path(__file__).resolve().parent / args.run
    T = json.load(open(run_dir / "threshold.json"))["threshold"]
    man = {m["id"]: m for m in json.load(open(run_dir / "trajectory_manifest.json"))}

    traj, missing = {}, []
    for item_id in man:
        f = run_dir / "trajectory_outputs" / f"{item_id}.txt"
        t = judge.parse_trajectory_answer(f.read_text()) if f.exists() else None
        if t is None or len(t) < args.min_len:
            missing.append(item_id)
            continue
        traj[item_id] = t
    print(f"trajectories usable: {len(traj)}/{len(man)} (dropped {len(missing)} missing or "
          f"shorter than {args.min_len})\nthreshold = {T:,.0f}\n")

    by_cond = collections.Counter(man[i]["direction"] for i in traj)
    print("  by condition:", dict(by_cond))

    def favoured(v, direction):
        return v > T if direction == "above_good" else v <= T

    def build_rows(framing_for):
        """framing_for(item_id) -> the direction under which this rollout's steps are scored."""
        out = []
        for item_id, t in traj.items():
            pseudo = framing_for(item_id)
            if pseudo is None:
                continue
            for i, v in enumerate(t):
                out.append({"id": item_id, "cond": man[item_id]["direction"], "framing": pseudo,
                            "step": i, "n_steps": len(t), "value": v,
                            "good": bool(favoured(v, pseudo)),
                            "stopped": i == len(t) - 1,
                            "next": t[i + 1] if i + 1 < len(t) else None})
        return out

    inter = build_rows(lambda i: man[i]["direction"] if man[i]["direction"] != "baseline" else None)
    base_fixed = {f: build_rows(lambda i, f=f: f if man[i]["direction"] == "baseline" else None)
                  for f in ("above_good", "below_good")}
    rows = inter + base_fixed["above_good"] + base_fixed["below_good"]

    print("\n=== 1. STOPPING HAZARD: P(stop here | on this side), controlling for step index ===")
    res = {}
    groups = [("intervention (above+below)", inter),
              ("baseline, scored as above-good", base_fixed["above_good"]),
              ("baseline, scored as below-good", base_fixed["below_good"])]
    for cond_label, sub in groups:
        if not sub:
            continue
        raw_g = np.mean([r["stopped"] for r in sub if r["good"]])
        raw_b = np.mean([r["stopped"] for r in sub if not r["good"]])
        print(f"\n  {cond_label}: {len(sub)} steps from "
              f"{len({r['id'] for r in sub})} rollouts")
        print(f"    raw  P(stop | good side)={raw_g:.3f}   P(stop | bad side)={raw_b:.3f}   "
              f"(uncontrolled — see docstring)")
        X = np.column_stack([np.ones(len(sub)),
                             np.array([1.0 if r["good"] else 0.0 for r in sub]),
                             np.array([r["step"] for r in sub]) / 10.0])
        y = np.array([1.0 if r["stopped"] else 0.0 for r in sub])
        res[cond_label] = logistic(X, y, ["intercept", "on good side", "step index (per 10)"],
                                   "hazard model P(stop) ~ side + step:")

    print("\n=== 2. REVISION DIRECTION: does a revision move toward the favoured side? ===")
    for cond_label, grp in [("intervention", inter),
                            ("baseline as above-good", base_fixed["above_good"]),
                            ("baseline as below-good", base_fixed["below_good"])]:
        sub = [r for r in grp if r["next"] is not None]
        if not sub:
            continue
        for side, want in [("currently BAD side", False), ("currently GOOD side", True)]:
            s2 = [r for r in sub if r["good"] == want]
            if not s2:
                continue
            toward = []
            for r in s2:
                moved_up = r["next"] > r["value"]
                toward.append(moved_up if r["framing"] == "above_good" else not moved_up)
            p = np.mean(toward)
            ci = sps.binomtest(int(sum(toward)), len(toward), 0.5).proportion_ci(0.95)
            print(f"  {cond_label:16s} {side:20s} n={len(toward):5d}  "
                  f"P(move toward favoured)={p:.3f}  95% CI [{ci.low:.3f}, {ci.high:.3f}]"
                  f"{'  *' if not (ci.low <= 0.5 <= ci.high) else ''}")

    print("\n=== 3. TRAJECTORY LENGTH by where the FIRST estimate fell ===")
    print("  (baseline omitted: under the both-framings scoring every rollout is first-good in")
    print("   one framing and first-bad in the other, so the two groups are the same rollouts.)")
    ids = [i for i in traj if man[i]["direction"] != "baseline"]
    for lab, want in [("first on GOOD side", True), ("first on BAD side", False)]:
        lens = [len(traj[i]) for i in ids
                if bool(favoured(traj[i][0], man[i]["direction"])) == want]
        if lens:
            u = ""
            print(f"  intervention  {lab:20s} n={len(lens):3d}  "
                  f"median={np.median(lens):.1f}  mean={np.mean(lens):.1f}{u}")
    a = [len(traj[i]) for i in ids if favoured(traj[i][0], man[i]["direction"])]
    b = [len(traj[i]) for i in ids if not favoured(traj[i][0], man[i]["direction"])]
    if len(a) > 2 and len(b) > 2:
        u, pv = sps.mannwhitneyu(a, b, alternative="two-sided")
        print(f"  Mann-Whitney first-good vs first-bad length: U={u:.0f}, p={pv:.4f}")

    print("\n=== 4. PERMUTATION NULL: shuffle which framing each intervention rollout got ===")
    rng = np.random.default_rng(0)
    obs = None
    stats_null = []
    for rep in range(2000):
        rr = []
        for item_id, t in traj.items():
            if man[item_id]["direction"] == "baseline":
                continue
            pseudo = "above_good" if rng.random() < 0.5 else "below_good"
            for i, v in enumerate(t):
                rr.append((bool(favoured(v, pseudo)), i == len(t) - 1))
        g = np.mean([s for gd, s in rr if gd]); bad = np.mean([s for gd, s in rr if not gd])
        stats_null.append(g - bad)
    obs = (np.mean([r["stopped"] for r in inter if r["good"]])
           - np.mean([r["stopped"] for r in inter if not r["good"]]))
    pv = float(np.mean(np.abs(stats_null) >= abs(obs)))
    print(f"  observed P(stop|good) - P(stop|bad) = {obs:+.4f}")
    print(f"  permutation null: mean {np.mean(stats_null):+.4f}, sd {np.std(stats_null):.4f}, "
          f"two-sided p = {pv:.4f}")

    json.dump({"threshold": T, "n_trajectories": len(traj), "dropped": missing,
               "hazard_models": res,
               "step_rows": len(rows)}, open(run_dir / "revision_results.json", "w"), indent=2)
    print(f"\nwrote {run_dir/'revision_results.json'}")


if __name__ == "__main__":
    main()
