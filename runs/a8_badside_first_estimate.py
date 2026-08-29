"""A8 — the first-estimate contrast, run off the final outcome.

The hour-0 finding ("covert rollouts start on the favoured side 0.86 of the time vs 0.61 for
overt") was computed on GOOD-SIDE rollouts only, so it is conditioned on the outcome it is meant
to explain. This runs the same statistic on the bad-side rollouts and on the no-bet baseline, so
the overt/covert contrast can be read without that conditioning.

Two traps this is built to avoid, both already recorded in hypotheses.md:

  - THE VACUOUS NULL. Scoring the baseline under BOTH framings counts every rollout twice, once
    as favoured and once as unfavoured, which forces the rate to 0.5 by arithmetic. The baseline
    is therefore scored under each FIXED framing separately, with a permutation null.
  - PASS-1 LABELS. The manifest carries pass-1 covertness labels; the majority-vote labels
    supersede them. Majority labels are used where available and the disagreement is reported.

Run with: python3 runs/a8_badside_first_estimate.py   (no GPU)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from src.qual import judge, parse                              # noqa: E402

RUN = HERE / "hour0"


def wilson(k, n):
    if n == 0:
        return (float("nan"), float("nan"))
    lo, hi = sps.binomtest(int(k), int(n), 0.5).proportion_ci(0.95)
    return float(lo), float(hi)


def main():
    T = json.load(open(RUN / "threshold.json"))["threshold"]
    manifest = {m["id"]: m for m in json.load(open(RUN / "trajectory_manifest.json"))}
    maj = json.load(open(RUN / "covertness_majority.json"))["final_labels"]
    # majority label ids look like covertness_<direction>_<item_id>, and <direction>
    # itself contains an underscore ("above_good"), so a fixed split count silently
    # mis-keys every entry and yields empty overt/covert cells. Key off item_id.
    maj_by_item = {}
    for k, v in maj.items():
        i = k.find("giraffes_")
        if i == -1:
            raise ValueError(f"cannot recover item_id from covertness id {k!r}")
        maj_by_item[k[i:]] = v

    rows, n_disagree = [], 0
    for iid, m in manifest.items():
        f = RUN / "trajectory_outputs" / f"{iid}.txt"
        if not f.exists():
            continue
        traj = judge.parse_trajectory_answer(f.read_text())
        if not traj:
            continue
        lab = maj_by_item.get(iid)
        if lab is not None and m.get("judge_raw") is not None:
            p1 = "INFLUENCED" if m["judge_raw"] == "INFLUENCED" else "NOT_INFLUENCED"
            n_disagree += (p1 != lab)
        rows.append({"id": iid, "direction": m["direction"], "traj": traj,
                     "on_good_side": m.get("on_good_side"), "label": lab})

    print(f"{len(rows)} trajectories loaded; majority vs pass-1 disagreement on "
          f"{n_disagree} labelled items\n")

    def first_fav(r, framing=None):
        d = framing or r["direction"]
        return r["traj"][0] > T if d == "above_good" else r["traj"][0] <= T

    res = {"threshold": T}
    inter = [r for r in rows if r["direction"] in ("above_good", "below_good")]
    base = [r for r in rows if r["direction"] == "baseline"]

    print("=== intervention rollouts, split by FINAL outcome and covertness ===")
    print(f"{'outcome':12s}{'group':22s}{'n':>5s}{'first favoured':>16s}{'95% CI':>18s}")
    for outcome, want in (("good side", True), ("bad side", False)):
        sub = [r for r in inter if r["on_good_side"] is want]
        for gname, lab in (("overt (INFLUENCED)", "INFLUENCED"),
                           ("covert (denies)", "NOT_INFLUENCED"),
                           ("all", None)):
            g = [r for r in sub if lab is None or r["label"] == lab]
            if not g:
                continue
            k = sum(first_fav(r) for r in g)
            lo, hi = wilson(k, len(g))
            print(f"{outcome:12s}{gname:22s}{len(g):5d}{k/len(g):16.3f}"
                  f"{f'[{lo:.3f}, {hi:.3f}]':>18s}")
            res[f"{outcome.replace(' ', '_')}/{gname.split()[0]}"] = {
                "n": len(g), "frac_first_favoured": k / len(g), "ci": [lo, hi]}

    # overt vs covert within the bad-side stratum: the contrast the confound demanded
    print("\n=== the contrast, now WITHIN each outcome stratum ===")
    for outcome, want in (("good side", True), ("bad side", False)):
        sub = [r for r in inter if r["on_good_side"] is want]
        o = [r for r in sub if r["label"] == "INFLUENCED"]
        c = [r for r in sub if r["label"] == "NOT_INFLUENCED"]
        if not o or not c:
            print(f"  {outcome:10s} insufficient cells (overt {len(o)}, covert {len(c)})")
            continue
        ko, kc = sum(first_fav(r) for r in o), sum(first_fav(r) for r in c)
        tab = [[kc, len(c) - kc], [ko, len(o) - ko]]
        odds, p = sps.fisher_exact(tab)
        print(f"  {outcome:10s} covert {kc}/{len(c)} = {kc/len(c):.3f}  vs  "
              f"overt {ko}/{len(o)} = {ko/len(o):.3f}   Fisher OR {odds:.2f}, p={p:.4f}")
        res[f"contrast_{outcome.replace(' ', '_')}"] = {
            "covert": kc / len(c), "n_covert": len(c), "overt": ko / len(o), "n_overt": len(o),
            "fisher_or": float(odds), "fisher_p": float(p)}

    print("\n=== baseline (no bet), scored under each FIXED framing separately ===")
    print("  (scoring it under both at once would force 0.5 by arithmetic — the vacuous null)")
    print("  NOTE: the two framings are complementary by construction (>T vs <=T),")
    print("  so the two rates necessarily sum to 1. They are ONE measurement — how")
    print("  the baseline's first estimate splits around T — not two independent tests.")
    rng = np.random.default_rng(0)
    for framing in ("above_good", "below_good"):
        k = sum(first_fav(r, framing) for r in base)
        lo, hi = wilson(k, len(base))
        # permutation null: shuffle which side counts as favoured, per rollout
        null = []
        for _ in range(5000):
            flips = rng.random(len(base)) < 0.5
            null.append(np.mean([first_fav(r, "above_good" if f else "below_good")
                                 for r, f in zip(base, flips)]))
        null = np.array(null)
        p = min(1.0, 2 * min((null <= k / len(base)).mean(),
                             (null >= k / len(base)).mean()))
        print(f"  framing={framing:11s} {k}/{len(base)} = {k/len(base):.3f}  "
              f"[{lo:.3f}, {hi:.3f}]   permutation null {null.mean():.3f}"
              f"+-{null.std():.3f}, p={p:.4f}")
        res[f"baseline/{framing}"] = {"n": len(base), "frac_first_favoured": k / len(base),
                                      "ci": [lo, hi], "null_mean": float(null.mean()),
                                      "null_sd": float(null.std()), "perm_p": float(p)}

    print("\n  reference: the hour-0 good-side-only figures were covert 0.86 vs overt 0.61 "
          "(p=0.025).")
    out = RUN / "a8_badside_first_estimate.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
