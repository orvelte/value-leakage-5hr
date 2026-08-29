"""A11 — the covert slice, restated as a residual on a mostly stance-faithful narration.

The report currently leads with the paper's Fig. 6 covert-share lower bound (12.6%) without
saying what makes it a *lower bound*. The reframe: admitters and deniers do not land favoured at
the same rate. Deniers are substantially less biased. The lower bound is the part of the
deniers' residual that the pigeonhole argument forces to be biased-and-unacknowledged, not a
claim that 12.6% of rollouts are lying.

Recomputed from the majority-vote labels, since the pass-1 labels are superseded.

Run with: python3 runs/a11_covert_reframe.py   (no GPU)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from src.qual import parse, stats                              # noqa: E402

RUN = HERE / "hour0"


def main():
    T = json.load(open(RUN / "threshold.json"))["threshold"]
    labels = json.load(open(RUN / "covertness_majority.json"))["final_labels"]

    # label id -> (direction, rollout index)
    by_dir = {"above_good": {}, "below_good": {}}
    for k, v in labels.items():
        d = "above_good" if "_above_good_" in k else "below_good"
        by_dir[d][int(k.rsplit("_", 1)[1])] = v

    cells = {}
    for d in ("above_good", "below_good"):
        recs = list(parse.parse_jsonl_file(RUN / "raw" / f"giraffes_{d}.jsonl"))
        for idx, lab in by_dir[d].items():
            if idx >= len(recs):
                continue
            v, _ = parse.parse_estimate(recs[idx]["raw_completion"])
            if v is None or not parse.apply_outlier_filter(v, T):
                continue
            good = parse.classify_on_good_side(v, T, d)
            cells.setdefault(lab, {}).setdefault(d, []).append(1 if good else 0)

    res = {"threshold": T, "groups": {}}
    print(f"{'group':18s}{'n':>5s}{'P(good) pooled':>16s}{'95% CI':>20s}"
          f"{'bias':>9s}{'bias 95% CI':>22s}")
    for lab in ("INFLUENCED", "NOT_INFLUENCED"):
        ab, be = cells[lab]["above_good"], cells[lab]["below_good"]
        pooled = ab + be
        p = float(np.mean(pooled))
        lo, hi = sps.binomtest(int(sum(pooled)), len(pooled), 0.5).proportion_ci(0.95)
        b, blo, bhi = stats.balanced_bias_bootstrap(ab, be)
        res["groups"][lab] = {"n": len(pooled), "n_above": len(ab), "n_below": len(be),
                              "p_good_pooled": p, "p_good_ci": [lo, hi],
                              "bias": b, "bias_ci": [blo, bhi]}
        print(f"{lab:18s}{len(pooled):5d}{p:16.3f}{f'[{lo:.3f}, {hi:.3f}]':>20s}"
              f"{b:+9.3f}{f'[{blo:+.3f}, {bhi:+.3f}]':>22s}")

    a = res["groups"]["INFLUENCED"]["p_good_pooled"]
    n_a = res["groups"]["INFLUENCED"]["n"]
    d_ = res["groups"]["NOT_INFLUENCED"]["p_good_pooled"]
    n_d = res["groups"]["NOT_INFLUENCED"]["n"]
    tab = [[int(round(a * n_a)), n_a - int(round(a * n_a))],
           [int(round(d_ * n_d)), n_d - int(round(d_ * n_d))]]
    odds, p_fish = sps.fisher_exact(tab)
    res["admitters_vs_deniers"] = {"diff_p_good": a - d_, "fisher_or": float(odds),
                                   "fisher_p": float(p_fish)}
    print(f"\n  admitters - deniers, P(good): {a - d_:+.3f}  (Fisher OR {odds:.2f}, p={p_fish:.4f})")
    print(f"  covert-share lower bound (majority labels): "
          f"{json.load(open(RUN / 'covertness_majority.json'))['covert_share']:.3f}")
    print("\n  Reading: the narration is mostly stance-faithful — rollouts that admit the")
    print("  influence are the ones that land favoured. The covert share is the residual on")
    print("  the deniers that the pigeonhole argument forces to be biased, not a count of lies.")

    out = RUN / "a11_covert_reframe.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
