"""A2 — revision structure inside the prefilled arms.

Addendum prediction: neutral and denial arms have few revisions AND no stopping asymmetry;
the admission arm has few revisions but its FIRST estimate is already on the favoured side
(an H1 signature, not H2).

Run with: source env.sh && python3 runs/a2_analyze_prefill_traj.py
"""
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RUN = Path(__file__).resolve().parent / "prefill_tests"
FREE_MEDIAN_EST = 24        # median intermediate estimates in free CoT (from hour0)


def hazard_or(rows):
    """OR for stopping when on the favoured side, controlling for step index."""
    if len(rows) < 20 or len({r[0] for r in rows}) < 2:
        return None
    X = np.column_stack([np.ones(len(rows)), [r[0] for r in rows], [r[1] for r in rows]])
    y = np.array([r[2] for r in rows])
    b = np.zeros(3)
    for _ in range(500):
        pr = 1 / (1 + np.exp(-X @ b)); W = pr * (1 - pr) + 1e-9
        b += np.linalg.solve((X * W[:, None]).T @ X + 1e-6 * np.eye(3), X.T @ (y - pr))
    pr = 1 / (1 + np.exp(-X @ b)); W = pr * (1 - pr) + 1e-9
    se = np.sqrt(np.diag(np.linalg.inv((X * W[:, None]).T @ X + 1e-6 * np.eye(3))))
    z = b[1] / se[1]
    return float(np.exp(b[1])), float(2 * (1 - sps.norm.cdf(abs(z))))


def main():
    T = json.load(open(Path(__file__).resolve().parent / "hour0" / "threshold.json"))["threshold"]
    man = {m["id"]: m for m in json.load(open(RUN / "trajectory_manifest.json"))}
    traj = json.load(open(RUN / "trajectories.json"))
    res = {"arms": {}}

    def fav(v, d):
        return v > T if d == "above_good" else v <= T

    print(f"{'arm':12s}{'n':>5s}{'med #est':>10s}{'mean #est':>11s}{'med tokens':>12s}"
          f"{'first favoured':>16s}{'stop OR':>10s}{'p':>8s}")
    print("-" * 84)
    for arm in ["neutral", "denial", "admission"]:
        ids = [i for i in traj if man[i]["arm"] == arm and traj[i]]
        n_est = [len(traj[i]) for i in ids]
        toks = [man[i]["n_tokens"] for i in ids]
        first = [fav(traj[i][0], man[i]["direction"]) for i in ids]
        rows = []
        for i in ids:
            t = traj[i]
            if len(t) < 3:
                continue
            d = man[i]["direction"]
            for k, v in enumerate(t):
                rows.append((1.0 if fav(v, d) else 0.0, k / 10.0,
                             1.0 if k == len(t) - 1 else 0.0))
        h = hazard_or(rows)
        ors = f"{h[0]:.2f}" if h else "—"
        ps = f"{h[1]:.3f}" if h else "—"
        print(f"{arm:12s}{len(ids):5d}{st.median(n_est):10.0f}{np.mean(n_est):11.1f}"
              f"{st.median(toks):12.0f}{np.mean(first):16.3f}{ors:>10s}{ps:>8s}")
        lo, hi = sps.binomtest(int(sum(first)), len(first), 0.5).proportion_ci(0.95)
        res["arms"][arm] = {"n": len(ids), "median_n_est": st.median(n_est),
                            "mean_n_est": float(np.mean(n_est)),
                            "median_tokens": st.median(toks),
                            "frac_first_favoured": float(np.mean(first)),
                            "first_favoured_ci": [lo, hi],
                            "stop_or": h[0] if h else None, "stop_p": h[1] if h else None,
                            "n_steps": len(rows)}
    print(f"\n  free-CoT reference: median #est {FREE_MEDIAN_EST}, median tokens 9046, "
          f"stop OR 2.11 (p=0.0053)")

    print("\n=== first estimate already on the favoured side (the H1 signature) ===")
    for arm in ["neutral", "denial", "admission"]:
        a = res["arms"][arm]
        star = "  *" if not (a["first_favoured_ci"][0] <= 0.5 <= a["first_favoured_ci"][1]) else ""
        print(f"  {arm:12s} {a['frac_first_favoured']:.3f}  95% CI "
              f"[{a['first_favoured_ci'][0]:.3f}, {a['first_favoured_ci'][1]:.3f}]{star}")
    print("  free-CoT reference: 0.52 overall (41/79)")

    print("\n=== revision DIRECTION (the other half of the H2 signature) ===")
    print(f"  {'arm':12s}{'on BAD side':>28s}{'on GOOD side':>28s}")
    for arm in ["neutral", "denial", "admission"]:
        ids = [i for i in traj if man[i]["arm"] == arm and len(traj[i]) >= 2]
        cells = {}
        for want in (False, True):
            tw = []
            for i in ids:
                t, d = traj[i], man[i]["direction"]
                for k in range(len(t) - 1):
                    if fav(t[k], d) is want:
                        up = t[k + 1] > t[k]
                        tw.append(up if d == "above_good" else not up)
            if tw:
                lo, hi = sps.binomtest(int(sum(tw)), len(tw), 0.5).proportion_ci(0.95)
                sig = "" if lo <= 0.5 <= hi else " *"
                cells[want] = f"{np.mean(tw):.3f} [{lo:.2f},{hi:.2f}] n={len(tw)}{sig}"
            else:
                cells[want] = "—"
        print(f"  {arm:12s}{cells[False]:>28s}{cells[True]:>28s}")
        res["arms"][arm]["revision_direction"] = {"bad_side": cells[False],
                                                  "good_side": cells[True]}
    print("  free-CoT reference: bad side 0.621 [0.586, 0.655] *, good side 0.452 *")

    print("\n=== where does the bias appear, first estimate vs last? ===")
    print(f"  {'arm':12s}{'first favoured':>16s}{'last favoured':>15s}{'drift':>9s}")
    for arm in ["neutral", "denial", "admission"]:
        ids = [i for i in traj if man[i]["arm"] == arm and traj[i]]
        f0 = np.mean([fav(traj[i][0], man[i]["direction"]) for i in ids])
        f1 = np.mean([fav(traj[i][-1], man[i]["direction"]) for i in ids])
        print(f"  {arm:12s}{f0:16.3f}{f1:15.3f}{f1-f0:+9.3f}")
        res["arms"][arm]["last_favoured"] = float(f1)
        res["arms"][arm]["drift_first_to_last"] = float(f1 - f0)

    print("\n=== verdict vs the addendum's A2 prediction ===")
    a_n, d_n, m_n = (res["arms"][k]["median_n_est"] for k in ("neutral", "denial", "admission"))
    print(f"  'far fewer revisions than free CoT': median {a_n:.0f}/{d_n:.0f}/{m_n:.0f} "
          f"vs {FREE_MEDIAN_EST} -> "
          f"{'CONFIRMED' if max(a_n, d_n, m_n) < FREE_MEDIAN_EST * 0.6 else 'NOT as predicted'}")
    print(f"  'admission arm starts favoured': "
          f"{res['arms']['admission']['frac_first_favoured']:.3f} vs neutral "
          f"{res['arms']['neutral']['frac_first_favoured']:.3f}")

    (RUN / "a2_results.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {RUN/'a2_results.json'}")


if __name__ == "__main__":
    main()
