"""A9 — revision structure in the `minimal` arm, against free CoT and the neutral arm.

The A9 prediction: minimal ("Okay.") keeps free-CoT length, so if the revision loop is a property
of the trace rather than of being prefilled at all, minimal should look like free CoT — ~24
intermediate estimates and H2's stopping asymmetry intact — and NOT like neutral's ~4 with no
asymmetry. That is the cleanest available test of whether prefilling per se disables H2.

Reuses hazard_or from the A2 script rather than reimplementing it, so the stopping OR here and
the one reported for neutral/denial/admission come from the same estimator.

Run with: python3 runs/a9_analyze_minimal_traj.py   (no GPU; needs the judge outputs)
"""
import json
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from a2_analyze_prefill_traj import hazard_or          # noqa: E402  (shared estimator)

RUN = HERE / "prefill_extra"
# references, from hour0 (free CoT) and prefill_tests/a2_results.json (neutral)
REF = {"free_cot": {"med_est": 24, "med_tok": 9046, "stop_or": 2.11, "stop_p": 0.0053,
                    "first_fav": 0.52, "bad_pull": 0.621},
       "neutral":  {"med_est": 4, "med_tok": 1240, "stop_or": 1.14, "stop_p": 0.59,
                    "first_fav": 0.453, "bad_pull": 0.651}}

LINE = re.compile(r"^([A-Za-z0-9_]+)\|(.+)$")


def build_trajectories():
    """trajectory_outputs/*.txt -> {id: [ints]}. Reports coverage and malformed lines rather
    than silently dropping them; a judge batch that returned prose instead of the format would
    otherwise look like a set of rollouts with no estimates."""
    traj, bad, seen = {}, [], set()
    for f in sorted((RUN / "trajectory_outputs").glob("tbatch*.txt")):
        for ln in f.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            m = LINE.match(ln)
            if not m:
                bad.append((f.name, ln[:60]))
                continue
            iid, payload = m.group(1), m.group(2).strip()
            seen.add(iid)
            if payload.upper() == "NONE":
                traj[iid] = []
                continue
            try:
                traj[iid] = [int(x) for x in payload.split(",") if x.strip()]
            except ValueError:
                bad.append((f.name, ln[:60]))
    return traj, bad, seen


def main():
    T = json.load(open(HERE / "hour0" / "threshold.json"))["threshold"]
    man = {m["id"]: m for m in json.load(open(RUN / "trajectory_manifest.json"))}
    traj, bad, seen = build_trajectories()
    (RUN / "trajectories.json").write_text(json.dumps(traj))

    missing = sorted(set(man) - seen)
    print(f"coverage: {len(seen)}/{len(man)} ids judged, {len(bad)} malformed lines, "
          f"{sum(1 for v in traj.values() if not v)} empty (NONE)")
    if missing:
        print(f"  MISSING ({len(missing)}): {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    for f, ln in bad[:5]:
        print(f"  malformed in {f}: {ln!r}")

    def fav(v, d):
        return v > T if d == "above_good" else v <= T

    ids = [i for i in traj if traj[i] and i in man]
    n_est = [len(traj[i]) for i in ids]
    toks = [man[i]["n_tokens"] for i in ids]
    first = [fav(traj[i][0], man[i]["direction"]) for i in ids]
    last = [fav(traj[i][-1], man[i]["direction"]) for i in ids]

    rows = []
    for i in ids:
        t = traj[i]
        if len(t) < 3:
            continue
        d = man[i]["direction"]
        for k, v in enumerate(t):
            rows.append((1.0 if fav(v, d) else 0.0, k / 10.0, 1.0 if k == len(t) - 1 else 0.0))
    h = hazard_or(rows)

    lo, hi = sps.binomtest(int(sum(first)), len(first), 0.5).proportion_ci(0.95)
    res = {"n": len(ids), "median_n_est": st.median(n_est), "mean_n_est": float(np.mean(n_est)),
           "median_tokens": st.median(toks), "frac_first_favoured": float(np.mean(first)),
           "first_favoured_ci": [lo, hi], "frac_last_favoured": float(np.mean(last)),
           "drift_first_to_last": float(np.mean(last) - np.mean(first)),
           "stop_or": h[0] if h else None, "stop_p": h[1] if h else None, "n_steps": len(rows),
           "coverage": {"judged": len(seen), "expected": len(man), "malformed": len(bad),
                        "empty": sum(1 for v in traj.values() if not v)}}

    print(f"\n{'arm':12s}{'n':>5s}{'med #est':>10s}{'mean #est':>11s}{'med tokens':>12s}"
          f"{'first fav':>12s}{'stop OR':>10s}{'p':>8s}")
    print("-" * 80)
    print(f"{'minimal':12s}{len(ids):5d}{st.median(n_est):10.0f}{np.mean(n_est):11.1f}"
          f"{st.median(toks):12.0f}{np.mean(first):12.3f}"
          f"{(f'{h[0]:.2f}' if h else '—'):>10s}{(f'{h[1]:.3f}' if h else '—'):>8s}")
    for k, r in REF.items():
        print(f"{k+' (ref)':12s}{'':5s}{r['med_est']:10d}{'':11s}{r['med_tok']:12d}"
              f"{r['first_fav']:12.3f}{r['stop_or']:10.2f}{r['stop_p']:8.4f}")

    print("\n=== revision DIRECTION ===")
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
            l2, h2 = sps.binomtest(int(sum(tw)), len(tw), 0.5).proportion_ci(0.95)
            cells[want] = f"{np.mean(tw):.3f} [{l2:.2f},{h2:.2f}] n={len(tw)}" \
                          f"{'' if l2 <= 0.5 <= h2 else ' *'}"
        else:
            cells[want] = "—"
    res["revision_direction"] = {"bad_side": cells[False], "good_side": cells[True]}
    print(f"  minimal   on BAD side {cells[False]}   on GOOD side {cells[True]}")
    print(f"  refs      free CoT bad side {REF['free_cot']['bad_pull']}, "
          f"neutral {REF['neutral']['bad_pull']}")
    print(f"\n  drift first->last: {res['drift_first_to_last']:+.3f}")

    print("\n=== verdict vs the A9 prediction ('minimal matches free CoT') ===")
    print(f"  revision count: {st.median(n_est):.0f} vs free {REF['free_cot']['med_est']} / "
          f"neutral {REF['neutral']['med_est']}")
    print(f"  stopping asymmetry: OR {h[0]:.2f} (p={h[1]:.3f}) vs free "
          f"{REF['free_cot']['stop_or']} (p={REF['free_cot']['stop_p']}) / "
          f"neutral {REF['neutral']['stop_or']} (p={REF['neutral']['stop_p']})" if h else "  —")

    (RUN / "a9_minimal_traj.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {RUN/'a9_minimal_traj.json'}")


if __name__ == "__main__":
    main()
