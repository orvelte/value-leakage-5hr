"""A15.2 — transition matrix on the `minimal` arm, as a check on the CROSSING RATE.

Reframed deliberately. A13 showed that "nobody who starts favoured ends unfavoured" is the base
rate for these traces rather than a ratchet, so this is no longer a confirmation of an asymmetry.
The question it can answer is narrower and still worth having: does PREFILLING change how often a
trace crosses the threshold at all?

That matters because the minimal arm has MORE intermediate estimates than free CoT (34 vs 24) at
comparable length, so if prefilling left the search intact one would expect a similar crossing
rate; if it flattens the trace into a single decomposition, fewer.

THE LIMITATION, up front: there is no prefilled NO-BET arm. The prefill runs were above_good and
below_good only. So "prefilled vs unprefilled" and "bet vs no bet" cannot be fully separated
here, and the baseline row below is unprefilled. Crossing rates are reported per trajectory AND
per intermediate estimate, because the arms differ several-fold in how many estimates they float
and a raw count would mostly measure that.

Run with: python3 runs/a15_minimal_transitions.py   (no GPU)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from src.qual import judge, parse                              # noqa: E402


def wilson(k, n):
    if n == 0:
        return float("nan"), (float("nan"), float("nan"))
    lo, hi = sps.binomtest(int(k), int(n), 0.5).proportion_ci(0.95)
    return k / n, (float(lo), float(hi))


def load_free(T):
    run = HERE / "hour0"
    man = {m["id"]: m for m in json.load(open(run / "trajectory_manifest.json"))}
    out = {"free_cot": [], "baseline": []}
    for iid, m in man.items():
        f = run / "trajectory_outputs" / f"{iid}.txt"
        if not f.exists():
            continue
        t = judge.parse_trajectory_answer(f.read_text())
        t = [v for v in t or [] if v != 0]
        if not t:
            continue
        key = "baseline" if m["direction"] == "baseline" else "free_cot"
        out[key].append({"direction": m["direction"], "traj": t})
    return out


def load_prefill(arm, run_name, T):
    run = HERE / run_name
    man = {m["id"]: m for m in json.load(open(run / "trajectory_manifest.json"))}
    traj = json.load(open(run / "trajectories.json"))
    rows = []
    for iid, t in traj.items():
        if iid not in man or not t:
            continue
        if man[iid].get("arm", arm) != arm:
            continue
        t = [v for v in t if v != 0]
        if t:
            rows.append({"direction": man[iid]["direction"], "traj": t})
    return rows


def crossings(t, T):
    s = [v > T for v in t]
    return sum(1 for a, b in zip(s, s[1:]) if a != b)


def main():
    T = json.load(open(HERE / "hour0" / "threshold.json"))["threshold"]
    free = load_free(T)
    sets = [("free CoT (bet)", free["free_cot"]),
            ("minimal (bet, prefilled)", load_prefill("minimal", "prefill_extra", T)),
            ("neutral (bet, prefilled)", load_prefill("neutral", "prefill_tests", T)),
            ("admission (bet, prefilled)", load_prefill("admission", "prefill_tests", T)),
            ("baseline (NO bet, unprefilled)", free["baseline"])]

    res = {"threshold": T, "sets": {}}
    print("=== crossing rate: how often does a trace move across T at all? ===")
    print(f"{'set':32s}{'n':>5s}{'med #est':>10s}{'>=1 crossing':>14s}"
          f"{'med crossings':>15s}{'crossings/est':>15s}")
    for name, rows in sets:
        c = [crossings(r["traj"], T) for r in rows]
        ne = [len(r["traj"]) for r in rows]
        rate = float(np.sum(c) / np.sum([max(n - 1, 1) for n in ne]))
        p, ci = wilson(sum(1 for x in c if x > 0), len(c))
        res["sets"][name] = {"n": len(rows), "median_n_est": float(np.median(ne)),
                             "frac_with_crossing": p, "frac_ci": list(ci),
                             "median_crossings": float(np.median(c)),
                             "crossings_per_step": rate}
        print(f"{name:32s}{len(rows):5d}{np.median(ne):10.0f}{p:14.3f}"
              f"{np.median(c):15.0f}{rate:15.3f}")

    print("\n=== transition matrix (favoured framing; baseline uses a FIXED above_good "
          "framing) ===")
    print(f"{'set':32s}{'first':12s}{'n':>5s}{'P(final favoured)':>19s}{'95% CI':>18s}")
    for name, rows in sets:
        def fav(v, d):
            if d == "baseline":
                return v > T
            return v > T if d == "above_good" else v <= T
        for flab, want in (("favoured", True), ("unfavoured", False)):
            g = [r for r in rows if fav(r["traj"][0], r["direction"]) is want]
            if not g:
                continue
            k = sum(fav(r["traj"][-1], r["direction"]) for r in g)
            p, ci = wilson(k, len(g))
            res["sets"][name][f"first_{flab}"] = {"n": len(g), "p_final_favoured": p,
                                                  "ci": list(ci)}
            print(f"{name:32s}{flab:12s}{len(g):5d}{p:19.3f}"
                  f"{f'[{ci[0]:.3f}, {ci[1]:.3f}]':>18s}")

    m = res["sets"]["minimal (bet, prefilled)"]
    f = res["sets"]["free CoT (bet)"]
    tab = [[int(round(f["frac_with_crossing"] * f["n"])),
            f["n"] - int(round(f["frac_with_crossing"] * f["n"]))],
           [int(round(m["frac_with_crossing"] * m["n"])),
            m["n"] - int(round(m["frac_with_crossing"] * m["n"]))]]
    odds, pv = sps.fisher_exact(tab)
    res["free_vs_minimal_crossing"] = {"fisher_or": float(odds), "fisher_p": float(pv)}
    print(f"\n  free CoT vs minimal, fraction with >=1 crossing: "
          f"{f['frac_with_crossing']:.3f} vs {m['frac_with_crossing']:.3f}, "
          f"Fisher OR {odds:.2f}, p={pv:.4f}")
    print("\n  CAVEAT: no prefilled no-bet arm exists, so prefilling and the bet are not fully "
          "separable here; the baseline row is unprefilled.")

    p = HERE / "hour0" / "a15_minimal_transitions.json"
    json.dump(res, open(p, "w"), indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
