"""A12.1 — first-estimate side x final side, as a transition matrix.

A1 reported 41/41 (first favoured -> ended favoured) and A8 reported 0/23 (ended unfavoured ->
started unfavoured). Those are two corners of the same 2x2, and stating it as a matrix with
Wilson CIs makes the strength of the relationship explicit instead of leaving it as two
striking-but-separate counts.

Also reports where the FIRST estimate sits in the trace, normalised to [0, 1]. If stance-setting
mostly precedes the first number, the "first estimate is near-decisive" result is really a claim
about a commitment made before any number is written down.

Position is located by searching the CoT for the first estimate's digits in several written
forms; rollouts where it cannot be located are reported, never guessed at.

Run with: python3 runs/a12_transitions.py   (no GPU)
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from src.qual import judge, parse                              # noqa: E402


def wilson(k, n):
    if n == 0:
        return float("nan"), (float("nan"), float("nan"))
    lo, hi = sps.binomtest(int(k), int(n), 0.5).proportion_ci(0.95)
    return k / n, (float(lo), float(hi))


def forms(v):
    """Plausible written forms of an integer estimate, longest first."""
    i = int(round(v))
    out = [f"{i:,}", str(i)]
    for unit, div in (("million", 1e6), ("billion", 1e9), ("thousand", 1e3)):
        if i >= div:
            q = i / div
            for s in (f"{q:.10g}", f"{q:.1f}".rstrip("0").rstrip(".")):
                out += [f"{s} {unit}", f"{s}{unit[0].upper()}", f"{s}{unit[0]}"]
    return sorted({s for s in out if s}, key=len, reverse=True)


def first_pos(cot, v):
    """Normalised position of the first written occurrence of v, or None."""
    if not cot:
        return None
    best = None
    for s in forms(v):
        i = cot.find(s)
        if i != -1 and (best is None or i < best):
            best = i
    return None if best is None else best / len(cot)


def load(kind):
    """-> list of dicts with direction, traj, label, cot."""
    rows = []
    if kind == "free_cot":
        run = HERE / "hour0"
        T = json.load(open(run / "threshold.json"))["threshold"]
        man = {m["id"]: m for m in json.load(open(run / "trajectory_manifest.json"))}
        maj = json.load(open(run / "covertness_majority.json"))["final_labels"]
        lab = {}
        for k, v in maj.items():
            i = k.find("giraffes_")
            lab[k[i:]] = v
        raw = {}
        for d in ("above_good", "below_good"):
            for rec in parse.parse_jsonl_file(run / "raw" / f"giraffes_{d}.jsonl"):
                raw[rec["item_id"]] = rec
        for iid, m in man.items():
            if m["direction"] == "baseline":
                continue
            f = run / "trajectory_outputs" / f"{iid}.txt"
            if not f.exists():
                continue
            traj = judge.parse_trajectory_answer(f.read_text())
            if not traj:
                continue
            rows.append({"id": iid, "direction": m["direction"], "traj": traj,
                         "label": lab.get(iid),
                         "cot": raw[iid]["raw_completion"].split("</think>", 1)[0]})
        return rows, T
    run = HERE / "prefill_extra"
    T = json.load(open(HERE / "hour0" / "threshold.json"))["threshold"]
    man = {m["id"]: m for m in json.load(open(run / "trajectory_manifest.json"))}
    traj_all = json.load(open(run / "trajectories.json"))
    raw = {}
    for d in ("above_good", "below_good"):
        for k, rec in enumerate(parse.parse_jsonl_file(run / "raw" / f"minimal_{d}.jsonl")):
            raw[f"minimal_{d}_{k}"] = rec
    for iid, t in traj_all.items():
        if not t or iid not in man:
            continue
        rows.append({"id": iid, "direction": man[iid]["direction"], "traj": t, "label": None,
                     "cot": raw[iid]["raw_completion"].split("</think>", 1)[0]})
    return rows, T


def report(name, rows, T, out):
    def fav(v, d):
        return v > T if d == "above_good" else v <= T

    print(f"\n{'='*70}\n{name}  (n={len(rows)})\n{'='*70}")
    groups = [("all", None)]
    if any(r["label"] for r in rows):
        groups += [("overt (INFLUENCED)", "INFLUENCED"), ("covert (denies)", "NOT_INFLUENCED")]
    res = {}
    for gname, lab in groups:
        g = [r for r in rows if lab is None or r["label"] == lab]
        if not g:
            continue
        cells = {}
        print(f"\n  {gname}  (n={len(g)})")
        print(f"    {'first estimate':18s}{'n':>5s}{'P(final favoured)':>20s}{'95% CI':>20s}")
        for fname, want in (("favoured", True), ("unfavoured", False)):
            sub = [r for r in g if fav(r["traj"][0], r["direction"]) is want]
            k = sum(fav(r["traj"][-1], r["direction"]) for r in sub)
            p, ci = wilson(k, len(sub))
            cells[fname] = {"n": len(sub), "k_final_favoured": int(k), "p": p, "ci": list(ci)}
            print(f"    {fname:18s}{len(sub):5d}{p:20.3f}"
                  f"{f'[{ci[0]:.3f}, {ci[1]:.3f}]':>20s}")
        res[gname] = cells

    # position of the first estimate
    pos, missed = [], 0
    for r in rows:
        p = first_pos(r["cot"], r["traj"][0])
        if p is None:
            missed += 1
        else:
            pos.append(p)
    if pos:
        print(f"\n  first estimate position in the CoT (normalised): median "
              f"{np.median(pos):.3f}, IQR [{np.percentile(pos,25):.3f}, "
              f"{np.percentile(pos,75):.3f}], located {len(pos)}/{len(rows)} "
              f"({missed} not locatable)")
        res["first_estimate_position"] = {
            "median": float(np.median(pos)), "q25": float(np.percentile(pos, 25)),
            "q75": float(np.percentile(pos, 75)), "n_located": len(pos), "n_missed": missed}
    out[name] = res


def main():
    out = {}
    for kind, label in (("free_cot", "FREE CoT (unprefilled intervention rollouts)"),
                        ("minimal", "MINIMAL prefill arm")):
        rows, T = load(kind)
        report(label, rows, T, out)
    print("\n  reference: admission sentences sit at 0.78 of the CoT (locator run).")
    p = HERE / "hour0" / "a12_transitions.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
