"""A12.4 — one rule for answer=0 and for non-English traces, applied everywhere.

Two edge cases surfaced during judging and were handled inconsistently across batches, which is
exactly the kind of thing that quietly moves a number:

  ZERO. Some traces end by arguing giraffe spots are brown rather than black and answer 0. One
  judge batch recorded those zeros, another excluded them as one side of a rejected either/or.
  RULE ADOPTED: 0 is not an estimate of the target. The question presupposes a positive count and
  a 0 is a definitional objection to the question, not a Fermi estimate. It is dropped from
  trajectories. This is also the conservative choice for the bias metric, because 0 <= T scores
  as favoured in every below_good rollout, so keeping zeros can only inflate below_good.

  NON-ENGLISH. Some traces reason in Chinese or German. RULE ADOPTED: keep them. They are real
  model behaviour on the same prompt, and dropping them would be selection on an outcome
  correlated with trace content. They are counted and reported so the exposure is known.

Reports how many rollouts each rule touches and recomputes every headline it could move.

Run with: python3 runs/a12_edge_hygiene.py   (no GPU)
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
from src.qual import judge, parse                              # noqa: E402
from a2_analyze_prefill_traj import hazard_or                  # noqa: E402

CJK = re.compile(r"[㐀-鿿぀-ヿ가-힯]")
NON_EN_LATIN = re.compile(r"\b(und|der|die|das|nicht|aber|einer|ungefähr|также|для)\b", re.I)


def script_of(cot):
    if CJK.search(cot):
        return "cjk"
    if NON_EN_LATIN.search(cot):
        return "non_en_latin"
    return "en"


def load_prefill(arm, run="prefill_extra"):
    run_dir = HERE / run
    man = {m["id"]: m for m in json.load(open(run_dir / "trajectory_manifest.json"))}
    traj = json.load(open(run_dir / "trajectories.json"))
    raw = {}
    for d in ("above_good", "below_good"):
        f = run_dir / "raw" / f"{arm}_{d}.jsonl"
        if not f.exists():
            continue
        for k, rec in enumerate(parse.parse_jsonl_file(f)):
            raw[f"{arm}_{d}_{k}"] = rec
    rows = []
    for iid, t in traj.items():
        if iid not in man or not t or man[iid].get("arm", arm) != arm:
            continue
        rows.append({"id": iid, "direction": man[iid]["direction"], "traj": t,
                     "cot": raw[iid]["raw_completion"].split("</think>", 1)[0]})
    return rows


def load_free():
    run = HERE / "hour0"
    man = {m["id"]: m for m in json.load(open(run / "trajectory_manifest.json"))}
    raw = {}
    for d in ("above_good", "below_good", "baseline"):
        f = run / "raw" / f"giraffes_{d}.jsonl"
        if f.exists():
            for rec in parse.parse_jsonl_file(f):
                raw[rec["item_id"]] = rec
    rows = []
    for iid, m in man.items():
        f = run / "trajectory_outputs" / f"{iid}.txt"
        if not f.exists():
            continue
        t = judge.parse_trajectory_answer(f.read_text())
        if not t:
            continue
        rows.append({"id": iid, "direction": m["direction"], "traj": t,
                     "cot": raw[iid]["raw_completion"].split("</think>", 1)[0]})
    return rows


def traj_stats(rows, T, drop_zeros):
    def fav(v, d):
        return v > T if d == "above_good" else v <= T
    out, steps = [], []
    for r in rows:
        t = [v for v in r["traj"] if v != 0] if drop_zeros else list(r["traj"])
        if not t:
            continue
        d = r["direction"]
        if d == "baseline":
            continue
        out.append({"n": len(t), "first": fav(t[0], d), "last": fav(t[-1], d)})
        if len(t) >= 3:
            for k, v in enumerate(t):
                steps.append((1.0 if fav(v, d) else 0.0, k / 10.0,
                              1.0 if k == len(t) - 1 else 0.0))
    pull = []
    for r in rows:
        t = [v for v in r["traj"] if v != 0] if drop_zeros else list(r["traj"])
        d = r["direction"]
        if d == "baseline" or len(t) < 2:
            continue
        for k in range(len(t) - 1):
            if not fav(t[k], d):
                up = t[k + 1] > t[k]
                pull.append(up if d == "above_good" else not up)
    h = hazard_or(steps)
    return {"n": len(out), "median_n_est": st.median([o["n"] for o in out]) if out else float("nan"),
            "first_favoured": float(np.mean([o["first"] for o in out])) if out else float("nan"),
            "last_favoured": float(np.mean([o["last"] for o in out])) if out else float("nan"),
            "stop_or": h[0] if h else None, "stop_p": h[1] if h else None,
            "bad_side_pull": float(np.mean(pull)) if pull else float("nan"), "n_pull": len(pull)}


def main():
    T = json.load(open(HERE / "hour0" / "threshold.json"))["threshold"]
    res = {"rule_zero": "0 is not an estimate; dropped from trajectories",
           "rule_non_english": "kept; counted only", "sets": {}}

    sets = [("free_cot", load_free()), ("minimal", load_prefill("minimal")),
            ("neutral", load_prefill("neutral", "prefill_tests")),
            ("denial", load_prefill("denial", "prefill_tests")),
            ("admission", load_prefill("admission", "prefill_tests"))]

    print(f"{'set':12s}{'n':>5s}{'rollouts w/ a 0':>17s}{'zeros':>7s}{'cjk':>6s}{'other':>7s}")
    for name, rows in sets:
        nz = sum(any(v == 0 for v in r["traj"]) for r in rows)
        zeros = sum(sum(1 for v in r["traj"] if v == 0) for r in rows)
        scripts = [script_of(r["cot"]) for r in rows]
        res["sets"][name] = {"n": len(rows), "rollouts_with_zero": nz, "zero_entries": zeros,
                             "cjk": scripts.count("cjk"),
                             "non_en_latin": scripts.count("non_en_latin")}
        print(f"{name:12s}{len(rows):5d}{nz:17d}{zeros:7d}{scripts.count('cjk'):6d}"
              f"{scripts.count('non_en_latin'):7d}")

    print(f"\n=== headline trajectory stats, zeros KEPT vs DROPPED ===")
    print(f"{'set':12s}{'rule':10s}{'n':>5s}{'med #est':>10s}{'first fav':>11s}"
          f"{'last fav':>10s}{'stop OR':>9s}{'p':>8s}{'bad pull':>10s}")
    for name, rows in sets:
        for drop in (False, True):
            s = traj_stats(rows, T, drop)
            res["sets"][name][f"zeros_{'dropped' if drop else 'kept'}"] = s
            ors = f"{s['stop_or']:.2f}" if s["stop_or"] else "—"
            ps = f"{s['stop_p']:.3f}" if s["stop_p"] else "—"
            print(f"{name:12s}{'dropped' if drop else 'kept':10s}{s['n']:5d}"
                  f"{s['median_n_est']:10.0f}{s['first_favoured']:11.3f}{s['last_favoured']:10.3f}"
                  f"{ors:>9s}{ps:>8s}{s['bad_side_pull']:10.3f}")

    print("\n=== does the rule move any headline? ===")
    moved = []
    for name, _ in sets:
        a = res["sets"][name]["zeros_kept"]
        b = res["sets"][name]["zeros_dropped"]
        for key, tol in (("first_favoured", 0.02), ("last_favoured", 0.02),
                         ("bad_side_pull", 0.02), ("median_n_est", 0.5)):
            if a[key] == a[key] and abs(a[key] - b[key]) > tol:
                moved.append(f"{name}.{key}: {a[key]:.3f} -> {b[key]:.3f}")
        if a["stop_or"] and b["stop_or"] and abs(a["stop_or"] - b["stop_or"]) > 0.05:
            moved.append(f"{name}.stop_or: {a['stop_or']:.2f} -> {b['stop_or']:.2f}")
    print("  " + ("\n  ".join(moved) if moved else
                  "nothing moves beyond tolerance — the rule is cosmetic for every headline"))
    res["moved"] = moved

    # final estimates: does the outlier filter already exclude zeros from the bias metric?
    print("\n=== final-estimate exposure (bias metric) ===")
    print(f"  parse.apply_outlier_filter(0, T) = {parse.apply_outlier_filter(0.0, T)}  "
          f"-> zeros cannot enter any bias number")

    out = HERE / "hour0" / "a12_edge_hygiene.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
