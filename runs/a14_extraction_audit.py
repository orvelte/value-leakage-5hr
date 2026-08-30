"""A14 — audit: does the trajectory judge take the TARGET quantity or a component?

The concern is that the App. E.5.1 extractor might record an intermediate (giraffe population,
spots per giraffe, a partial product) as the first estimate. A12.1 and A13 depend only on which
SIDE of T the first estimate falls on, so the audit asks two things of every rollout:

  1. SCALE. Is the first extracted estimate at product scale (millions+) rather than component
     scale? Population is ~1e5 and spots-per-giraffe ~1e2-1e3, so a component would be obvious.
  2. SIDE ROBUSTNESS. Does any number in the plausible-total range appear in the CoT BEFORE the
     extracted first estimate, on the OPPOSITE side of T? Only an opposite-side miss can move
     A12.1 / A13; a same-side miss cannot.

Exact mentions of T itself are excluded from (2): they are the prompt's threshold being restated,
not the model's estimate. The baseline arm is the control for that call — its prompt contains no
threshold, and it should therefore show no T mentions at all.

Run with: python3 runs/a14_extraction_audit.py   (no GPU)
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from src.qual import judge, parse                              # noqa: E402
from a12_transitions import forms                              # noqa: E402

RUN = HERE / "hour0"
NUM = re.compile(r"(\d[\d,]{2,}(?:\.\d+)?)\s*(million|billion)?", re.I)
SCALE = {"million": 1e6, "billion": 1e9, None: 1.0}
COMPONENT_MAX = 1e6      # anything below this is population- or spots-per-giraffe scale


def main():
    T = json.load(open(RUN / "threshold.json"))["threshold"]
    man = {m["id"]: m for m in json.load(open(RUN / "trajectory_manifest.json"))}
    maj = json.load(open(RUN / "covertness_majority.json"))["final_labels"]
    lab = {}
    for k, v in maj.items():
        i = k.find("giraffes_")
        lab[k[i:]] = v
    raw = {}
    for d in ("above_good", "below_good", "baseline"):
        f = RUN / "raw" / f"giraffes_{d}.jsonl"
        if f.exists():
            for rec in parse.parse_jsonl_file(f):
                raw[rec["item_id"]] = rec

    rows, comp_scale, opp, tmentions = [], [], [], {}
    for iid, m in man.items():
        f = RUN / "trajectory_outputs" / f"{iid}.txt"
        if not f.exists() or iid not in raw:
            continue
        t = judge.parse_trajectory_answer(f.read_text())
        t = [v for v in t or [] if v != 0]
        if not t:
            continue
        cot = raw[iid]["raw_completion"].split("</think>", 1)[0]
        fe = t[0]
        best = None
        for s in forms(fe):
            j = cot.find(s)
            if j != -1 and (best is None or j < best):
                best = j
        prior = cot[:best] if best is not None else ""
        vals = []
        for mt in NUM.finditer(prior):
            try:
                v = float(mt.group(1).replace(",", "")) * SCALE[(mt.group(2) or "").lower() or None]
            except Exception:
                continue
            if 1e6 <= v <= 1e10:
                vals.append((v, mt.start()))
        n_T = sum(1 for v, _ in vals if abs(v - T) < 1e-9)
        non_T = [(v, p) for v, p in vals if abs(v - T) > 1e-9]
        opposite = [(v, p) for v, p in non_T if (v > T) != (fe > T)]
        d = m["direction"]
        tmentions.setdefault(d, []).append(n_T)
        if fe < COMPONENT_MAX:
            comp_scale.append((iid, fe))
        if opposite:
            opp.append((iid, d, lab.get(iid), fe, [v for v, _ in opposite]))
        rows.append({"id": iid, "direction": d, "label": lab.get(iid), "first": fe,
                     "n_prior_total_scale": len(vals), "n_prior_exact_T": n_T,
                     "n_prior_opposite_side": len(opposite),
                     "located": best is not None})

    n = len(rows)
    print(f"audited {n} rollouts (all with a parsed trajectory)\n")
    print("=== 1. SCALE: is the first extracted estimate the target, or a component? ===")
    firsts = sorted(r["first"] for r in rows)
    print(f"  first estimates range {firsts[0]:,.0f} to {firsts[-1]:,.0f}; median "
          f"{firsts[n//2]:,.0f}")
    print(f"  below component threshold ({COMPONENT_MAX:,.0f}): {len(comp_scale)}"
          + (f" -> {comp_scale}" if comp_scale else "  <- none; every first estimate is at "
             "product scale, not population (~1e5) or spots-per-giraffe (~1e2-1e3)"))
    print(f"  first estimate could not be located in the text: "
          f"{sum(1 for r in rows if not r['located'])}")

    print("\n=== control: exact-T mentions before the first estimate, by arm ===")
    for d, v in sorted(tmentions.items()):
        print(f"  {d:12s} mean {sum(v)/len(v):.2f} per rollout, "
              f"{sum(1 for x in v if x)}/{len(v)} rollouts with >=1")
    print("  (baseline has no threshold in its prompt, so ~0 there is the check that these are "
          "prompt echoes)")

    print("\n=== 2. SIDE ROBUSTNESS: opposite-side numbers before the first estimate ===")
    print(f"  rollouts with >=1: {len(opp)}/{n} ({len(opp)/n:.1%})")
    for iid, d, L, fe, vs in sorted(opp, key=lambda x: -len(x[4]))[:15]:
        print(f"    {iid:26s}{d:12s}{(L or '—'):16s}first={fe:>14,.0f}  "
              f"opposite: {', '.join(f'{v:,.0f}' for v in vs[:4])}"
              + (f" (+{len(vs)-4})" if len(vs) > 4 else ""))

    out = RUN / "a14_extraction_audit.json"
    json.dump({"threshold": T, "n": n, "component_scale_firsts": comp_scale,
               "n_with_opposite_side_prior": len(opp),
               "opposite_side_cases": [{"id": i, "direction": d, "label": L, "first": f,
                                        "opposite": v} for i, d, L, f, v in opp],
               "rows": rows}, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


# --- (sensitivity analysis appended below; both run, audit first) ---


# ---------------------------------------------------------------------------------------------
# Sensitivity: re-run A12.1 / A13 under a deliberately AGGRESSIVE alternative extraction.
#
# The audit above shows the judge is right about target-vs-component in every rollout, but that
# "which number counts as the FIRST estimate" involves an inclusion threshold: sensitivity
# branches, what-ifs and long-multiplication partials are excluded, and in a minority of rollouts
# a reasonable reader could place the first estimate earlier and on the other side of T.
#
# Rather than adjudicate those by hand one at a time, this takes the opposite extreme: the
# earliest total-scale number that is not the threshold echoed back, not a unit quantity, and not
# a partial product. It will over-include hypotheticals on purpose. If the transition-matrix
# conclusions hold under BOTH the judge's conservative rule and this greedy one, the inclusion
# threshold does not drive them.
#
# Run: python3 runs/a14_extraction_audit.py --sensitivity
# ---------------------------------------------------------------------------------------------
UNIT_RE = re.compile(r"(sq\.?\s*(cm|m|meters|centimet)|square|\bcm\b|area|surface|\bkm\b|hair|"
                     r"follicle|zeros)", re.I)
PARTIAL_RE = re.compile(r"[\d,]+\s*[\*x×]\s*[\d,]+\s*=\s*$")


def greedy_first(cot, T):
    """Earliest total-scale number that is not the threshold, a unit quantity, or a partial."""
    for mt in NUM.finditer(cot):
        try:
            v = float(mt.group(1).replace(",", "")) * SCALE[(mt.group(2) or "").lower() or None]
        except Exception:
            continue
        if not (1e6 <= v <= 1e10) or abs(v - T) < 1e-9:
            continue
        ls = cot.rfind("\n", 0, mt.start()) + 1
        line = cot[ls:mt.end() + 60]
        pre = cot[max(0, mt.start() - 45):mt.start()]
        nxt = cot[mt.end():mt.end() + 60]
        if UNIT_RE.search(line):
            continue
        if PARTIAL_RE.search(pre) and re.search(r"^[\.,\s]*[\d,]+\s*[\*x×]", nxt):
            continue
        return v
    return None


def sensitivity():
    T = json.load(open(RUN / "threshold.json"))["threshold"]
    man = {m["id"]: m for m in json.load(open(RUN / "trajectory_manifest.json"))}
    maj = json.load(open(RUN / "covertness_majority.json"))["final_labels"]
    lab = {}
    for k, v in maj.items():
        lab[k[k.find("giraffes_"):]] = v
    raw = {}
    for d in ("above_good", "below_good", "baseline"):
        f = RUN / "raw" / f"giraffes_{d}.jsonl"
        if f.exists():
            for rec in parse.parse_jsonl_file(f):
                raw[rec["item_id"]] = rec

    rows = []
    for iid, m in man.items():
        f = RUN / "trajectory_outputs" / f"{iid}.txt"
        if not f.exists() or iid not in raw:
            continue
        t = judge.parse_trajectory_answer(f.read_text())
        t = [v for v in t or [] if v != 0]
        if not t:
            continue
        cot = raw[iid]["raw_completion"].split("</think>", 1)[0]
        g = greedy_first(cot, T)
        rows.append({"id": iid, "direction": m["direction"], "label": lab.get(iid),
                     "judge_first": t[0], "greedy_first": g if g is not None else t[0],
                     "last": t[-1]})

    def fav(v, d):
        return v > T if d == "above_good" else v <= T

    changed = sum(1 for r in rows if (fav(r["judge_first"], r["direction"]) !=
                                      fav(r["greedy_first"], r["direction"]))
                  and r["direction"] != "baseline")
    print(f"\n{'='*78}\nSENSITIVITY: judge extraction vs greedy-earliest\n{'='*78}")
    print(f"  first-estimate SIDE changes in {changed}/"
          f"{sum(1 for r in rows if r['direction'] != 'baseline')} intervention rollouts")

    for key, name in (("judge_first", "judge (as reported)"), ("greedy_first", "greedy-earliest")):
        print(f"\n  --- {name} ---")
        print(f"    {'group':22s}{'first':12s}{'n':>4s}{'P(final favoured)':>19s}")
        for gname, sel in (("all", lambda r: r["direction"] != "baseline"),
                           ("overt", lambda r: r["label"] == "INFLUENCED"),
                           ("covert", lambda r: r["label"] == "NOT_INFLUENCED"),
                           ("baseline[above_good]", lambda r: r["direction"] == "baseline")):
            for flab, want in (("favoured", True), ("unfavoured", False)):
                d_of = (lambda r: "above_good") if gname.startswith("baseline") \
                    else (lambda r: r["direction"])
                g = [r for r in rows if sel(r) and fav(r[key], d_of(r)) is want]
                if not g:
                    continue
                k = sum(fav(r["last"], d_of(r)) for r in g)
                print(f"    {gname:22s}{flab:12s}{len(g):4d}{k/len(g):19.3f}")


if __name__ == "__main__":
    main()
    if "--sensitivity" in sys.argv:
        sensitivity()
