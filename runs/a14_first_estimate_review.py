"""A14 — pull first-estimate extractions for hand review.

A12.1 and A13 both rest entirely on which side the FIRST extracted estimate falls on. If the
judge missed an earlier candidate number, the first side flips and both results move. This dumps
a stratified sample with enough context to check that by eye.

Sampling is stratified and seeded, not cherry-picked:
  6 free-CoT overt, 6 free-CoT covert, 8 baseline.
Baseline gets the largest share because A13 is the newest claim and its cells are only n=15.

For each case the file shows the CoT from the very beginning — the failure mode is a MISSED
earlier estimate, so the opening is the part that matters — through the located first estimate,
plus the extracted trajectory so the sequence can be sanity-checked.

Run with: python3 runs/a14_first_estimate_review.py [--n-per-stratum 6 6 8] [--seed 0]
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from src.qual import judge, parse                              # noqa: E402
from a12_transitions import forms, first_pos                   # noqa: E402

RUN = HERE / "hour0"
HEAD_CHARS = 1100          # always show this much of the CoT opening
CTX_BEFORE, CTX_AFTER = 700, 260


def excerpt(cot, pos_char):
    """CoT opening, then (if the match is past the opening) a window around the match."""
    head = cot[:HEAD_CHARS]
    if pos_char is None:
        return head + ("\n\n[...] (first estimate could not be located in the text)", )[0]
    if pos_char < HEAD_CHARS:
        return head + ("\n\n[... rest of trace omitted ...]" if len(cot) > HEAD_CHARS else "")
    a = max(HEAD_CHARS, pos_char - CTX_BEFORE)
    return (head + f"\n\n[... {a - HEAD_CHARS:,} chars omitted ...]\n\n"
            + cot[a:pos_char + CTX_AFTER] + "\n\n[... rest of trace omitted ...]")


def locate_char(cot, v):
    best = None
    for s in forms(v):
        i = cot.find(s)
        if i != -1 and (best is None or i < best):
            best = i
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/hour0/first_estimate_review.md")
    args = ap.parse_args()
    import random
    rng = random.Random(args.seed)

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

    pool = {"overt": [], "covert": [], "baseline": []}
    for iid, m in man.items():
        f = RUN / "trajectory_outputs" / f"{iid}.txt"
        if not f.exists() or iid not in raw:
            continue
        t = judge.parse_trajectory_answer(f.read_text())
        t = [v for v in t or [] if v != 0]
        if not t:
            continue
        d = m["direction"]
        if d == "baseline":
            k = "baseline"
        else:
            L = lab.get(iid)
            k = "overt" if L == "INFLUENCED" else "covert" if L == "NOT_INFLUENCED" else None
        if k:
            pool[k].append({"id": iid, "direction": d, "traj": t, "label": lab.get(iid),
                            "cot": raw[iid]["raw_completion"].split("</think>", 1)[0],
                            "final": parse.parse_estimate(raw[iid]["raw_completion"])[0]})

    want = {"overt": 6, "covert": 6, "baseline": 8}
    picked = []
    for k, n in want.items():
        picked += rng.sample(pool[k], min(n, len(pool[k])))

    lines = [f"# First-estimate extractions for hand review",
             "", f"Threshold **T = {T:,.0f}**. Sample is stratified and seeded "
             f"(`--seed {args.seed}`): {want['overt']} free-CoT overt, {want['covert']} "
             f"free-CoT covert, {want['baseline']} baseline. Zeros dropped per the A12.4 rule.",
             "", "**What to check:** does any single-number estimate of the target appear in the "
             "trace BEFORE the one marked as first? A missed earlier candidate flips the "
             "first-estimate side, which is what A12.1 and A13 are built on.", "",
             "| # | id | condition | label | first est | first | final est | final | #est | pos |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    def side(v):
        return ">T" if v > T else "<=T"

    def favmark(v, d):
        """For below_good, FAVOURED is <=T -- the inversion is the easiest thing to misread by
        hand, so it is spelled out rather than left to the reader."""
        if d == "baseline":
            return side(v)
        f = (v > T) if d == "above_good" else (v <= T)
        return f"{side(v)} = {'FAV' if f else 'unfav'}"

    for i, r in enumerate(picked, 1):
        pos = first_pos(r["cot"], r["traj"][0])
        lines.append(
            f"| {i} | `{r['id']}` | {r['direction']} | {r['label'] or '—'} | "
            f"{r['traj'][0]:,.0f} | {favmark(r['traj'][0], r['direction'])} | "
            f"{r['final']:,.0f} | {favmark(r['final'], r['direction'])} | {len(r['traj'])} | "
            f"{(f'{pos:.2f}' if pos is not None else 'n/a')} |")
    lines.append("")
    lines += ["> For `below_good`, **favoured is <= T** — the bet pays the good cause when the "
              "estimate does NOT exceed the threshold. The FAV/unfav labels above already "
              "account for that; `baseline` has no favoured side.", ""]

    for i, r in enumerate(picked, 1):
        c = locate_char(r["cot"], r["traj"][0])
        pos = first_pos(r["cot"], r["traj"][0])
        seq = ", ".join(f"{v:,.0f}" for v in r["traj"][:10])
        lines += [f"---", "",
                  f"## {i}. `{r['id']}` — {r['direction']}"
                  f"{'' if r['label'] is None else ' / ' + r['label']}", "",
                  f"- **first extracted estimate: {r['traj'][0]:,.0f}** "
                  f"({favmark(r['traj'][0], r['direction'])})",
                  f"- final estimate: {r['final']:,.0f} "
                  f"({favmark(r['final'], r['direction'])})",
                  f"- trajectory ({len(r['traj'])} estimates, first 10): {seq}"
                  + (" ..." if len(r["traj"]) > 10 else ""),
                  f"- located at char {c if c is not None else 'NOT FOUND'}"
                  + (f" of {len(r['cot']):,} ({pos:.3f} of the trace)" if pos is not None else ""),
                  f"- CoT length: {len(r['cot']):,} chars", "",
                  "```text", excerpt(r["cot"], c), "```", ""]

    out = Path(args.out)
    out.write_text("\n".join(lines))
    print(f"{len(picked)} cases -> {out}")
    print(f"\n{'#':>3s}  {'id':34s}{'label':16s}{'first':>14s}{'first side':>14s}"
          f"{'final':>14s}{'final side':>14s}{'pos':>7s}")
    for i, r in enumerate(picked, 1):
        pos = first_pos(r["cot"], r["traj"][0])
        print(f"{i:3d}  {r['id']:34s}{(r['label'] or '—'):16s}{r['traj'][0]:14,.0f}"
              f"{favmark(r['traj'][0], r['direction']):>14s}{r['final']:14,.0f}"
              f"{favmark(r['final'], r['direction']):>14s}"
              f"{(f'{pos:.2f}' if pos is not None else 'n/a'):>7s}")


if __name__ == "__main__":
    main()
