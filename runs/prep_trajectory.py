"""Write trajectory-extraction judge inputs (paper App. E.5.1, verbatim via judge.py).

The Figure 5 trajectory analysis extracts every intermediate single-number estimate a rollout
floats, in order. That gives two things the raw JSONL cannot: how many candidate answers a
rollout entertained, and where its FIRST estimate fell relative to the threshold -- the
"is the leak upstream of the verbalized reasoning?" question from the plan.

Unlike the covertness judge, this one is NOT number-blurred: it needs the numbers.

Run with: python3 runs/prep_trajectory.py [--run hour0] [--good-side-only]
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.qual import judge, parse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="hour0")
    ap.add_argument("--good-side-only", action="store_true")
    ap.add_argument("--conditions", default="above_good,below_good",
                    help="comma-separated; include 'baseline' for the no-incentive null, which "
                         "is what makes the revision-asymmetry test interpretable")
    ap.add_argument("--only-missing", action="store_true",
                    help="skip items that already have a trajectory output")
    ap.add_argument("--batches", type=int, default=8)
    args = ap.parse_args()

    run_dir = Path(__file__).resolve().parent / args.run
    cov = {r["id"].replace(f"covertness_{r['direction']}_", ""): r
           for r in json.load(open(run_dir / "covertness_results.json"))}
    done = {p.stem for p in (run_dir / "trajectory_outputs").glob("*.txt")}

    out_dir = run_dir / "trajectory_inputs"
    out_dir.mkdir(exist_ok=True)
    (run_dir / "trajectory_outputs").mkdir(exist_ok=True)

    manifest, todo = [], []
    for direction in args.conditions.split(","):
        f = run_dir / "raw" / f"giraffes_{direction}.jsonl"
        for rec in parse.parse_jsonl_file(f):
            c = cov.get(rec["item_id"])
            if direction != "baseline":
                # baseline has no covertness label by design: no threshold is mentioned
                if c is None or c["raw"] not in ("INFLUENCED", "NOT_INFLUENCED"):
                    continue
                if args.good_side_only and not c["on_good_side"]:
                    continue
            entry = {"id": rec["item_id"], "direction": direction,
                     "judge_raw": c["raw"] if c else None,
                     "on_good_side": c["on_good_side"] if c else None,
                     "final_estimate": c["estimate"] if c else
                                       parse.parse_estimate(rec["raw_completion"])[0]}
            manifest.append(entry)
            if args.only_missing and rec["item_id"] in done:
                continue
            cot = rec["raw_completion"].split("</think>", 1)[0]
            (out_dir / f"{rec['item_id']}.txt").write_text(judge.build_trajectory_prompt(cot))
            todo.append(rec["id"] if "id" in rec else rec["item_id"])

    # merge rather than overwrite, so earlier passes are not lost
    mpath = run_dir / "trajectory_manifest.json"
    existing = {m["id"]: m for m in json.load(open(mpath))} if mpath.exists() else {}
    existing.update({m["id"]: m for m in manifest})
    mpath.write_text(json.dumps(list(existing.values()), indent=2))

    b_dir = run_dir / "trajectory_batches"
    b_dir.mkdir(exist_ok=True)
    for f in b_dir.glob("tbatch*.txt"):
        f.unlink()
    n_b = min(args.batches, max(1, len(todo)))
    for i in range(n_b):
        chunk = todo[i::n_b]
        (b_dir / f"tbatch{i}.txt").write_text("\n".join(chunk))
        print(f"  tbatch{i}: n={len(chunk)}")
    print(f"\n{len(manifest)} in manifest, {len(todo)} need extraction "
          f"({len(manifest)-len(todo)} already done)\ninputs -> {out_dir}\nbatches -> {b_dir}")


if __name__ == "__main__":
    main()
