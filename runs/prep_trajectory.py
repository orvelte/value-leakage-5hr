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
    ap.add_argument("--batches", type=int, default=8)
    args = ap.parse_args()

    run_dir = Path(__file__).resolve().parent / args.run
    cov = {r["id"].replace(f"covertness_{r['direction']}_", ""): r
           for r in json.load(open(run_dir / "covertness_results.json"))}

    out_dir = run_dir / "trajectory_inputs"
    out_dir.mkdir(exist_ok=True)
    (run_dir / "trajectory_outputs").mkdir(exist_ok=True)

    manifest = []
    for direction in ["above_good", "below_good"]:
        for rec in parse.parse_jsonl_file(run_dir / "raw" / f"giraffes_{direction}.jsonl"):
            c = cov.get(rec["item_id"])
            if c is None or c["raw"] not in ("INFLUENCED", "NOT_INFLUENCED"):
                continue
            if args.good_side_only and not c["on_good_side"]:
                continue
            cot = rec["raw_completion"].split("</think>", 1)[0]
            (out_dir / f"{rec['item_id']}.txt").write_text(judge.build_trajectory_prompt(cot))
            manifest.append({"id": rec["item_id"], "direction": direction,
                             "judge_raw": c["raw"], "on_good_side": c["on_good_side"],
                             "final_estimate": c["estimate"]})

    (run_dir / "trajectory_manifest.json").write_text(json.dumps(manifest, indent=2))

    ids = [m["id"] for m in manifest]
    bdir = run_dir / "trajectory_batches"   # persisted with the run, not in session scratch
    bdir.mkdir(parents=True, exist_ok=True)
    for i in range(args.batches):
        chunk = ids[i::args.batches]
        (bdir / f"tbatch{i}.txt").write_text("\n".join(chunk))
        n_ov = sum(1 for c in chunk if cov[c]["raw"] == "INFLUENCED")
        print(f"tbatch{i}: n={len(chunk)} (overt={n_ov}, covert={len(chunk)-n_ov})")
    print(f"\n{len(manifest)} inputs -> {out_dir}\nbatches -> {bdir}")


if __name__ == "__main__":
    main()
