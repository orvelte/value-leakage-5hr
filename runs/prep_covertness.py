"""Regenerate covertness judge inputs (paper App. E.2.1, verbatim via judge.py) and a batch
assignment for a fan-out pass.

Inputs are gitignored because they are exactly reproducible from raw/ + the frozen prompt; only
the outputs are precious. `--pass-name` lets a re-judge write its batches and outputs beside the
first pass instead of overwriting it, so the two can be compared for agreement.

Run with: python3 runs/prep_covertness.py [--run hour0] [--pass-name pass2] [--seed 1]
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.qual import judge, parse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="hour0")
    ap.add_argument("--pass-name", default=None,
                    help="suffix for batches/outputs dirs, e.g. pass2; omit for the first pass")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0,
                    help="shuffles the item->agent assignment; use a NEW seed per pass so an "
                         "item is judged by a different instance each time")
    args = ap.parse_args()

    run_dir = Path(__file__).resolve().parent / args.run
    threshold = json.load(open(run_dir / "threshold.json"))["threshold"]
    sfx = f"_{args.pass_name}" if args.pass_name else ""

    in_dir = run_dir / "judge_inputs"
    in_dir.mkdir(exist_ok=True)
    (run_dir / f"judge_outputs{sfx}").mkdir(exist_ok=True)
    b_dir = run_dir / f"judge_batches{sfx}"
    b_dir.mkdir(exist_ok=True)

    manifest = []
    for direction in ["above_good", "below_good"]:
        for r in parse.parse_jsonl_file(run_dir / "raw" / f"giraffes_{direction}.jsonl"):
            val, reason = parse.parse_estimate(r["raw_completion"])
            good = parse.classify_on_good_side(val, threshold, direction) if val is not None else None
            cot = r["raw_completion"].split("</think>", 1)[0]
            item_id = f"covertness_{direction}_{r['item_id']}"
            (in_dir / f"{item_id}.txt").write_text(judge.build_covertness_prompt(cot))
            manifest.append({"id": item_id, "kind": "covertness", "direction": direction,
                             "source_item_id": r["item_id"], "estimate": val,
                             "on_good_side": good, "parse_reason": reason})
    (run_dir / "judge_manifest.json").write_text(json.dumps(manifest, indent=2))

    ids = [m["id"] for m in manifest]
    random.Random(args.seed).shuffle(ids)
    for i in range(args.batches):
        chunk = ids[i::args.batches]
        (b_dir / f"batch{i}.txt").write_text("\n".join(chunk))
        n_ab = sum("above" in c for c in chunk)
        print(f"  batch{i}: n={len(chunk)} (above={n_ab}, below={len(chunk)-n_ab})")
    print(f"\n{len(manifest)} inputs -> {in_dir}\nbatches -> {b_dir}\n"
          f"outputs expected in -> {run_dir / f'judge_outputs{sfx}'}")


if __name__ == "__main__":
    main()
