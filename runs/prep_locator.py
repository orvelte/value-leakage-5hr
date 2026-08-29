"""Locate the admission sentence in overt rollouts, so resampling can truncate just before it.

H1's direct test needs a position: the sentence where the model states it is steering the number.
`src/experiments/locate_disclaimer.py` builds the prompt (numbers blurred, sentences numbered);
this writes one input per rollout plus a batch assignment for a judge fan-out.

Only rollouts whose MAJORITY covertness label is INFLUENCED are included — asking a locator to
find an admission in a trace that does not contain one produces a confident wrong answer, which
is exactly what `build_presence_prompt` exists to avoid elsewhere.

Run with: source env.sh && python3 runs/prep_locator.py [--max-items 24]
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.experiments import locate_disclaimer as loc
from src.qual import parse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="hour0")
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--max-items", type=int, default=24,
                    help="resampling downstream is expensive; cap the set and sample it seeded")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(__file__).resolve().parent / args.run
    final = json.load(open(run_dir / "covertness_majority.json"))["final_labels"]
    lab = {k.replace("covertness_above_good_", "").replace("covertness_below_good_", ""): v
           for k, v in final.items()}

    recs = {}
    for d in ["above_good", "below_good"]:
        for r in parse.parse_jsonl_file(run_dir / "raw" / f"giraffes_{d}.jsonl"):
            r["direction"] = d
            recs[r["item_id"]] = r

    overt = sorted(i for i in recs if lab.get(i) == "INFLUENCED")
    rng = random.Random(args.seed)
    rng.shuffle(overt)
    chosen = sorted(overt[: args.max_items])
    print(f"{len(overt)} overt rollouts; using {len(chosen)}")

    in_dir = run_dir / "locator_inputs"
    in_dir.mkdir(exist_ok=True)
    (run_dir / "locator_outputs").mkdir(exist_ok=True)
    manifest = []
    for iid in chosen:
        prompt, sentences = loc.build_locator_prompt(recs[iid]["raw_completion"], "INFLUENCED")
        (in_dir / f"{iid}.txt").write_text(prompt)
        manifest.append({"id": iid, "direction": recs[iid]["direction"],
                         "n_sentences": len(sentences)})
    (run_dir / "locator_manifest.json").write_text(json.dumps(manifest, indent=2))

    b_dir = run_dir / "locator_batches"
    b_dir.mkdir(exist_ok=True)
    for f in b_dir.glob("*.txt"):
        f.unlink()
    for i in range(args.batches):
        chunk = chosen[i::args.batches]
        (b_dir / f"lbatch{i}.txt").write_text("\n".join(chunk))
        print(f"  lbatch{i}: n={len(chunk)}")
    print(f"\ninputs -> {in_dir}\nbatches -> {b_dir}\noutputs expected -> "
          f"{run_dir/'locator_outputs'}")


if __name__ == "__main__":
    main()
