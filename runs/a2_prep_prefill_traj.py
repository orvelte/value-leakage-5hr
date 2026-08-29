"""A2 — trajectory extraction on the prefill rollouts (judge calls only, no GPU).

Prefilled CoTs are ~1k tokens against ~9k for free CoT. The addendum's prediction is that they
therefore contain far fewer intermediate estimates and no room for H2's stopping rule to run,
except in the admission arm where the first estimate should already sit on the favoured side.

Run with: source env.sh && python3 runs/a2_prep_prefill_traj.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.qual import judge, parse

RUN = Path(__file__).resolve().parent / "prefill_tests"
ARMS = ["neutral", "denial", "admission"]
DIRS = ["above_good", "below_good"]


def main():
    T = json.load(open(Path(__file__).resolve().parent / "hour0" / "threshold.json"))["threshold"]
    in_dir = RUN / "trajectory_inputs"
    in_dir.mkdir(parents=True, exist_ok=True)
    (RUN / "trajectory_outputs").mkdir(exist_ok=True)
    b_dir = RUN / "trajectory_batches"
    b_dir.mkdir(exist_ok=True)
    for f in b_dir.glob("*.txt"):
        f.unlink()

    manifest, ids = [], []
    for arm in ARMS:
        for d in DIRS:
            f = RUN / "raw" / f"{arm}_{d}.jsonl"
            for k, rec in enumerate(parse.parse_jsonl_file(f)):
                iid = f"{arm}_{d}_{k}"
                cot = rec["raw_completion"].split("</think>", 1)[0]
                (in_dir / f"{iid}.txt").write_text(judge.build_trajectory_prompt(cot))
                v, _ = parse.parse_estimate(rec["raw_completion"])
                manifest.append({"id": iid, "arm": arm, "direction": d, "final_estimate": v,
                                 "n_tokens": rec["num_tokens"]})
                ids.append(iid)
    (RUN / "trajectory_manifest.json").write_text(json.dumps(manifest, indent=2))

    n_b = 6
    for i in range(n_b):
        chunk = ids[i::n_b]
        (b_dir / f"tbatch{i}.txt").write_text("\n".join(chunk))
        print(f"  tbatch{i}: n={len(chunk)}")
    print(f"\n{len(ids)} inputs -> {in_dir}\nbatches -> {b_dir}")


if __name__ == "__main__":
    main()
