"""A9 — trajectory extraction on the `minimal` arm (judge calls only, no GPU).

A2 covered neutral/denial/admission. `minimal` is the arm that matters for A9's third prediction:
"Okay." keeps free-CoT length (7948 median think tokens vs ~8615), so if the revision loop is a
function of the trace rather than of the prefill, minimal should look like free CoT — ~24
intermediate estimates and H2's stopping asymmetry intact — not like the ~4 of the short arms.

Batches are smaller than A2's because these CoTs are ~8x longer; the per-agent reading load is
what matters, not the rollout count.

Run with: python3 runs/a9_prep_minimal_traj.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.qual import judge, parse

HERE = Path(__file__).resolve().parent
RUN = HERE / "prefill_extra"
ARMS = ["minimal"]
DIRS = ["above_good", "below_good"]
N_BATCHES = 8


def main():
    T = json.load(open(HERE / "hour0" / "threshold.json"))["threshold"]
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

    for i in range(N_BATCHES):
        chunk = ids[i::N_BATCHES]
        (b_dir / f"tbatch{i}.txt").write_text("\n".join(chunk))
        print(f"  tbatch{i}: n={len(chunk)}")
    print(f"\n{len(ids)} inputs -> {in_dir}\nbatches -> {b_dir}")


if __name__ == "__main__":
    main()
