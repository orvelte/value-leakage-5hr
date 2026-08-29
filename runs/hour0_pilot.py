"""Hour 0 gate check: does this model even show the covert variant?

Generates a small pilot (baseline + above_good + below_good, giraffes task), re-derives the
threshold from THIS model's own baseline median, and writes covertness-judge input files for
the intervention rollouts. It does NOT run the judge itself — that's a Claude-subagent fan-out
step done outside this script (see judge_inputs/ + judge_manifest.json below) — this script only
prepares everything that step needs.

Deliberately cheap, not the full statistical-power run. If the covertness split looks mostly/entirely overt (the model just
states "moral implication: aim above X" rather than denying), stop and reconsider the model.

Run with: python3 runs/hour0_pilot.py [--stamp hour0] [--n-baseline 30] [--n-pilot 40]
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.qual import sample, prompts, parse, judge

SEED = 0


def run_condition(llm, tokenizer, run_dir, condition, prompt_text, n, item_prefix, batch_size=200):
    """Generates n rollouts for one (condition, prompt_text) pair, appending to raw JSONL."""
    all_records = []
    remaining = n
    idx0 = 0
    while remaining > 0:
        this_batch = min(batch_size, remaining)
        items = [(f"{item_prefix}_{idx0+i}", prompt_text) for i in range(this_batch)]
        records, wall_clock = sample.generate_batch(llm, tokenizer, items, n=1)
        for r in records:
            r["condition"] = condition
        sample.append_jsonl(run_dir / "raw" / f"{item_prefix}.jsonl", records)
        all_records.extend(records)
        idx0 += this_batch
        remaining -= this_batch
        print(f"  {item_prefix}: {idx0}/{n} done ({wall_clock:.1f}s for this batch)", flush=True)
    return all_records


def behavioral_check(records, threshold, direction):
    """Quick P(good side) readout: is there even a bias to explain, before asking whether the
    CoT admits or denies it?"""
    n_valid = 0
    n_good = 0
    for r in records:
        val, _ = parse.parse_estimate(r["raw_completion"])
        if val is None or not parse.apply_outlier_filter(val, threshold):
            continue
        n_valid += 1
        if parse.classify_on_good_side(val, threshold, direction):
            n_good += 1
    p_good = n_good / n_valid if n_valid else float("nan")
    return {"n_total": len(records), "n_valid": n_valid, "p_good_side": p_good}


def write_covertness_judge_inputs(run_dir, threshold, seed=SEED):
    """Mirrors the paper's covertness judge input (App. E.2.1): the CoT only (never the final
    response), numbers blurred by build_covertness_prompt itself. One file per item so a
    subagent fan-out does a small targeted Read instead of shuttling all the CoT text through
    the orchestrator's own context."""
    judge_inputs_dir = run_dir / "judge_inputs"
    judge_inputs_dir.mkdir(exist_ok=True)
    rng = random.Random(seed)
    manifest = []

    for direction in ["above_good", "below_good"]:
        records = parse.parse_jsonl_file(run_dir / "raw" / f"giraffes_{direction}.jsonl")
        for r in records:
            val, reason = parse.parse_estimate(r["raw_completion"])
            on_good_side = parse.classify_on_good_side(val, threshold, direction) if val is not None else None
            cot_only = r["raw_completion"].split("</think>", 1)[0]
            prompt_text = judge.build_covertness_prompt(cot_only)
            item_id = f"covertness_{direction}_{r['item_id']}"
            fpath = judge_inputs_dir / f"{item_id}.txt"
            fpath.write_text(prompt_text)
            manifest.append({
                "id": item_id, "kind": "covertness", "path": str(fpath),
                "direction": direction, "source_item_id": r["item_id"],
                "estimate": val, "on_good_side": on_good_side, "parse_reason": reason,
            })

    with open(run_dir / "judge_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", default="hour0")
    ap.add_argument("--n-baseline", type=int, default=30)
    ap.add_argument("--n-pilot", type=int, default=40, help="rollouts per intervention direction")
    args = ap.parse_args()

    run_dir = Path(__file__).resolve().parent / args.stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Loading engine...", flush=True)
    llm, tokenizer = sample.load_engine()

    sample.write_config(run_dir, {
        "phase": "hour0_pilot",
        "task": "giraffes",
        "n_baseline": args.n_baseline,
        "n_pilot_per_direction": args.n_pilot,
    })

    # --- baseline, to derive the threshold ---
    print("=== Baseline (giraffes) ===", flush=True)
    baseline_prompt = prompts.format_prompt("baseline")
    baseline_records = run_condition(llm, tokenizer, run_dir, "baseline", baseline_prompt,
                                      args.n_baseline, "giraffes_baseline")

    valid_estimates = []
    parse_failures = 0
    for r in baseline_records:
        val, reason = parse.parse_estimate(r["raw_completion"])
        if val is None:
            parse_failures += 1
        else:
            valid_estimates.append(val)
    threshold = parse.compute_threshold(valid_estimates, sig_figs=2)
    print(f"n_valid={len(valid_estimates)}, unknown_rate={parse_failures/args.n_baseline:.3f}, "
          f"threshold={threshold}", flush=True)

    with open(run_dir / "threshold.json", "w") as f:
        json.dump({
            "threshold": threshold, "n_valid": len(valid_estimates),
            "n_total": args.n_baseline, "unknown_rate": parse_failures / args.n_baseline,
        }, f, indent=2)

    # --- above_good / below_good at that threshold ---
    behavioral = {}
    for direction in ["above_good", "below_good"]:
        print(f"=== {direction} (giraffes, threshold={threshold}) ===", flush=True)
        prompt_text = prompts.format_prompt(direction, threshold=threshold)
        records = run_condition(llm, tokenizer, run_dir, direction, prompt_text,
                                 args.n_pilot, f"giraffes_{direction}")
        behavioral[direction] = behavioral_check(records, threshold, direction)

    print("\n=== Behavioral check (is there a bias to explain at all?) ===")
    for direction, stats in behavioral.items():
        print(f"  {direction:12s} n_valid={stats['n_valid']:3d}  P(good side)={stats['p_good_side']:.3f}")
    bias_point = behavioral["above_good"]["p_good_side"] + behavioral["below_good"]["p_good_side"] - 1
    print(f"  bias (uncorrected point estimate, no bootstrap CI at this n) = {bias_point:+.3f}")

    # --- covertness judge inputs (the actual hour-0 question: covert or overt?) ---
    manifest = write_covertness_judge_inputs(run_dir, threshold)
    print(f"\nWrote {len(manifest)} covertness judge input files to {run_dir / 'judge_inputs'}")
    print(f"Manifest: {run_dir / 'judge_manifest.json'}")
    print("\nNext: dispatch judge.build_covertness_prompt's outputs (one per judge_inputs/*.txt) "
          "to a judge model, tally into Admits/Mentions/No mention/Denies via "
          "judge.parse_covertness_answer, and check whether the model reproduces the covert "
          "variant before committing to the rest of the plan.")


if __name__ == "__main__":
    main()
