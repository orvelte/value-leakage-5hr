"""Donation Bet across Qwen releases: generate every cell for ONE model (see runs/releases/PREREG.md).

Two phases, because the thresholds have to exist before the bet prompts can be written:

  phase 1  thinking ON   giraffes baseline (n=40) + turns baseline (n=30)  -> T_giraffes, T_turns
           thinking OFF  giraffes baseline (n=150)                         -> distribution, anchoring
  phase 2  thinking ON   giraffes above_good / below_good (n=40 each)
           thinking OFF  giraffes above_good / below_good (150 each), userpick_above / _below
                         (100 each), turns above_good / below_good (100 each)

Both thresholds come from the model's own THINKING-ON baseline median and are reused for the
thinking-off cells, exactly as the original run did (runs/thinking_off.py reused hour0's 75M;
runs/a5_second_question.py derived turns' 1600 from a thinking-on n=30 baseline). DECISIONS.md #1.

File layout under runs/releases/<model>/ mirrors runs/hour0 + runs/thinking_off so the original
analysis scripts (analyze_revision.py, prep_covertness.py, ...) run unchanged with
--run releases/<model>. Raw JSONL is append-only; a cell that already has >= n records is
skipped, so a crashed run resumes where it stopped.

Run with: python3 runs/releases/generate.py --model q35 [--phase 1|2|all]
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))

from src.qual import parse, prompts
from lib import sample

MAX_TOKENS_THINK = 16000     # original config (sample.MAX_TOKENS)
MAX_TOKENS_NO_THINK = 4096   # original thinking_off.py

# (file stem, condition, question, enable_thinking, n)
PHASE1 = [
    ("giraffes_baseline", "baseline", "giraffes", True, 40),
    ("turns_thinkon_baseline", "baseline", "turns", True, 30),
    ("nothink_baseline", "baseline", "giraffes", False, 150),
]
PHASE2 = [
    ("giraffes_above_good", "above_good", "giraffes", True, 40),
    ("giraffes_below_good", "below_good", "giraffes", True, 40),
    ("nothink_above_good", "above_good", "giraffes", False, 150),
    ("nothink_below_good", "below_good", "giraffes", False, 150),
    ("nothink_userpick_above", "userpick_above", "giraffes", False, 100),
    ("nothink_userpick_below", "userpick_below", "giraffes", False, 100),
    ("nothink_turns_above_good", "above_good", "turns", False, 100),
    ("nothink_turns_below_good", "below_good", "turns", False, 100),
]


def n_done(path):
    return len(parse.parse_jsonl_file(path)) if path.exists() else 0


def run_cells(llm, tok, run_dir, cells, thresholds):
    """All cells sharing a thinking mode go into ONE generate call so vLLM batches them."""
    for think in (True, False):
        items, owners = [], []
        for stem, cond, q, th, n in cells:
            if th != think:
                continue
            path = run_dir / "raw" / f"{stem}.jsonl"
            have = n_done(path)
            if have >= n:
                print(f"  {stem}: {have}/{n} already, skipping")
                continue
            T = None if cond == "baseline" else thresholds[q]
            text = prompts.format_prompt(cond, question=prompts.QUESTIONS[q], threshold=T)
            for i in range(have, n):
                items.append((f"{stem}_{i}", text))
                owners.append((stem, cond, q))
        if not items:
            continue
        print(f"=== thinking {'ON' if think else 'OFF'}: {len(items)} rollouts ===", flush=True)
        recs, wall = sample.generate_batch(
            llm, tok, items, n=1, enable_thinking=think,
            max_tokens=MAX_TOKENS_THINK if think else MAX_TOKENS_NO_THINK)
        by_stem = {}
        for r, (stem, cond, q) in zip(recs, owners):
            r.update({"condition": cond, "question": q, "model": sample.MODEL_ID})
            by_stem.setdefault(stem, []).append(r)
        for stem, rs in by_stem.items():
            sample.append_jsonl(run_dir / "raw" / f"{stem}.jsonl", rs)
            print(f"  {stem}: +{len(rs)}  median tokens "
                  f"{sorted(x['num_tokens'] for x in rs)[len(rs)//2]}", flush=True)
        print(f"  wall {wall/60:.1f} min", flush=True)


def derive_threshold(path, question):
    vals = [v for v, _ in (parse.parse_estimate(r["raw_completion"])
                           for r in parse.parse_jsonl_file(path)) if v is not None]
    T = parse.compute_threshold(vals)
    n_total = n_done(path)
    return {"threshold": T, "question": question, "n_valid": len(vals), "n_total": n_total,
            "unknown_rate": 1 - len(vals) / n_total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(sample.MODELS))
    ap.add_argument("--phase", default="all", choices=["1", "2", "all"])
    args = ap.parse_args()

    sample.set_model(args.model)
    run_dir = HERE / args.model
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    llm, tok = sample.load_engine()
    print(f"engine up in {(time.time()-t0)/60:.1f} min: {sample.MODEL_ID}@{sample.MODEL_REVISION[:8]} "
          f"effort={sample.REASONING_EFFORT}", flush=True)

    # Eyeball check: print both prompt renderings once so the effort/system-message handling is
    # in the log for every model.
    for th in (True, False):
        print(f"--- rendered prompt, thinking {'ON' if th else 'OFF'} ---\n"
              f"{sample.build_prompt_text(tok, prompts.format_prompt('baseline'), th)!r}\n")

    sample.write_config(run_dir, {"phase": "releases", "model_key": args.model,
                                  "max_tokens_think": MAX_TOKENS_THINK,
                                  "max_tokens_no_think": MAX_TOKENS_NO_THINK,
                                  "cells": [c[:5] for c in PHASE1 + PHASE2]})

    if args.phase in ("1", "all"):
        run_cells(llm, tok, run_dir, PHASE1, {})
        tg = derive_threshold(run_dir / "raw" / "giraffes_baseline.jsonl", "giraffes")
        tt = derive_threshold(run_dir / "raw" / "turns_thinkon_baseline.jsonl", "turns")
        (run_dir / "threshold.json").write_text(json.dumps(tg, indent=2))
        (run_dir / "turns_threshold.json").write_text(json.dumps(tt, indent=2))
        print(f"T_giraffes = {tg['threshold']:,.0f} ({tg['n_valid']}/{tg['n_total']})   "
              f"T_turns = {tt['threshold']:,.0f} ({tt['n_valid']}/{tt['n_total']})", flush=True)

    if args.phase in ("2", "all"):
        th = {"giraffes": json.load(open(run_dir / "threshold.json"))["threshold"],
              "turns": json.load(open(run_dir / "turns_threshold.json"))["threshold"]}
        run_cells(llm, tok, run_dir, PHASE2, th)

    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
