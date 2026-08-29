"""H1's direct test: does removing the admission sentence remove the bias?

For each overt rollout, truncate the CoT just BEFORE the located admission sentence and let the
model regenerate from there many times. If the admission is doing causal work, continuations
from a prefix that stops short of it should land on the favoured side less often.

The control matters more than the treatment here. Truncating anywhere discards the rest of the
trace, so "P(good) changed after truncation" is not evidence about the admission specifically.
Two controls:

  random    truncate at a random OTHER sentence in the same rollout. Isolates "this sentence"
            from "truncation in general".
  post      truncate just AFTER the admission, i.e. keep it. If pre < post, the sentence itself
            is what mattered; if pre == post, the position is.

Truncation position still differs systematically between pre-admission and a uniformly random
index, so the analysis regresses P(good) on condition AND normalised position rather than
relying on the design to match them.

Cost note: continuations carry the full remaining CoT, so this is the most expensive block in
the project. Defaults are sized for ~1.5h on one A100.

Run with: source env.sh && python3 runs/resample_admission.py [--n-cont 10]
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.qual import parse, prompts, resample, sample, segment, stats


def load_locations(run_dir):
    out = {}
    for p in (run_dir / "locator_outputs").glob("*.txt"):
        try:
            d = json.loads(p.read_text().strip())
            out[p.stem] = d
        except Exception as e:  # noqa: BLE001 - a malformed locator answer is data, not a crash
            print(f"  ! unparseable locator output for {p.stem}: {e}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="hour0")
    ap.add_argument("--stamp", default="resample_admission")
    ap.add_argument("--n-cont", type=int, default=10, help="continuations per (rollout, cut)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-model-len", type=int, default=20480,
                    help="prefixes run to 11.2k tokens here, so the default 16384 would leave "
                         "too little room for the continuation and silently truncate")
    ap.add_argument("--max-tokens", type=int, default=8000)
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    src = here / args.run
    run_dir = here / args.stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    T = json.load(open(src / "threshold.json"))["threshold"]

    locs = load_locations(src)
    recs = {}
    for d in ["above_good", "below_good"]:
        for r in parse.parse_jsonl_file(src / "raw" / f"giraffes_{d}.jsonl"):
            r["direction"] = d
            recs[r["item_id"]] = r
    print(f"locator answers: {len(locs)}; threshold {T:,.0f}")

    rng = random.Random(args.seed)
    items, meta = [], {}
    for iid, loc in sorted(locs.items()):
        r = recs[iid]
        sents = segment.segment_sentences(r["raw_completion"])
        k = int(loc["sentence_index"])
        if not (1 <= k < len(sents)):
            print(f"  ! {iid}: index {k} out of range for {len(sents)} sentences, skipping")
            continue
        others = [j for j in range(1, len(sents)) if abs(j - k) > 1]
        if not others:
            continue
        cuts = {"pre": k, "post": k + 1, "random": rng.choice(others)}
        for cut_name, idx in cuts.items():
            if idx > len(sents):
                continue
            key = f"{iid}::{cut_name}"
            items.append({"item_id": key,
                          "user_content": prompts.format_prompt(r["direction"], threshold=T),
                          "forced_prefix": resample.build_forced_prefix(r["raw_completion"], idx)})
            meta[key] = {"source": iid, "cut": cut_name, "idx": idx,
                         "n_sentences": len(sents), "direction": r["direction"],
                         "norm_pos": idx / max(1, len(sents) - 1),
                         "confidence": loc.get("confidence")}
    print(f"{len(items)} (rollout, cut) cells x {args.n_cont} continuations "
          f"= {len(items)*args.n_cont} generations\n")

    llm, tokenizer = sample.load_engine(max_model_len=args.max_model_len)
    sample.write_config(run_dir, {"phase": "resample_admission", "n_cont": args.n_cont,
                                  "threshold": T, "n_cells": len(items)})

    CHUNK = 24
    all_rows = []
    for s0 in range(0, len(items), CHUNK):
        batch = items[s0:s0 + CHUNK]
        got = resample.generate_forced_continuations_per_item(
            llm, tokenizer, batch, n_continuations=args.n_cont, max_tokens=args.max_tokens)
        for key, rows in got.items():
            m = meta[key]
            for r in rows:
                r.update(m)
                v, reason = parse.parse_estimate(r["raw_completion"])
                r["estimate"] = v
                r["on_good_side"] = (parse.classify_on_good_side(v, T, m["direction"])
                                     if v is not None and parse.apply_outlier_filter(v, T)
                                     else None)
            sample.append_jsonl(run_dir / "raw" / "continuations.jsonl", rows)
            all_rows += rows
        print(f"  {min(s0+CHUNK, len(items))}/{len(items)} cells done", flush=True)

    n_trunc = sum(1 for r in all_rows if r["finish_reason"] == "length")
    print(f"\ntruncated continuations (hit max_tokens): {n_trunc}/{len(all_rows)} "
          f"({n_trunc/len(all_rows):.1%})")

    print("\n=== P(favoured side) by where the CoT was cut ===")
    res = {"threshold": T, "n_cont": args.n_cont, "n_truncated": n_trunc,
           "n_continuations": len(all_rows), "by_cut": {}}
    for cut in ["pre", "post", "random"]:
        sub = [r for r in all_rows if r["cut"] == cut and r["on_good_side"] is not None]
        if not sub:
            continue
        p = sum(r["on_good_side"] for r in sub) / len(sub)
        pos = sum(r["norm_pos"] for r in sub) / len(sub)
        res["by_cut"][cut] = {"n": len(sub), "p_favored": p, "mean_norm_pos": pos}
        print(f"  cut={cut:7s} n={len(sub):4d}  P(favoured)={p:.3f}  "
              f"mean cut position={pos:.2f} of the CoT")

    o = {}
    for cut in ["pre", "post", "random"]:
        o[cut] = {}
        for d in ["above_good", "below_good"]:
            o[cut][d] = [1 if r["on_good_side"] else 0 for r in all_rows
                         if r["cut"] == cut and r["direction"] == d
                         and r["on_good_side"] is not None]
    print("\n=== bias by cut ===")
    for cut in ["pre", "post", "random"]:
        if o[cut]["above_good"] and o[cut]["below_good"]:
            pt, lo, hi = stats.balanced_bias_bootstrap(o[cut]["above_good"], o[cut]["below_good"])
            res["by_cut"].setdefault(cut, {}).update(
                {"bias": pt, "ci_low": lo, "ci_high": hi})
            print(f"  cut={cut:7s} bias={pt:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    print("  unprefilled reference: +0.420 [+0.220, +0.622]")

    json.dump(res, open(run_dir / "results.json", "w"), indent=2)
    print(f"\nwrote {run_dir/'results.json'}")


if __name__ == "__main__":
    main()
