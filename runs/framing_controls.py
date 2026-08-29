"""H3 framing sweep: is the Donation Bet effect values, a numeric anchor, or user-desire?

Run after the thinking-off gate sent us toward H3. Four control framings against the existing
good/bad-cause condition, all sharing the paper's template except the consequence clause:

  threshold_only  names the threshold, no consequence at all
  coin            keeps the conditional structure, outcome decided by a coin either way
  userpick_above  above the threshold the USER picks the charity, below the friend picks
  userpick_below  the mirror image

Two readouts, because the controls answer different questions:

  ANCHORING (threshold_only, coin): no favored side exists, so a bias metric is undefined.
  Compare the estimate distribution against the no-bet baseline instead. This only has power
  in the thinking-OFF arm, where the baseline median (~422M) is far from the threshold (75M);
  in the thinking-on arm the threshold was defined as the baseline median, so an anchor would
  be invisible by construction. That is why this runs thinking-off by default.

  USER-DESIRE (userpick_above/below): a favored side exists, so these give a bias metric
  directly comparable to the +0.517 measured for the good/bad-cause framing.

Run with: source env.sh && python3 runs/framing_controls.py [--n 150]
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.qual import parse, prompts, sample, stats

MAX_TOKENS_NO_THINK = 4096
ANCHOR_CONDS = ["threshold_only", "coin"]
BIAS_CONDS = ["userpick_above", "userpick_below"]


def run_condition(llm, tokenizer, run_dir, cond, text, n, enable_thinking, batch_size=200):
    recs_all, idx0, remaining = [], 0, n
    while remaining > 0:
        this = min(batch_size, remaining)
        items = [(f"{cond}_{idx0+i}", text) for i in range(this)]
        recs, wall = sample.generate_batch(
            llm, tokenizer, items, n=1,
            max_tokens=MAX_TOKENS_NO_THINK if not enable_thinking else sample.MAX_TOKENS,
            enable_thinking=enable_thinking)
        for r in recs:
            r["condition"] = cond
        sample.append_jsonl(run_dir / "raw" / f"{cond}.jsonl", recs)
        recs_all += recs
        idx0 += this
        remaining -= this
        print(f"  {cond}: {idx0}/{n} ({wall:.0f}s)", flush=True)
    return recs_all


def values(recs, threshold, apply_filter=True):
    out = []
    for r in recs:
        v, _ = parse.parse_estimate(r["raw_completion"])
        if v is None or (apply_filter and not parse.apply_outlier_filter(v, threshold)):
            continue
        out.append(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", default="framing_controls")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--threshold-from", default="hour0")
    ap.add_argument("--thinking", action="store_true",
                    help="run with CoT enabled (default off: the anchor test has no power on")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    run_dir = here / args.stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    T = json.load(open(here / args.threshold_from / "threshold.json"))["threshold"]
    enable_thinking = args.thinking
    print(f"threshold = {T:,.0f} (reused, not re-derived) | thinking={'ON' if enable_thinking else 'OFF'}\n")

    llm, tokenizer = sample.load_engine()
    sample.write_config(run_dir, {"phase": "framing_controls", "task": "giraffes",
                                  "enable_thinking": enable_thinking, "n_per_condition": args.n,
                                  "threshold": T, "threshold_source": args.threshold_from})

    recs = {}
    for cond in ANCHOR_CONDS + BIAS_CONDS:
        print(f"=== {cond} ===", flush=True)
        recs[cond] = run_condition(llm, tokenizer, run_dir, cond,
                                   prompts.format_prompt(cond, threshold=T), args.n,
                                   enable_thinking)

    # reference arms from the thinking-off run, for the anchoring comparison
    ref_dir = here / ("thinking_off" if not enable_thinking else args.threshold_from)
    ref = {}
    for name, fn in [("baseline", "nothink_baseline"), ("above_good", "nothink_above_good"),
                     ("below_good", "nothink_below_good")]:
        f = ref_dir / "raw" / f"{fn}.jsonl"
        if f.exists():
            ref[name] = parse.parse_jsonl_file(f)

    results = {"threshold": T, "enable_thinking": enable_thinking, "n": args.n, "conditions": {}}

    print("\n=== ANCHORING: estimate distribution vs the no-bet baseline ===")
    print(f"  {'condition':16s}{'n':>5s}{'median':>16s}{'ratio to T':>12s}{'KS vs baseline':>18s}")
    base_vals = values(ref["baseline"], T, apply_filter=False) if "baseline" in ref else []
    if base_vals:
        print(f"  {'baseline (no bet)':16s}{len(base_vals):5d}{st.median(base_vals):16,.0f}"
              f"{st.median(base_vals)/T:12.2f}{'-':>18s}")
    for cond in ANCHOR_CONDS + BIAS_CONDS + list(ref):
        v = values(recs[cond] if cond in recs else ref[cond], T, apply_filter=False)
        if not v or cond == "baseline":
            continue
        ks = f"D={stats.two_sample_ks(v, base_vals)[0]:.3f} p={stats.two_sample_ks(v, base_vals)[1]:.1e}" if base_vals else "-"
        print(f"  {cond:16s}{len(v):5d}{st.median(v):16,.0f}{st.median(v)/T:12.2f}{ks:>18s}")
        results["conditions"][cond] = {"n": len(v), "median": st.median(v),
                                       "median_over_threshold": st.median(v) / T}

    print("\n=== USER-DESIRE: bias metric, comparable to the good/bad-cause framing ===")
    outcomes = {}
    for cond in BIAS_CONDS:
        d = prompts.FAVORED_DIRECTION[cond]
        v = values(recs[cond], T)
        outcomes[d] = [1 if parse.classify_on_good_side(x, T, d) else 0 for x in v]
        print(f"  {cond:16s} n={len(v):3d}  P(favored)={sum(outcomes[d])/len(outcomes[d]):.3f}")
    pt, lo, hi = stats.balanced_bias_bootstrap(outcomes["above_good"], outcomes["below_good"])
    print(f"  user-picks-charity bias = {pt:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"  good/bad-cause reference = +0.517 [+0.415, +0.612] (thinking off, n=148/146)")
    results["userpick_bias"] = {"point": pt, "ci_low": lo, "ci_high": hi}

    json.dump(results, open(run_dir / "results.json", "w"), indent=2)
    print(f"\nwrote {run_dir/'results.json'}")


if __name__ == "__main__":
    main()
