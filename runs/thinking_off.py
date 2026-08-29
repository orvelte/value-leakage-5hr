"""Hour 0:30 — the Donation Bet with thinking disabled.

The single most informative cut for "is this unfaithful CoT?": if the bias survives with no
CoT at all, the bias is a prior on the answer and the CoT narrates something that already
happened (H3 track). If it vanishes, the reasoning process is load-bearing (H1/H2 live). If it
drops but survives, the difference is the CoT's contribution.

TWO THINGS THIS SCRIPT IS CAREFUL ABOUT:

1. `enable_thinking=False` is a PREFILL, not a mode. The chat template emits an empty, already
   closed `<think>\\n\\n</think>\\n\\n` and the model answers directly. So this condition is the
   same class of manipulation as the forced-denial prefill and must be reported alongside it,
   with the off-distribution caveat on both. `sample.assistant_prefill` handles reconstruction;
   a hardcoded "<think>\\n" would make every rollout here parse to None.

2. The threshold stays at the THINKING run's 75M so the prompts are byte-identical across arms.

   CORRECTION to an earlier worry: p_favored|baseline is 0.5 by construction here regardless of
   where the distribution sits, because the two directions define favored as P(>T) and P(<=T),
   which sum to 1. The symmetric two-direction design already cancels any baseline shift -- that
   is exactly why the paper runs both directions. The latent-mixture call below is therefore
   equivalent to 2*(p_favored - 0.5); it is kept because it stays correct if the design ever
   becomes asymmetric.

   The baseline arm is still generated, for a different and more interesting reason: it is the
   only way to see the ANCHORING effect. With no CoT the unbiased median is ~422M against a 75M
   threshold, so naming the threshold collapses the estimate ~5x. That is invisible in the
   thinking-on arm, where the threshold was defined as the baseline median.

Run with: source env.sh && python3 runs/thinking_off.py [--n 150]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.qual import parse, prompts, sample, stats

MAX_TOKENS_NO_THINK = 4096  # no CoT, so answers are short; generous headroom vs the 16k arm


def run_condition(llm, tokenizer, run_dir, condition, prompt_text, n, batch_size=200):
    records, idx0, remaining = [], 0, n
    while remaining > 0:
        this = min(batch_size, remaining)
        items = [(f"nothink_{condition}_{idx0+i}", prompt_text) for i in range(this)]
        recs, wall = sample.generate_batch(llm, tokenizer, items, n=1,
                                           max_tokens=MAX_TOKENS_NO_THINK,
                                           enable_thinking=False)
        for r in recs:
            r["condition"] = condition
        sample.append_jsonl(run_dir / "raw" / f"nothink_{condition}.jsonl", recs)
        records += recs
        idx0 += this
        remaining -= this
        print(f"  {condition}: {idx0}/{n} ({wall:.0f}s this batch)", flush=True)
    return records


def summarize(records, threshold, direction):
    vals, n_trunc, n_nonum, n_out = [], 0, 0, 0
    for r in records:
        v, reason = parse.parse_estimate(r["raw_completion"])
        if reason == "truncated_no_close_tag":
            n_trunc += 1
        elif v is None:
            n_nonum += 1
        elif not parse.apply_outlier_filter(v, threshold):
            n_out += 1
        else:
            vals.append(v)
    if direction == "baseline":
        # p_favored | baseline, averaged over what each direction would call favored
        p_fav = 0.5 * ((sum(v > threshold for v in vals) + sum(v <= threshold for v in vals))
                       / len(vals)) if vals else float("nan")
        outcomes = None
    else:
        outcomes = [1 if parse.classify_on_good_side(v, threshold, direction) else 0 for v in vals]
        p_fav = sum(outcomes) / len(outcomes) if outcomes else float("nan")
    return {"n_raw": len(records), "n_valid": len(vals), "n_truncated": n_trunc,
            "n_no_number": n_nonum, "n_outlier_filtered": n_out, "p_favored": p_fav,
            "median": sorted(vals)[len(vals) // 2] if vals else None}, outcomes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", default="thinking_off")
    ap.add_argument("--n", type=int, default=150, help="rollouts per condition")
    ap.add_argument("--threshold-from", default="hour0",
                    help="run whose threshold.json to reuse; do NOT re-derive")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    run_dir = here / args.stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    threshold = json.load(open(here / args.threshold_from / "threshold.json"))["threshold"]
    print(f"threshold = {threshold:,.0f} (reused from {args.threshold_from}, NOT re-derived)\n")

    llm, tokenizer = sample.load_engine()

    # Eyeball check the plan asks for: confirm the turn really has an empty closed think block.
    probe, _ = sample.generate_batch(llm, tokenizer,
                                     [("probe", prompts.format_prompt("baseline"))],
                                     n=1, max_tokens=MAX_TOKENS_NO_THINK, enable_thinking=False)
    raw = probe[0]["raw_completion"]
    cot_body = raw.split("<think>", 1)[1].split("</think>", 1)[0]
    print("=== no-think sanity check ===")
    print(f"  prefill = {sample.assistant_prefill(tokenizer, False)!r}")
    print(f"  think-block body is empty: {cot_body.strip() == ''!r} (len={len(cot_body.strip())})")
    print(f"  first 200 chars after </think>: {raw.split('</think>',1)[1][:200]!r}")
    print(f"  parses to: {parse.parse_estimate(raw)}\n", flush=True)

    sample.write_config(run_dir, {"phase": "thinking_off", "task": "giraffes",
                                  "enable_thinking": False, "n_per_condition": args.n,
                                  "threshold": threshold, "threshold_source": args.threshold_from,
                                  "max_tokens": MAX_TOKENS_NO_THINK})

    summaries, outcomes = {}, {}
    for cond in ["baseline", "above_good", "below_good"]:
        print(f"=== {cond} (thinking OFF) ===", flush=True)
        text = prompts.format_prompt(cond, threshold=None if cond == "baseline" else threshold)
        recs = run_condition(llm, tokenizer, run_dir, cond, text, args.n)
        summaries[cond], outcomes[cond] = summarize(recs, threshold, cond)

    print("\n=== thinking-OFF results ===")
    for cond, s in summaries.items():
        print(f"  {cond:11s} n_valid={s['n_valid']:3d}/{s['n_raw']}  "
              f"trunc={s['n_truncated']} nonum={s['n_no_number']} outlier={s['n_outlier_filtered']}  "
              f"P(favored)={s['p_favored']:.3f}  median={s['median']:,.0f}")

    p_base = summaries["baseline"]["p_favored"]
    p_int = 0.5 * (summaries["above_good"]["p_favored"] + summaries["below_good"]["p_favored"])
    p_biased = stats.latent_mixture_p_biased(p_int, p_base)
    naive = 2 * (p_int - 0.5)
    pt, lo, hi = stats.balanced_bias_bootstrap(outcomes["above_good"], outcomes["below_good"])

    print(f"\n  p_favored|baseline    = {p_base:.3f}   <- 0.5 by construction (symmetric design)")
    print(f"  p_favored|intervention= {p_int:.3f}")
    print(f"  latent-mixture bias   = {p_biased:+.3f}")
    print(f"  naive 2(p-0.5)        = {naive:+.3f}  (identical here, by the symmetry above)")
    print(f"  balanced bootstrap    = {pt:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"\n  thinking-ON reference: bias +0.420, 95% CI [+0.220, +0.622] (n=39/40)")

    json.dump({"threshold": threshold, "n_per_condition": args.n, "summaries": summaries,
               "p_favored_baseline": p_base, "p_favored_intervention": p_int,
               "bias_latent_mixture": p_biased, "bias_naive_2p_minus_1": naive,
               "bias_bootstrap": {"point": pt, "ci_low": lo, "ci_high": hi}},
              open(run_dir / "results.json", "w"), indent=2)
    print(f"\nwrote {run_dir/'results.json'}")


if __name__ == "__main__":
    main()
