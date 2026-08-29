"""A5 — does the no-CoT effect generalise to a second estimation question?

Every CoT analysis in this project is on the giraffes question, and "one estimation question" is
the standing limitation at the top of the write-up. This does not fix that — the trajectory work
stays single-question — but it tests whether the THINKING-OFF result, which is the load-bearing
claim (the bias needs no CoT), reproduces on a second question from the paper's own Table 2.

Two phases, one engine session:
  1. thinking ON, no bet, n=30 -> the question's own baseline median -> threshold T, rounded to
     2 s.f. exactly as parse.compute_threshold does for giraffes. The threshold MUST come from
     this question's baseline; reusing giraffes' 75M would make the anchor arbitrary and the
     bias metric uninterpretable.
  2. thinking OFF, n=150 per arm: baseline / above_good / below_good / threshold_only / coin.

Readouts, matching the giraffes run so the two are comparable:
  - bias + bootstrap CI from above_good/below_good (the symmetric design makes it anchor-proof)
  - median estimate per arm, and the fraction within +-10% of T (the A6 magnitude signature)
  - KS of each control arm against the no-bet baseline (the anchoring test)

Run with: source env.sh && python3 runs/a5_second_question.py [--n 150] [--n-threshold 30]
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.qual import parse, prompts, sample, stats

MAX_TOKENS_NO_THINK = 4096
NOTHINK_ARMS = ["baseline", "above_good", "below_good", "threshold_only", "coin"]


def parse_vals(recs, T=None):
    """-> (values, n_unparsed). No outlier filter: it is calibrated to a threshold this
    question's no-think baseline need not sit near, exactly as in the giraffes run."""
    vals, bad = [], 0
    for r in recs:
        v, _ = parse.parse_estimate(r["raw_completion"])
        if v is None or v <= 0:
            bad += 1
        else:
            vals.append(v)
    return vals, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", default="second_question")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--n-threshold", type=int, default=30)
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    run_dir = here / args.stamp
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    Q = prompts.TURNS_QUESTION

    llm, tokenizer = sample.load_engine()

    # ---- phase 1: thinking-ON baseline, to derive this question's own threshold ----
    print(f"=== phase 1: thinking-ON baseline, n={args.n_threshold} (threshold derivation) ===",
          flush=True)
    # one item per rollout with n=1, matching runs/thinking_off.py exactly. Generating N
    # samples from a single request is a different call shape, and the giraffes data this is
    # being compared against was produced the first way.
    ptext = prompts.format_prompt("baseline", question=Q)
    items = [(f"turns_thinkon_baseline_{i}", ptext) for i in range(args.n_threshold)]
    recs, _ = sample.generate_batch(llm, tokenizer, items, n=1, enable_thinking=True)
    for r in recs:
        r["condition"] = "thinkon_baseline"
    sample.append_jsonl(run_dir / "raw" / "thinkon_baseline.jsonl", recs)
    vals, bad = parse_vals(recs)
    T = parse.compute_threshold(vals)
    print(f"  n_valid={len(vals)}/{len(recs)} unparsed={bad}  median={st.median(vals):,.0f}  "
          f"-> threshold T = {T:,.0f}", flush=True)

    (run_dir / "threshold.json").write_text(json.dumps(
        {"threshold": T, "question": "turns", "n": len(vals),
         "baseline_median": st.median(vals), "sig_figs": 2}, indent=2))
    sample.write_config(run_dir, {"phase": "A5_second_question", "task": "turns",
                                  "n_per_arm": args.n, "n_threshold": args.n_threshold,
                                  "threshold": T, "max_tokens_no_think": MAX_TOKENS_NO_THINK})

    # ---- phase 2: thinking-OFF arms ----
    results = {"question": "turns", "threshold": T, "n_per_arm": args.n,
               "thinkon_baseline_median": st.median(vals), "arms": {}}
    store = {}
    for arm in NOTHINK_ARMS:
        kw = {} if arm == "baseline" else {"threshold": T}
        ptext = prompts.format_prompt(arm, question=Q, **kw)
        items = [(f"turns_nothink_{arm}_{i}", ptext) for i in range(args.n)]
        print(f"\n=== phase 2: thinking-OFF {arm} (n={args.n}) ===", flush=True)
        recs, _ = sample.generate_batch(llm, tokenizer, items, n=1,
                                        max_tokens=MAX_TOKENS_NO_THINK, enable_thinking=False)
        for r in recs:
            r["condition"] = arm
        sample.append_jsonl(run_dir / "raw" / f"nothink_{arm}.jsonl", recs)
        v, bad = parse_vals(recs)
        store[arm] = v
        ratio = np.array(v) / T
        results["arms"][arm] = {
            "n_raw": len(recs), "n_valid": len(v), "n_unparsed": bad,
            "median": float(np.median(v)), "median_over_T": float(np.median(v) / T),
            "frac_within_10pct": float(np.mean(np.abs(ratio - 1) <= 0.10)),
            "frac_within_25pct": float(np.mean(np.abs(ratio - 1) <= 0.25)),
            "p_above_T": float(np.mean(np.array(v) > T))}
        a = results["arms"][arm]
        print(f"  n_valid={len(v)}/{len(recs)} median={a['median']:,.0f} "
              f"({a['median_over_T']:.2f}xT)  within10%={a['frac_within_10pct']:.3f}  "
              f"P(>T)={a['p_above_T']:.3f}", flush=True)

    # ---- bias ----
    ab = [1 if parse.classify_on_good_side(v, T, "above_good") else 0 for v in store["above_good"]]
    be = [1 if parse.classify_on_good_side(v, T, "below_good") else 0 for v in store["below_good"]]
    pt, lo, hi = stats.balanced_bias_bootstrap(ab, be)
    results["bias"] = {"point": pt, "ci_low": lo, "ci_high": hi,
                       "p_favored_above": float(np.mean(ab)), "p_favored_below": float(np.mean(be))}

    # ---- anchoring: KS of each arm against the no-bet baseline ----
    results["ks_vs_baseline"] = {}
    for arm in NOTHINK_ARMS:
        if arm == "baseline":
            continue
        d, p = stats.two_sample_ks(np.log10(store[arm]), np.log10(store["baseline"]))
        results["ks_vs_baseline"][arm] = {"D": float(d), "p": float(p)}

    print("\n" + "=" * 72)
    print(f"QUESTION: turns (Lisbon -> Singapore left turns)")
    print(f"threshold T = {T:,.0f}  (thinking-on baseline median "
          f"{results['thinkon_baseline_median']:,.0f})")
    print(f"\nbias = {pt:+.3f}  [{lo:+.3f}, {hi:+.3f}]   "
          f"(above {np.mean(ab):.3f} / below {np.mean(be):.3f})")
    print(f"  giraffes reference, thinking-off: +0.517 [+0.415, +0.612]")
    print(f"\n{'arm':16s}{'median':>14s}{'/T':>8s}{'w/in 10%':>10s}{'w/in 25%':>10s}"
          f"{'P(>T)':>8s}{'KS D':>8s}{'KS p':>10s}")
    for arm in NOTHINK_ARMS:
        a = results["arms"][arm]
        k = results["ks_vs_baseline"].get(arm)
        ksd = f"{k['D']:.3f}" if k else "—"
        ksp = f"{k['p']:.2e}" if k else "—"
        print(f"{arm:16s}{a['median']:14,.0f}{a['median_over_T']:8.2f}"
              f"{a['frac_within_10pct']:10.3f}{a['frac_within_25pct']:10.3f}{a['p_above_T']:8.3f}"
              f"{ksd:>8s}{ksp:>10s}")
    print("\n  giraffes reference: no-think baseline 422M vs T=75M (5.6xT); "
          "threshold_only 83M; coin 75,000,001")

    json.dump(results, open(run_dir / "results.json", "w"), indent=2)
    print(f"\nwrote {run_dir/'results.json'}")


if __name__ == "__main__":
    main()
