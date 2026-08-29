"""A12.3 — fresh-seed replication of the `neutral` prefill arm.

The neutral arm's -0.045 is load-bearing: it is the only arm in the project that sits at zero
with a bet present, and A9 showed that two other method sentences (+0.340, +0.251) and a
contentless prefill (+0.404) do not reproduce it. Either "Let me work through this step by step"
is specifically special, or that cell was a low draw. n=60 per direction gives a +-0.18 CI, which
is wide enough that one draw cannot settle it.

Identical prompt, prefill bytes, sampling settings and parse path as runs/prefill_tests.py — the
only difference is that these are fresh draws (seed=None, as always).

Run with: source env.sh && python3 runs/a12_neutral_replication.py [--n 60]
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.qual import parse, prompts, resample, sample, stats

NEUTRAL = "Let me work through this step by step."


def outcomes_from(run_dir, arm, T):
    out = {}
    for d in ("above_good", "below_good"):
        f = run_dir / "raw" / f"{arm}_{d}.jsonl"
        vals = []
        for rec in parse.parse_jsonl_file(f):
            v, reason = parse.parse_estimate(rec["raw_completion"])
            if v is not None and parse.apply_outlier_filter(v, T):
                vals.append(v)
        out[d] = [1 if parse.classify_on_good_side(v, T, d) else 0 for v in vals]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", default="neutral_replication")
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    run_dir = here / args.stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    T = json.load(open(here / "hour0" / "threshold.json"))["threshold"]

    llm, tokenizer = sample.load_engine()
    sample.write_config(run_dir, {"phase": "A12_neutral_replication", "n_per_cell": args.n,
                                  "threshold": T, "prefill": repr(NEUTRAL)})

    new = {}
    for direction in ("above_good", "below_good"):
        prefix = "<think>\n" + NEUTRAL + " "
        items = [{"item_id": f"neutral_{direction}",
                  "user_content": prompts.format_prompt(direction, threshold=T),
                  "forced_prefix": prefix}]
        print(f"=== neutral / {direction} (n={args.n}, fresh draw) ===", flush=True)
        recs = resample.generate_forced_continuations_per_item(
            llm, tokenizer, items, n_continuations=args.n)[items[0]["item_id"]]
        for r in recs:
            r["prefill"] = "neutral"
            r["condition"] = direction
        sample.append_jsonl(run_dir / "raw" / f"neutral_{direction}.jsonl", recs)
        vals = []
        for r in recs:
            v, reason = parse.parse_estimate(r["raw_completion"])
            if v is not None and parse.apply_outlier_filter(v, T):
                vals.append(v)
        new[direction] = [1 if parse.classify_on_good_side(v, T, direction) else 0 for v in vals]
        print(f"  n_valid={len(vals)}/{len(recs)} P(fav)={sum(new[direction])/len(new[direction]):.3f} "
              f"med_tokens={st.median([r['num_tokens'] for r in recs]):.0f}", flush=True)

    old = outcomes_from(here / "prefill_tests", "neutral", T)
    pooled = {d: old[d] + new[d] for d in ("above_good", "below_good")}

    res = {"threshold": T, "prefill": NEUTRAL, "draws": {}}
    print(f"\n{'draw':14s}{'n/dir':>10s}{'bias':>9s}{'95% CI':>22s}")
    for name, o in (("original", old), ("replication", new), ("pooled", pooled)):
        pt, lo, hi = stats.balanced_bias_bootstrap(o["above_good"], o["below_good"])
        res["draws"][name] = {"n_above": len(o["above_good"]), "n_below": len(o["below_good"]),
                             "bias": pt, "ci": [lo, hi],
                             "p_fav_above": sum(o["above_good"]) / len(o["above_good"]),
                             "p_fav_below": sum(o["below_good"]) / len(o["below_good"])}
        print(f"{name:14s}{len(o['above_good']):5d}/{len(o['below_good']):<4d}{pt:+9.3f}"
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>22s}")

    # do the two draws agree? difference of biases, same resampling scheme as every other contrast
    import numpy as np
    rng1, rng2 = np.random.default_rng(1), np.random.default_rng(2)
    def draws(o, rng):
        a = rng.binomial(len(o["above_good"]), np.mean(o["above_good"]), 20000) / len(o["above_good"])
        b = rng.binomial(len(o["below_good"]), np.mean(o["below_good"]), 20000) / len(o["below_good"])
        return b + a - 1.0
    d = draws(new, rng1) - draws(old, rng2)
    lo, hi = np.quantile(d, (0.025, 0.975))
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    res["replication_minus_original"] = {"diff": float(np.mean(d)), "ci": [float(lo), float(hi)],
                                         "boot_p": float(p)}
    print(f"\nreplication - original: {np.mean(d):+.3f} [{lo:+.3f}, {hi:+.3f}], p={p:.4f}")
    print(f"  references: minimal +0.404 | neutral_b +0.340 | neutral_c +0.251 | none +0.420")

    json.dump(res, open(run_dir / "results.json", "w"), indent=2)
    print(f"\nwrote {run_dir/'results.json'}")


if __name__ == "__main__":
    main()
