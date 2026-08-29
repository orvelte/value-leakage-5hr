"""A3 + A4 — between-arm contrasts for the extra prefill arms.

The driver (a34_prefill_extra.py) reports each arm's bias against a remembered reference. The
claims here are all DIFFERENCES between arms, so they need their own CIs: eyeballing whether two
percentile intervals overlap is a stricter and differently-shaped test than testing the contrast,
and for admit_twice vs admit_then_deny the two intervals graze each other while the difference is
nowhere near zero.

Bootstrap matches stats.balanced_bias_bootstrap exactly -- independent per-direction binomial
resampling -- so a contrast is the difference of two draws from that same scheme.

The newline arm is INVALID as an A3 control and is reported but never used in a contrast: its
prefix "<think>\\n" + "\\n" is the opening of Qwen's EMPTY think block, so the model closes it
immediately and answers with no CoT (median 11 tokens). It reproduces the thinking-off condition
rather than a contentless prefill. See sample.py's note that thinking-off IS this prefill.

Run with: python3 runs/a34_analyze.py   (no GPU)
"""
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.qual import parse, stats

HERE = Path(__file__).resolve().parent
# arm -> run dir holding its raw/ files
ARMS = {
    "neutral": "prefill_tests", "denial": "prefill_tests", "admission": "prefill_tests",
    "minimal": "prefill_extra", "newline": "prefill_extra",
    "admit_then_deny": "prefill_extra", "deny_then_admit": "prefill_extra",
    "admit_twice": "prefill_extra",
}
INVALID = {"newline"}

CONTRASTS = [
    # (a, b, what it settles)
    ("admit_twice", "admit_then_deny",
     "A4 decisive: same length, same admission, only the retraction differs"),
    ("admission", "admit_then_deny", "A4 vs the unpadded admission reference"),
    ("deny_then_admit", "admit_then_deny", "order / recency: same two sentences, swapped"),
    ("admit_twice", "deny_then_admit", "does an admission last differ from an admission first"),
    ("minimal", "neutral", "A3: does a contentless prefill differ from the method sentence"),
    ("minimal", "admission", "A3: contentless vs the stated goal"),
]


def load_arm(arm):
    """-> (outcomes_above, outcomes_below, token list, n_raw, n_trunc). Same parse path as the
    drivers: parse_estimate -> outlier filter -> classify_on_good_side."""
    run_dir = HERE / ARMS[arm]
    T = json.load(open(run_dir / "results.json"))["threshold"]
    out, toks, n_raw, n_trunc = {}, [], 0, 0
    for direction in ("above_good", "below_good"):
        recs = [json.loads(l) for l in open(run_dir / "raw" / f"{arm}_{direction}.jsonl")]
        n_raw += len(recs)
        vals = []
        for r in recs:
            toks.append(r["num_tokens"])
            v, reason = parse.parse_estimate(r["raw_completion"])
            if reason == "truncated_no_close_tag":
                n_trunc += 1
            elif v is not None and parse.apply_outlier_filter(v, T):
                vals.append(v)
        out[direction] = [1 if parse.classify_on_good_side(v, T, direction) else 0 for v in vals]
    return out["above_good"], out["below_good"], toks, n_raw, n_trunc


def bias_draws(above, below, n_resamples=20000, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.binomial(len(above), np.mean(above), size=n_resamples) / len(above)
    b = rng.binomial(len(below), np.mean(below), size=n_resamples) / len(below)
    return b + a - 1.0


def main():
    arms = {a: load_arm(a) for a in ARMS}
    res = {"arms": {}, "contrasts": {}}

    print(f"{'arm':18s}{'bias':>9s}{'95% CI':>20s}{'med tok':>9s}{'n_valid':>9s}{'drop':>7s}")
    for arm, (ab, be, toks, n_raw, n_trunc) in arms.items():
        pt, lo, hi = stats.balanced_bias_bootstrap(ab, be)
        n_valid = len(ab) + len(be)
        row = {"bias": pt, "ci": [lo, hi], "median_tokens": st.median(toks),
               "n_raw": n_raw, "n_valid": n_valid, "n_truncated": n_trunc,
               "drop_rate": 1 - n_valid / n_raw,
               "p_favored_above": float(np.mean(ab)), "p_favored_below": float(np.mean(be))}
        if arm in INVALID:
            row["invalid"] = ("prefix is the empty-think opening; model emits no CoT. "
                              "Reproduces thinking-off, not a contentless prefill.")
        res["arms"][arm] = row
        flag = "  <-- INVALID as an A3 control" if arm in INVALID else ""
        print(f"{arm:18s}{pt:+9.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>20s}"
              f"{st.median(toks):9.0f}{n_valid:9d}{1 - n_valid/n_raw:7.2f}{flag}")

    print(f"\n{'contrast':40s}{'diff':>9s}{'95% CI':>20s}{'boot p':>9s}")
    for a, b, why in CONTRASTS:
        # independent arms -> independent draws; different seeds so the two are not coupled
        da = bias_draws(arms[a][0], arms[a][1], seed=1)
        db = bias_draws(arms[b][0], arms[b][1], seed=2)
        d = da - db
        pt = (np.mean(arms[a][1]) + np.mean(arms[a][0]) - 1) - \
             (np.mean(arms[b][1]) + np.mean(arms[b][0]) - 1)
        lo, hi = np.quantile(d, (0.025, 0.975))
        # two-sided bootstrap p: twice the mass on the far side of zero
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        res["contrasts"][f"{a} - {b}"] = {"diff": float(pt), "ci": [float(lo), float(hi)],
                                          "boot_p": float(p), "why": why}
        print(f"{a+' - '+b:40s}{pt:+9.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>20s}{p:9.4f}")
        print(f"{'':40s}{why}")

    out = HERE / "prefill_extra" / "a34_contrasts.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
