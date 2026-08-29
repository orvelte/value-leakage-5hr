"""A9 (zero compute) — why does "step by step" kill the bias?

Two measurements on rollouts that already exist, no GPU, no judge:

1. LEXICAL. How often does the think block mention the incentive at all — the threshold value,
   the word "threshold", "donat*", "cause"? Prediction: the neutral arm ~0, minimal and free CoT
   at the free-CoT rate. If the neutral sentence works by making the model never bring the bet up
   inside the trace, that shows here.

2. ESTIMATE QUALITY. Median and IQR of log10(estimate) per arm, against the thinking-on baseline
   (T=75M, which IS that baseline's median by construction) and the no-think baseline (422M).
   Distinguishes "the bias was removed and the estimate is unharmed" from "the sentence broke
   engagement with the task", which would show as drift toward the no-think value or as a
   distribution that simply gets wider.

Two things this is careful about:
  - Counts are reported PER 1000 THINK-BLOCK TOKENS as well as per rollout. The arms differ
    ~7x in trace length, so a raw per-rollout count mostly measures how long the arm reasons for;
    reading "the neutral sentence stops the model mentioning the bet" off the unnormalised
    numbers would be an artefact of that.
  - The forced prefix is STRIPPED before counting. The admission and denial prefills contain
    "donation" themselves, so counting the whole completion would score the manipulation instead
    of the reasoning.
  - Estimate quality is reported UNFILTERED. parse.apply_outlier_filter is calibrated to
    [T/10, 10T], which is a statement about the thinking-on baseline; the no-think baseline sits
    5.6x above T and loses 68/150 rollouts to it (see hypotheses.md). Filtering here would
    manufacture agreement between arms by construction.

Run with: python3 runs/a9_lexical_quality.py   (no GPU)
"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.qual import parse

HERE = Path(__file__).resolve().parent

# arm -> (run dir, [raw file stems]).  "free" is the unprefilled thinking-on intervention data.
ARMS = {
    "free_cot":        ("hour0",         ["giraffes_above_good", "giraffes_below_good"]),
    "thinkon_base":    ("hour0",         ["giraffes_baseline"]),
    "neutral":         ("prefill_tests", ["neutral_above_good", "neutral_below_good"]),
    "denial":          ("prefill_tests", ["denial_above_good", "denial_below_good"]),
    "admission":       ("prefill_tests", ["admission_above_good", "admission_below_good"]),
    "minimal":         ("prefill_extra", ["minimal_above_good", "minimal_below_good"]),
    "admit_then_deny": ("prefill_extra", ["admit_then_deny_above_good",
                                          "admit_then_deny_below_good"]),
    "admit_twice":     ("prefill_extra", ["admit_twice_above_good", "admit_twice_below_good"]),
    "nothink_base":    ("thinking_off",  ["nothink_baseline"]),
}

# The threshold value as the model would write it. 75,000,000 / 75000000 / 75 million / 75M.
PATTERNS = {
    "threshold_value": re.compile(r"\b75[,.]?0{3}[,.]?0{3}\b|\b75\s*(?:million|m\b|M\b)", re.I),
    "threshold_word":  re.compile(r"threshold", re.I),
    "donat":           re.compile(r"donat\w*", re.I),
    "cause":           re.compile(r"\bcauses?\b", re.I),
}


def think_block(rec):
    """The reasoning text only: between the opening <think> and </think>, with the forced prefix
    removed. Returns None if the rollout was truncated before closing the block."""
    raw = rec["raw_completion"]
    i = raw.find("</think>")
    if i == -1:
        return None
    block = raw[:i]
    if block.startswith("<think>"):
        block = block[len("<think>"):]
    prefix = rec.get("forced_prefix")
    if prefix:
        p = prefix[len("<think>"):] if prefix.startswith("<think>") else prefix
        if block.startswith(p):
            block = block[len(p):]
    return block


def main():
    res = {}
    print("per-rollout count / per-1000-think-tokens rate")
    print(f"{'arm':17s}{'n':>5s}{'med tok':>9s}{'thr_val':>13s}{'thr_word':>13s}{'donat':>13s}"
          f"{'cause':>13s}{'any':>7s}")
    for arm, (run, stems) in ARMS.items():
        recs = []
        for s in stems:
            recs += [json.loads(l) for l in open(HERE / run / "raw" / f"{s}.jsonl")]
        pairs = [(b, r) for b, r in ((think_block(r), r) for r in recs) if b is not None]
        blocks = [b for b, _ in pairs]
        # denominator: think-block tokens. num_tokens covers the whole completion, and for these
        # arms the post-</think> answer is short, but use a block-proportional estimate rather
        # than the raw field so a long final answer cannot deflate the rate.
        ktok = np.array([max(r["num_tokens"] * len(b) / max(len(r["raw_completion"]), 1), 1.0)
                         for b, r in pairs]) / 1000.0
        row = {"n_rollouts": len(recs), "n_with_think_block": len(blocks)}
        anys = np.zeros(len(blocks), dtype=bool)
        cells = []
        for name, rx in PATTERNS.items():
            counts = np.array([len(rx.findall(b)) for b in blocks], dtype=float)
            anys |= counts > 0
            rate = counts / ktok
            row[name] = {"mean_per_rollout": float(counts.mean()) if len(counts) else float("nan"),
                         "median": float(np.median(counts)) if len(counts) else float("nan"),
                         "per_1k_think_tokens": float(rate.mean()) if len(rate) else float("nan"),
                         "frac_with_ge1": float((counts > 0).mean()) if len(counts) else float("nan")}
            cells.append(f"{counts.mean():.2f}/{rate.mean():.2f}" if len(counts) else "nan")
        row["frac_any_incentive_mention"] = float(anys.mean()) if len(blocks) else float("nan")

        # estimate quality, unfiltered on purpose
        vals = []
        for r in recs:
            v, reason = parse.parse_estimate(r["raw_completion"])
            if v is not None and v > 0:
                vals.append(math.log10(v))
        v = np.array(vals)
        row["log10_estimate"] = {
            "n": len(v), "median": float(np.median(v)), "iqr": float(np.subtract(*np.percentile(v, [75, 25]))),
            "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)),
            "median_as_value": float(10 ** np.median(v))}
        res[arm] = row
        row["median_think_ktokens"] = float(np.median(ktok))
        print(f"{arm:17s}{len(blocks):5d}{np.median(ktok)*1000:9.0f}{cells[0]:>13s}{cells[1]:>13s}"
              f"{cells[2]:>13s}{cells[3]:>13s}{anys.mean():>7.2f}")

    print(f"\n{'arm':17s}{'n':>5s}{'median est':>14s}{'log10 med':>11s}{'IQR(log10)':>12s}"
          f"{'  [q25, q75] as values':>30s}")
    for arm in ARMS:
        e = res[arm]["log10_estimate"]
        print(f"{arm:17s}{e['n']:5d}{e['median_as_value']:14.3e}{e['median']:11.3f}"
              f"{e['iqr']:12.3f}   [{10**e['q25']:.2e}, {10**e['q75']:.2e}]")
    print("\n  references: T = 7.500e+07 (log10 7.875) | no-think baseline median 4.22e+08")
    print("  estimates are UNFILTERED — the [T/10, 10T] filter is calibrated to the "
          "thinking-on baseline and would distort the no-think arm (see hypotheses.md)")

    out = HERE / "prefill_extra" / "a9_lexical_quality.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
