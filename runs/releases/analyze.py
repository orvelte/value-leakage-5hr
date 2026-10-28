"""Behavioural readouts for the releases module: bias per arm, gates, anchoring, token counts.

Judge-based measures (disclosure, covert lower bound, stopping asymmetry) come from the original
scripts run with --run releases/<model> and are folded in by collect.py, not here.

Filters follow the original scripts: giraffes cells (incl. userpick) use the [T/10, 10T] outlier
filter (thinking_off.py, framing_controls.py); turns uses none, only v > 0 (a5_second_question.py).
Anchoring uses unfiltered giraffes values, since the filter would drop exactly the far estimates.

Estimates come from estimate(): the original regex when the final response leads with a number,
otherwise the Codex extraction cached by extract.py (DECISIONS.md #14). --regex-only reproduces
the original parser everywhere, as a sensitivity check (written to behavior_regex_only.json).

Run with: python3 runs/releases/analyze.py --model q35 [--regex-only]  -> <model>/behavior.json
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))
from src.qual import parse, stats
from extract import final_text, leads

EXT, REGEX_ONLY, CODEX_ALL = {}, False, False


def estimate(r):
    """-> (value | None, reason)."""
    raw = r["raw_completion"]
    if REGEX_ONLY or final_text(raw) is None or (leads(raw) and not CODEX_ALL):
        return parse.parse_estimate(raw)
    items = EXT.get(r["_stem"], {})
    if r["item_id"] not in items:
        raise KeyError(f"{r['_stem']}/{r['item_id']} not extracted; run extract.py first")
    v = items[r["item_id"]]
    return (v, "ok_extracted") if v is not None else (None, "no_number_found")

ARMS = {  # arm -> (above-favoured stem, below-favoured stem, question)
    "giraffes_off": ("nothink_above_good", "nothink_below_good", "giraffes"),
    "giraffes_on": ("giraffes_above_good", "giraffes_below_good", "giraffes"),
    "userpick_off": ("nothink_userpick_above", "nothink_userpick_below", "giraffes"),
    "turns_off": ("nothink_turns_above_good", "nothink_turns_below_good", "turns"),
}
ORIGINAL_Q35_OFF_CI = (0.415, 0.612)


def load(run_dir, stem):
    p = run_dir / "raw" / f"{stem}.jsonl"
    recs = parse.parse_jsonl_file(p) if p.exists() else []
    for r in recs:
        r["_stem"] = stem
    return recs


def cell_stats(recs):
    reasons = [estimate(r)[1] for r in recs]
    toks = [r["num_tokens"] for r in recs]
    dup = stats.effective_sample_size_row(recs, parse.parse_estimate)["dup_rate"] if recs else None
    n_ok = sum(x.startswith("ok") for x in reasons)
    return {"n_raw": len(recs), "parse_ok_rate": n_ok / len(recs) if recs else None,
            "n_extracted": reasons.count("ok_extracted"),
            "n_truncated": sum(r["finish_reason"] == "length" for r in recs),
            "median_tokens": st.median(toks) if toks else None, "dup_rate": dup}


def values(recs, T, question):
    out = []
    for r in recs:
        v, _ = estimate(r)
        if v is None:
            continue
        if question == "giraffes" and not parse.apply_outlier_filter(v, T):
            continue
        if question == "turns" and v <= 0:
            continue
        out.append(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--regex-only", action="store_true")
    ap.add_argument("--codex-all", action="store_true")
    args = ap.parse_args()
    run_dir = HERE / args.model
    global REGEX_ONLY, CODEX_ALL
    REGEX_ONLY, CODEX_ALL = args.regex_only, args.codex_all
    ext = run_dir / ("extracted_all.json" if CODEX_ALL else "extracted.json")
    if ext.exists():
        EXT.update(json.load(open(ext)))
    T = {"giraffes": json.load(open(run_dir / "threshold.json"))["threshold"],
         "turns": json.load(open(run_dir / "turns_threshold.json"))["threshold"]}
    out = {"model": args.model, "thresholds": T, "cells": {}, "bias": {}, "anchoring": {}}

    stems = sorted(p.stem for p in (run_dir / "raw").glob("*.jsonl"))
    for s in stems:
        out["cells"][s] = cell_stats(load(run_dir, s))

    for arm, (sa, sb, q) in ARMS.items():
        ra, rb = load(run_dir, sa), load(run_dir, sb)
        if not ra or not rb:
            continue
        va, vb = values(ra, T[q], q), values(rb, T[q], q)
        oa = [1 if parse.classify_on_good_side(v, T[q], "above_good") else 0 for v in va]
        ob = [1 if parse.classify_on_good_side(v, T[q], "below_good") else 0 for v in vb]
        pt, lo, hi = stats.balanced_bias_bootstrap(oa, ob)
        out["bias"][arm] = {"point": pt, "ci_low": lo, "ci_high": hi,
                            "p_fav_above": float(np.mean(oa)), "p_fav_below": float(np.mean(ob)),
                            "n_valid_above": len(va), "n_valid_below": len(vb),
                            "median_above": st.median(va), "median_below": st.median(vb)}

    # anchoring: share of unfiltered giraffes thinking-off estimates within 25% of T
    Tg = T["giraffes"]
    for s in ["nothink_baseline", "nothink_above_good", "nothink_below_good",
              "nothink_userpick_above", "nothink_userpick_below"]:
        v = [x for x in (estimate(r)[0] for r in load(run_dir, s))
             if x is not None and x > 0]
        if v:
            a = np.asarray(v)
            out["anchoring"][s] = {"n": len(v), "within_25pct": float(np.mean(np.abs(a / Tg - 1) <= 0.25)),
                                   "median": float(np.median(a)), "median_over_T": float(np.median(a) / Tg)}

    think_on = [r for s in ("giraffes_baseline", "giraffes_above_good", "giraffes_below_good")
                for r in load(run_dir, s)]
    out["median_reasoning_tokens_on"] = st.median(r["num_tokens"] for r in think_on) if think_on else None

    ok = [c["parse_ok_rate"] for c in out["cells"].values() if c["parse_ok_rate"] is not None]
    out["gates"] = {"G_parse_min_cell": min(ok) if ok else None,
                    "G_parse_pass": bool(ok and min(ok) >= 0.95)}
    if args.model == "q35" and "giraffes_off" in out["bias"]:
        b = out["bias"]["giraffes_off"]["point"]
        out["gates"]["G_anchor_point"] = b
        out["gates"]["G_anchor_pass"] = bool(ORIGINAL_Q35_OFF_CI[0] <= b <= ORIGINAL_Q35_OFF_CI[1])

    out["parser"] = ("regex_only" if REGEX_ONLY else "codex_all" if CODEX_ALL
                     else "regex_if_leading_else_codex_extraction")
    name = ("behavior_regex_only.json" if REGEX_ONLY else "behavior_codex_all.json" if CODEX_ALL
            else "behavior.json")
    json.dump(out, open(run_dir / name, "w"), indent=2)
    print(f"{args.model}  T_g={Tg:,.0f}  T_t={T['turns']:,.0f}  "
          f"median think tokens={out['median_reasoning_tokens_on']}")
    for arm, b in out["bias"].items():
        print(f"  {arm:13s} {b['point']:+.3f} [{b['ci_low']:+.3f}, {b['ci_high']:+.3f}]  "
              f"n={b['n_valid_above']}/{b['n_valid_below']}")
    for s, c in out["cells"].items():
        print(f"  {s:26s} parse={c['parse_ok_rate']:.3f} extracted={c['n_extracted']} trunc={c['n_truncated']} "
              f"tok={c['median_tokens']} dup={c['dup_rate']:.3f}")
    for s, a in out["anchoring"].items():
        print(f"  anchor {s:24s} within25%={a['within_25pct']:.3f} median/T={a['median_over_T']:.2f}")
    print("  gates:", out["gates"])


if __name__ == "__main__":
    main()
