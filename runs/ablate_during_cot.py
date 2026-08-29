"""Ablate the incentive direction DURING chain-of-thought generation.

Every causal test so far has intervened at or after the position where the answer is emitted,
and all three saturated for the same reason: the final number already appears verbatim in the
CoT in 90% of rollouts, so those tests probe the copy step rather than the choice. This one
projects the direction out at every forward pass while the CoT is being written, so the model
never gets to use it in the first place.

Conditions:
  none    unablated, generated the same way (HF, not vLLM) so the comparison is like-for-like
  dom     the overt-fitted difference-of-means direction projected out from the fitted layer on
  random  a random direction of matched norm, the control for "perturbing the stream at all"

The baseline (no-bet) arm is run under `none` and `dom` too. The plan's bar for a real result is
a bias drop with baseline estimates UNTOUCHED -- if ablation also moves the no-bet estimate, the
direction is doing something general rather than carrying the incentive.

This is the expensive one: HF generation of full CoTs on a 27B model. Sized for a few hours.

Run with: source env.sh && python3 runs/ablate_during_cot.py [--n 12]
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.qual import parse, prompts, sample, stats

REVISION = "fc05daec18b0a78c049392ed2e771dde82bdf654"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", default="ablate_during_cot")
    ap.add_argument("--n", type=int, default=12, help="rollouts per (condition, arm)")
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--max-new", type=int, default=12000)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--threshold-from", default="hour0")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    here = Path(__file__).resolve().parent
    run_dir = here / args.stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    T = json.load(open(here / args.threshold_from / "threshold.json"))["threshold"]

    d = torch.load("runs/internals/acts.pt", weights_only=False)
    A, rows = d["acts"].numpy(), d["rows"]
    res = json.load(open("runs/internals/results.json"))
    layer = args.layer if args.layer is not None else res["best_layer"]
    y = np.array([1 if r["direction"] == "above_good" else 0 for r in rows])
    overt = np.array([r["label"] == "INFLUENCED" for r in rows])
    X = A[:, layer, :]
    dv = X[overt & (y == 1)].mean(0) - X[overt & (y == 0)].mean(0)
    dv = dv / np.linalg.norm(dv)
    rv = np.random.default_rng(0).standard_normal(dv.shape[0])
    rv = rv / np.linalg.norm(rv)
    print(f"direction from {int(overt.sum())} overt rollouts at layer {layer}; threshold {T:,.0f}")

    tok = AutoTokenizer.from_pretrained(sample.MODEL_ID, revision=REVISION)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        sample.MODEL_ID, revision=REVISION, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    layers = model.model.language_model.layers if hasattr(model.model, "language_model") \
        else model.model.layers

    VEC = {"v": None}

    def hook(_m, _i, out):
        if VEC["v"] is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        v = VEC["v"].to(h.device, h.dtype)
        h = h - (h @ v).unsqueeze(-1) * v
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    handles = [l.register_forward_hook(hook) for l in layers[layer:]]

    cells = [(c, arm) for arm in ("none", "dom", "random")
             for c in ("above_good", "below_good")]
    cells += [("baseline", "none"), ("baseline", "dom")]

    out_rows = []
    for cond, arm in cells:
        text = prompts.format_prompt(cond, threshold=None if cond == "baseline" else T)
        prompt = sample.build_prompt_text(tok, text)
        VEC["v"] = None if arm == "none" else torch.tensor(
            dv if arm == "dom" else rv, dtype=torch.float32)
        print(f"\n=== {cond} / ablate={arm} (n={args.n}) ===", flush=True)
        done = 0
        while done < args.n:
            bs = min(args.batch, args.n - done)
            enc = tok([prompt] * bs, return_tensors="pt", padding=True)
            enc = {k: v.cuda() for k, v in enc.items()}
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new, do_sample=True,
                                     temperature=1.0, top_p=1.0, top_k=0,
                                     pad_token_id=tok.pad_token_id)
            for g in gen:
                txt = tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=False)
                raw = sample.assistant_prefill(tok, True) + txt
                v, reason = parse.parse_estimate(raw)
                out_rows.append({"condition": cond, "arm": arm, "estimate": v,
                                 "parse_reason": reason, "n_tokens": int(g.shape[0]),
                                 "raw_completion": raw})
            done += bs
            print(f"  {done}/{args.n}", flush=True)
        sample.append_jsonl(run_dir / "raw" / f"{cond}_{arm}.jsonl",
                            [r for r in out_rows if r["condition"] == cond and r["arm"] == arm])
        VEC["v"] = None
    for h in handles:
        h.remove()

    print("\n=== bias by ablation arm (CoT generated under ablation) ===")
    summary = {"layer": layer, "n": args.n, "threshold": T, "arms": {}}
    for arm in ("none", "dom", "random"):
        o = {}
        for c in ("above_good", "below_good"):
            vals = [r["estimate"] for r in out_rows if r["arm"] == arm and r["condition"] == c
                    and r["estimate"] is not None and parse.apply_outlier_filter(r["estimate"], T)]
            o[c] = [1 if parse.classify_on_good_side(v, T, c) else 0 for v in vals]
        if not (o["above_good"] and o["below_good"]):
            continue
        pt, lo, hi = stats.balanced_bias_bootstrap(o["above_good"], o["below_good"])
        summary["arms"][arm] = {"bias": pt, "ci_low": lo, "ci_high": hi,
                                "n_above": len(o["above_good"]), "n_below": len(o["below_good"])}
        print(f"  {arm:7s} bias={pt:+.3f} [{lo:+.3f},{hi:+.3f}]  "
              f"n={len(o['above_good'])}/{len(o['below_good'])}")
    print("  reference, vLLM unablated: +0.420 [+0.220, +0.622]")

    print("\n=== side-effect check: are no-bet baseline estimates untouched? ===")
    for arm in ("none", "dom"):
        vals = [r["estimate"] for r in out_rows
                if r["condition"] == "baseline" and r["arm"] == arm and r["estimate"] is not None]
        if vals:
            summary.setdefault("baseline", {})[arm] = {"n": len(vals),
                                                       "median": float(st.median(vals))}
            print(f"  baseline / {arm:5s} n={len(vals):3d} median={st.median(vals):,.0f}")
    if summary.get("baseline", {}).get("none") and summary["baseline"].get("dom"):
        a = [r["estimate"] for r in out_rows if r["condition"] == "baseline"
             and r["arm"] == "none" and r["estimate"]]
        b = [r["estimate"] for r in out_rows if r["condition"] == "baseline"
             and r["arm"] == "dom" and r["estimate"]]
        ks, p = stats.two_sample_ks(a, b)
        summary["baseline"]["ks_none_vs_dom"] = {"D": ks, "p": p}
        print(f"  KS(none vs dom) D={ks:.3f} p={p:.3f}  -> "
              f"{'baseline MOVED, direction is not incentive-specific' if p < 0.05 else 'baseline untouched'}")

    json.dump(summary, open(run_dir / "results.json", "w"), indent=2)
    print(f"\nwrote {run_dir/'results.json'}")


if __name__ == "__main__":
    main()
