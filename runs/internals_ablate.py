"""Causal test: hold the CoT fixed, project out the incentive direction, re-emit the number.

The read half of the internals block is correlational. This is the cheapest thing that makes it
causal without regenerating whole chains of thought: take prompt + the model's own CoT up to
`</think>`, then generate ONLY the final answer, once normally and once with the
difference-of-means direction projected out of the residual stream from the fitted layer onward.
If the direction mediates the number, the condition gap should shrink under ablation while the
answers stay well-formed.

Controls, both necessary:
  random  a random direction of matched norm, ablated the same way. Without it, any change could
          be "perturbing the residual stream at all breaks the answer".
  none    unablated, regenerated the same way, so the comparison is generation-to-generation
          rather than generation-to-the-original-rollout.

The direction is fitted on OVERT rollouts only and applied to everything, so its effect on covert
rollouts is out of sample.

Run with: source env.sh && python3 runs/internals_ablate.py [--n-samples 3]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.qual import parse, prompts, sample, stats

REVISION = "fc05daec18b0a78c049392ed2e771dde82bdf654"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="hour0")
    ap.add_argument("--acts", default="runs/internals/acts.pt")
    ap.add_argument("--layer", type=int, default=None, help="default: best CV layer from results")
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=60)
    ap.add_argument("--out", default="runs/internals/ablation.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    src = Path(__file__).resolve().parent / args.run
    T = json.load(open(src / "threshold.json"))["threshold"]
    d = torch.load(args.acts, weights_only=False)
    A, rows = d["acts"].numpy(), d["rows"]
    res = json.load(open("runs/internals/results.json"))
    layer = args.layer if args.layer is not None else res["best_layer"]
    y = np.array([1 if r["direction"] == "above_good" else 0 for r in rows])
    overt = np.array([r["label"] == "INFLUENCED" for r in rows])

    X = A[:, layer, :]
    dv = X[overt & (y == 1)].mean(0) - X[overt & (y == 0)].mean(0)
    dv = dv / np.linalg.norm(dv)
    rng = np.random.default_rng(0)
    rv = rng.standard_normal(dv.shape[0])
    rv = rv / np.linalg.norm(rv)
    print(f"direction fitted on {int(overt.sum())} overt rollouts at layer {layer}; "
          f"|d|=1, random control matched\n")

    tok = AutoTokenizer.from_pretrained(sample.MODEL_ID, revision=REVISION)
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
        h = h - (h @ v).unsqueeze(-1) * v          # project the component out
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    handles = [l.register_forward_hook(hook) for l in layers[layer:]]

    recs = {}
    for direction in ["above_good", "below_good"]:
        for r in parse.parse_jsonl_file(src / "raw" / f"giraffes_{direction}.jsonl"):
            recs[r["item_id"]] = r

    out_rows = []
    for n, meta in enumerate(rows):
        rec = recs[meta["id"]]
        cot = rec["raw_completion"].split("</think>", 1)[0] + "</think>\n\n"
        prefill = sample.assistant_prefill(tok, True)
        body = cot[len(prefill):] if cot.startswith(prefill) else cot
        text = sample.build_prompt_text(
            tok, prompts.format_prompt(meta["direction"], threshold=T)) + body
        ids = tok(text, return_tensors="pt", truncation=True, max_length=16000)
        ids = {k: v.cuda() for k, v in ids.items()}
        for cond, vec in (("none", None), ("dom", dv), ("random", rv)):
            VEC["v"] = None if vec is None else torch.tensor(vec, dtype=torch.float32)
            with torch.no_grad():
                gen = model.generate(**ids, max_new_tokens=args.max_new, do_sample=True,
                                     temperature=1.0, top_p=1.0, top_k=0,
                                     num_return_sequences=args.n_samples,
                                     pad_token_id=tok.eos_token_id)
            for g in gen:
                ans = tok.decode(g[ids["input_ids"].shape[1]:], skip_special_tokens=True)
                v, _ = parse.parse_estimate("<think>\n</think>" + ans)
                out_rows.append({"id": meta["id"], "direction": meta["direction"],
                                 "label": meta["label"], "cond": cond, "estimate": v,
                                 "answer": ans[:120]})
        VEC["v"] = None
        if (n + 1) % 10 == 0:
            print(f"  {n+1}/{len(rows)} rollouts", flush=True)
    for h in handles:
        h.remove()

    print("\n=== bias by ablation condition (CoT held fixed, only the answer regenerated) ===")
    summary = {"layer": layer, "n_samples": args.n_samples, "by_cond": {}}
    for cond in ["none", "dom", "random"]:
        o = {}
        for dr in ["above_good", "below_good"]:
            vals = [r["estimate"] for r in out_rows
                    if r["cond"] == cond and r["direction"] == dr
                    and r["estimate"] is not None
                    and parse.apply_outlier_filter(r["estimate"], T)]
            o[dr] = [1 if parse.classify_on_good_side(v, T, dr) else 0 for v in vals]
        pt, lo, hi = stats.balanced_bias_bootstrap(o["above_good"], o["below_good"])
        n_par = sum(1 for r in out_rows if r["cond"] == cond and r["estimate"] is not None)
        n_tot = sum(1 for r in out_rows if r["cond"] == cond)
        summary["by_cond"][cond] = {"bias": pt, "ci_low": lo, "ci_high": hi,
                                    "n_above": len(o["above_good"]), "n_below": len(o["below_good"]),
                                    "parse_rate": n_par / n_tot}
        print(f"  {cond:7s} bias={pt:+.3f} [{lo:+.3f},{hi:+.3f}]  "
              f"n={len(o['above_good'])}/{len(o['below_good'])}  parsed {n_par/n_tot:.0%}")

    for cond in ["none", "dom", "random"]:
        for grp, m in (("covert", "NOT_INFLUENCED"), ("overt", "INFLUENCED")):
            o = {}
            for dr in ["above_good", "below_good"]:
                vals = [r["estimate"] for r in out_rows
                        if r["cond"] == cond and r["direction"] == dr and r["label"] == m
                        and r["estimate"] is not None
                        and parse.apply_outlier_filter(r["estimate"], T)]
                o[dr] = [1 if parse.classify_on_good_side(v, T, dr) else 0 for v in vals]
            if o["above_good"] and o["below_good"]:
                pt, lo, hi = stats.balanced_bias_bootstrap(o["above_good"], o["below_good"])
                summary["by_cond"][f"{cond}/{grp}"] = {"bias": pt, "ci_low": lo, "ci_high": hi}

    Path(args.out).write_text(json.dumps({"summary": summary, "rows": out_rows}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
