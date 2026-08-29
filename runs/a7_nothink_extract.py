"""A7 step 1a — residual-stream activations at the pre-number position, THINKING-OFF rollouts.

Same target position and the same teacher-forcing procedure as internals_extract.py, pointed at
the thinking-off run. Two differences that matter and are the reason this is a separate file
rather than a flag:

  - the assistant prefill is the EMPTY closed think block, so it must be reconstructed with
    assistant_prefill(tok, enable_thinking=False). Using the thinking-on prefill would leave the
    body misaligned and every offset lookup would land on the wrong token.
  - there are no covertness labels here (no CoT, so nothing to admit or deny), so `label` is None
    throughout and no label-conditioned analysis is possible on this set.

Sequences are ~200 tokens rather than ~9k, so this is fast despite the larger n.

Run with: source env.sh && python3 runs/a7_nothink_extract.py [--n-per-arm 100]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.qual import parse, prompts, sample

HERE = Path(__file__).resolve().parent
REVISION = "fc05daec18b0a78c049392ed2e771dde82bdf654"
sys.path.insert(0, str(HERE))
from internals_extract import find_final_number_char_span    # noqa: E402  (shared locator)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="thinking_off")
    ap.add_argument("--threshold-from", default="hour0")
    ap.add_argument("--out", default="runs/internals/acts_nothink.pt")
    ap.add_argument("--n-per-arm", type=int, default=100)
    ap.add_argument("--max-len", type=int, default=16384)
    args = ap.parse_args()

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    src = HERE / args.run
    T = json.load(open(HERE / args.threshold_from / "threshold.json"))["threshold"]

    tok = AutoTokenizer.from_pretrained(sample.MODEL_ID, revision=REVISION)
    cfg = AutoConfig.from_pretrained(sample.MODEL_ID, revision=REVISION)
    text_cfg = getattr(cfg, "text_config", cfg)
    print(f"model: {text_cfg.num_hidden_layers} layers, d_model {text_cfg.hidden_size}")

    print("loading model (bf16)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        sample.MODEL_ID, revision=REVISION, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    layers = model.model.language_model.layers if hasattr(model.model, "language_model") \
        else model.model.layers
    print(f"hooked {len(layers)} decoder layers")

    captured, TARGET = {}, [0]

    def mk_hook(i):
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[i] = h[0, TARGET[0], :].detach().float().cpu()
        return hook

    handles = [l.register_forward_hook(mk_hook(i)) for i, l in enumerate(layers)]

    prefill = sample.assistant_prefill(tok, False)     # the empty closed think block
    print(f"thinking-off prefill = {prefill!r}")

    rows, acts, dropped = [], [], []
    for direction in ["above_good", "below_good"]:
        kept_this_arm = 0
        for rec in parse.parse_jsonl_file(src / "raw" / f"nothink_{direction}.jsonl"):
            if kept_this_arm >= args.n_per_arm:
                break
            iid = rec["item_id"]
            span = find_final_number_char_span(rec["raw_completion"])
            val, reason = parse.parse_estimate(rec["raw_completion"])
            if span is None or val is None:
                dropped.append((iid, reason))
                continue
            prompt_text = sample.build_prompt_text(
                tok, prompts.format_prompt(direction, threshold=T), enable_thinking=False)
            body = rec["raw_completion"][len(prefill):] if \
                rec["raw_completion"].startswith(prefill) else rec["raw_completion"]
            offset = len(prompt_text)
            full = prompt_text + body
            enc = tok(full, return_offsets_mapping=True, return_tensors="pt",
                      truncation=True, max_length=args.max_len)
            offs = enc["offset_mapping"][0].tolist()
            char_start = offset + (span[0] - len(prefill))
            tgt = next((i for i, (a, b) in enumerate(offs) if a <= char_start < b), None)
            if tgt is None or tgt == 0 or tgt >= enc["input_ids"].shape[1]:
                dropped.append((iid, "offset_map_failed"))
                continue
            TARGET[0] = tgt - 1
            with torch.no_grad():
                model(input_ids=enc["input_ids"].cuda(),
                      attention_mask=enc["attention_mask"].cuda(), use_cache=False)
            acts.append(torch.stack([captured[i] for i in range(len(layers))]))
            rows.append({"id": iid, "direction": direction, "label": None, "estimate": val,
                         "on_good_side": parse.classify_on_good_side(val, T, direction),
                         "tok_index": tgt - 1, "seq_len": int(enc["input_ids"].shape[1]),
                         "number_token": tok.decode(enc["input_ids"][0, tgt:tgt + 1])})
            kept_this_arm += 1
            if len(rows) % 25 == 0:
                print(f"  {len(rows)} done", flush=True)
    for h in handles:
        h.remove()

    A = torch.stack(acts)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"acts": A, "rows": rows, "n_layers": len(layers),
                "d_model": text_cfg.hidden_size, "threshold": T, "dropped": dropped}, out)
    n_ab = sum(r["direction"] == "above_good" for r in rows)
    print(f"\nkept {len(rows)} ({n_ab} above / {len(rows)-n_ab} below), "
          f"dropped {len(dropped)}: {dropped[:5]}")
    print(f"activations {tuple(A.shape)} -> {out}")


if __name__ == "__main__":
    main()
