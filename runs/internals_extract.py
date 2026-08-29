"""Extract residual-stream activations at the position immediately before the final number.

That position is where the plan says to look: whatever is going to bias the digit has to be
present there. Teacher-forcing the model over prompt+completion and hooking each decoder layer
gives one [n_layers, d_model] vector per rollout, which is all the downstream analysis needs.

Two things worth knowing about the target position:
  - It is the token BEFORE the first token of the final estimate, located by character offset of
    the regex match that parse.parse_estimate itself used, then mapped through the tokenizer's
    offset mapping. Any rollout where that mapping fails is dropped and reported, never guessed.
  - Digit tokens are deliberately NOT read through the J-lens downstream (adjacent digit
    directions are near-collinear, cos 0.92-0.997). This file only stores activations; the
    constraint applies to how they are used.

vLLM and HF transformers both want the whole GPU. Do not run this while a generation job is up.

Run with: source env.sh && python3 runs/internals_extract.py
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.qual import parse, prompts, sample

REVISION = "fc05daec18b0a78c049392ed2e771dde82bdf654"


def find_final_number_char_span(raw_completion):
    """Return (start, end) char offsets of the final estimate inside raw_completion, or None.

    Mirrors parse.parse_estimate's own search so the activation position corresponds to the
    number that every behavioural number in this project was computed from.
    """
    idx = raw_completion.find(parse._THINK_CLOSE)
    if idx == -1:
        return None
    tail_start = idx + len(parse._THINK_CLOSE)
    tail = raw_completion[tail_start:]
    stripped = parse._strip_markdown_bold(tail)
    if stripped != tail:          # bold-stripping shifts offsets; fall back to searching raw
        m = parse._NUMBER_RE.search(tail)
    else:
        m = parse._NUMBER_RE.search(tail)
    if m is None:
        return None
    return tail_start + m.start(1), tail_start + m.end(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="hour0")
    ap.add_argument("--out", default="runs/internals/acts.pt")
    ap.add_argument("--max-len", type=int, default=16384)
    args = ap.parse_args()

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    src = Path(__file__).resolve().parent / args.run
    T = json.load(open(src / "threshold.json"))["threshold"]
    final = json.load(open(src / "covertness_majority.json"))["final_labels"]
    lab = {k.replace("covertness_above_good_", "").replace("covertness_below_good_", ""): v
           for k, v in final.items()}

    tok = AutoTokenizer.from_pretrained(sample.MODEL_ID, revision=REVISION)
    cfg = AutoConfig.from_pretrained(sample.MODEL_ID, revision=REVISION)
    text_cfg = getattr(cfg, "text_config", cfg)
    n_layers, d_model = text_cfg.num_hidden_layers, text_cfg.hidden_size
    print(f"model: {n_layers} layers, d_model {d_model}")

    print("loading model (bf16)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        sample.MODEL_ID, revision=REVISION, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    layers = model.model.language_model.layers if hasattr(model.model, "language_model") \
        else model.model.layers
    print(f"hooked {len(layers)} decoder layers")

    captured = {}

    def mk_hook(i):
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[i] = h[0, TARGET[0], :].detach().float().cpu()
        return hook

    handles = [l.register_forward_hook(mk_hook(i)) for i, l in enumerate(layers)]
    TARGET = [0]

    rows, acts, dropped = [], [], []
    for direction in ["above_good", "below_good"]:
        for rec in parse.parse_jsonl_file(src / "raw" / f"giraffes_{direction}.jsonl"):
            iid = rec["item_id"]
            span = find_final_number_char_span(rec["raw_completion"])
            val, reason = parse.parse_estimate(rec["raw_completion"])
            if span is None or val is None:
                dropped.append((iid, reason))
                continue
            prompt_text = sample.build_prompt_text(
                tok, prompts.format_prompt(direction, threshold=T))
            prefill = sample.assistant_prefill(tok, True)
            body = rec["raw_completion"][len(prefill):] if \
                rec["raw_completion"].startswith(prefill) else rec["raw_completion"]
            offset = len(prompt_text)
            full = prompt_text + body
            enc = tok(full, return_offsets_mapping=True, return_tensors="pt",
                      truncation=True, max_length=args.max_len)
            offs = enc["offset_mapping"][0].tolist()
            # span is relative to raw_completion, which begins with the prefill
            char_start = offset + (span[0] - len(prefill))
            tgt = next((i for i, (a, b) in enumerate(offs) if a <= char_start < b), None)
            if tgt is None or tgt == 0 or tgt >= enc["input_ids"].shape[1]:
                dropped.append((iid, "offset_map_failed"))
                continue
            TARGET[0] = tgt - 1          # the position BEFORE the first digit token
            with torch.no_grad():
                model(input_ids=enc["input_ids"].cuda(),
                      attention_mask=enc["attention_mask"].cuda(), use_cache=False)
            acts.append(torch.stack([captured[i] for i in range(len(layers))]))
            rows.append({"id": iid, "direction": direction, "label": lab.get(iid),
                         "estimate": val,
                         "on_good_side": parse.classify_on_good_side(val, T, direction),
                         "tok_index": tgt - 1, "seq_len": int(enc["input_ids"].shape[1]),
                         "number_token": tok.decode(enc["input_ids"][0, tgt:tgt + 1])})
            if len(rows) % 10 == 0:
                print(f"  {len(rows)} done", flush=True)
    for h in handles:
        h.remove()

    A = torch.stack(acts)  # [n, n_layers, d_model]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"acts": A, "rows": rows, "n_layers": len(layers), "d_model": d_model,
                "threshold": T, "dropped": dropped}, out)
    print(f"\nkept {len(rows)}, dropped {len(dropped)}: {dropped[:5]}")
    print(f"activations {tuple(A.shape)} -> {out}")
    print("sample number tokens:", [r["number_token"] for r in rows[:8]])


if __name__ == "__main__":
    main()
