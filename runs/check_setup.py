"""Pre-flight checks before spending GPU time on the hour-0 pilot.

Each check is one thing that would silently corrupt a downstream number if it were wrong:
CUDA forward-compat actually working, the chat template still auto-opening <think> (which
resample.py asserts and sample.py's raw_completion reconstruction assumes), single-token
coverage for the J-lens concept probes, and the lens's layer indexing against this model.

Run with: source env.sh && python3 runs/check_setup.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL_ID = "Qwen/Qwen3.5-27B"
REVISION = "fc05daec18b0a78c049392ed2e771dde82bdf654"
PROBE_WORDS = ["donate", "donation", "charity", "cause", "threshold", "above", "below",
               "honest", "bias", "good", "bad", "estimate"]

results = {}


def check(name, fn):
    try:
        ok, detail = fn()
    except Exception as e:  # noqa: BLE001 - a failed check is a reportable result, not a crash
        ok, detail = False, f"{type(e).__name__}: {e}"
    results[name] = {"ok": bool(ok), "detail": detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    return ok


def c_torch_cuda():
    import torch
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False"
    x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    y = (x @ x).float().abs().mean().item()
    return True, (f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
                  f"bf16 matmul ok (mean |y|={y:.1f})")


def c_versions():
    import importlib.metadata as md
    v = {p: md.version(p) for p in ["vllm", "torch", "transformers", "huggingface_hub",
                                    "numpy", "scipy"]}
    return v["vllm"] == "0.27.1", json.dumps(v)


def c_chat_template():
    """resample.py asserts base_prompt.endswith("<think>\\n"); sample.py prepends "<think>\\n"
    to every completion. If the template stops auto-opening the think block, prefix-forcing
    silently produces a malformed prompt instead of raising."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    txt = tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False,
                                  add_generation_prompt=True)
    globals()["_TOK"] = tok
    return txt.endswith("<think>\n"), f"template tail = {txt[-40:]!r}"


def c_probe_tokens():
    """The planning doc's constraint: check single-token coverage before assuming a J-lens
    vector exists for a concept. Reports both bare and leading-space forms."""
    tok = globals().get("_TOK")
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    table = {}
    for w in PROBE_WORDS:
        row = {}
        for variant, s in [("bare", w), ("space", " " + w), ("cap", w.capitalize()),
                           ("space_cap", " " + w.capitalize())]:
            ids = tok.encode(s, add_special_tokens=False)
            row[variant] = {"n_tokens": len(ids), "ids": ids}
        table[w] = row
    single = [w for w, r in table.items() if r["space"]["n_tokens"] == 1]
    multi = [w for w, r in table.items() if r["space"]["n_tokens"] > 1]
    out = Path(__file__).resolve().parent / "probe_token_coverage.json"
    out.write_text(json.dumps(table, indent=2))
    return len(single) > 0, (f"single-token (' word' form): {single} | multi-token: {multi} "
                             f"-> {out}")


def c_lens():
    from src.qual import lens as qual_lens
    from transformers import AutoConfig
    path = qual_lens.download_lens("qwen3.5-27b", "j-lens")
    L = qual_lens.load_lens(path)
    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION)
    findings = qual_lens.assert_layer_indexing(L, cfg)
    # J is a dict {source_layer:int -> [d_model, d_model] fp16}, NOT the stacked tensor the
    # lens.py docstring implied. lens.readout's `J[source_layers.index(l)]` still resolves
    # correctly only because source_layers is contiguous 0..62, so index(l) == l == the dict
    # key. Record the shape per layer so a non-contiguous lens would show up here.
    J = L["J"]
    per_layer = {int(k): (list(v.shape), str(v.dtype)) for k, v in J.items()}
    shapes = {tuple(s) for s, _ in per_layer.values()}
    findings["J_container"] = type(J).__name__
    findings["J_n_layers"] = len(per_layer)
    findings["J_layer_keys_contiguous"] = sorted(per_layer) == list(range(len(per_layer)))
    findings["J_shape_per_layer"] = sorted(shapes)
    findings["J_dtype"] = sorted({d for _, d in per_layer.values()})
    findings["n_prompts"] = int(L["n_prompts"])
    findings["provenance"] = {k: str(v) for k, v in L["provenance"].items()}
    out = Path(__file__).resolve().parent / "lens_indexing_check.json"
    out.write_text(json.dumps(findings, indent=2, default=str))
    ok = (findings["target_layer"] == 62
          and findings["target_layer_type"] == "linear_attention"
          and findings["J_layer_keys_contiguous"]
          and len(shapes) == 1
          and findings["provenance"]["model_id"] == MODEL_ID)
    return ok, (f"J = {findings['J_container']} of {findings['J_n_layers']} x "
                f"{findings['J_shape_per_layer'][0]} {findings['J_dtype']}, "
                f"target_layer={findings['target_layer']} ({findings['target_layer_type']}), "
                f"source_layers={findings['source_layers'][0]}..{findings['source_layers'][-1]}, "
                f"provenance model={findings['provenance']['model_id']} -> {out}")


def c_parse_segment():
    from src.qual import parse, segment
    cases = [
        ("<think>\nSome reasoning here. More of it.</think>\n\n**1.5 billion** spots.", 1.5e9),
        ("<think>\nx</think>\nAround 250,000,000.", 250_000_000.0),
        ("<think>\nno close tag ever", None),
    ]
    bad = []
    for raw, want in cases:
        got, reason = parse.parse_estimate(raw)
        if got != want:
            bad.append((raw[:30], got, want, reason))
    sents = segment.segment_sentences("<think>\nFirst one. Second is 1.5 here.\n\nNew para.</think>")
    return not bad, f"parse mismatches={bad}, segment -> {sents}"


for nm, fn in [("torch+cuda (forward-compat)", c_torch_cuda), ("package versions", c_versions),
               ("chat template opens <think>", c_chat_template),
               ("J-lens probe token coverage", c_probe_tokens),
               ("J-lens layer indexing", c_lens), ("parse/segment sanity", c_parse_segment)]:
    check(nm, fn)

Path(__file__).resolve().parent.joinpath("check_setup_results.json").write_text(
    json.dumps(results, indent=2))
n_fail = sum(1 for r in results.values() if not r["ok"])
print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
sys.exit(1 if n_fail else 0)
