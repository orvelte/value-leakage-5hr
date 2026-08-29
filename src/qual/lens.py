"""J-lens / R-lens loading + application.

Recipe (confirmed against camilablank/workspace-lenses' own README):
  lens.pt = {"J": stacked per-layer Jacobians, "n_prompts", "source_layers", "d_model",
             "provenance": {model_id, target_layer, skip_first, n_prompts, dataset_id, ...}}
  readout: softmax(W_U @ norm(J_l @ h_l))  — apply the MODEL's OWN final norm + unembedding
  directly to the Jacobian-transported vector; do not run the real remaining transformer layers.

qwen3.5-27b: target_layer=62 (0-indexed, n_layers-2), d_model=5120, skip_first=4.
Qwen3.5-27B config: num_hidden_layers=64, hidden_size=5120, full_attention_interval=4,
layer_types has full_attention at 0-indexed {3,7,...,63} and linear_attention elsewhere —
layer 62 (the lens's target) is itself a linear_attention layer, one before the model's own
final (full_attention) layer 63. This file asserts source_layers against the model's actual
layer_types before trusting any readout — an easy off-by-one to get wrong silently.
"""
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

LENS_REPO = "camilablank/workspace-lenses"


def download_lens(model_dir, lens_kind, local_dir=None):
    """model_dir e.g. 'qwen3.5-27b', lens_kind 'j-lens' or 'r-lens'."""
    filename = f"{model_dir}/{lens_kind}/lens.pt"
    path = hf_hub_download(repo_id=LENS_REPO, filename=filename, local_dir=local_dir)
    return path


def load_lens(path):
    lens = torch.load(path, map_location="cpu", weights_only=False)
    required = {"J", "n_prompts", "source_layers", "d_model", "provenance"}
    missing = required - set(lens.keys())
    if missing:
        raise ValueError(f"lens.pt at {path} missing expected keys: {missing}")
    return lens


def assert_layer_indexing(lens, model_config):
    """Cross-check the lens's source_layers / target_layer against the model's ACTUAL
    layer_types, rather than assuming naive model.layers[i] correspondence. Returns a dict of
    findings; raises AssertionError only on a hard shape mismatch (d_model), since layer-type
    mismatches are informative rather than fatal — report them, don't crash on them."""
    text_cfg = getattr(model_config, "text_config", model_config)
    d_model = getattr(text_cfg, "hidden_size")
    n_layers = getattr(text_cfg, "num_hidden_layers")
    layer_types = getattr(text_cfg, "layer_types", None)

    assert lens["d_model"] == d_model, (
        f"lens d_model={lens['d_model']} != model hidden_size={d_model}"
    )

    prov = lens["provenance"]
    target_layer = prov.get("target_layer")
    source_layers = lens["source_layers"]
    if hasattr(source_layers, "tolist"):
        source_layers = source_layers.tolist()

    findings = {
        "d_model_match": True,
        "n_layers_model": n_layers,
        "target_layer": target_layer,
        "target_layer_in_range": target_layer is not None and 0 <= target_layer < n_layers,
        "target_layer_type": layer_types[target_layer] if layer_types and target_layer is not None else None,
        "source_layers": source_layers,
        "source_layers_in_range": all(0 <= s < n_layers for s in source_layers) if source_layers else None,
        "source_layer_types": [layer_types[s] for s in source_layers] if layer_types and source_layers else None,
    }
    return findings


def readout(lens, hidden_state_at_layer, source_layer_idx, final_norm_fn, lm_head_weight,
            topk=10):
    """hidden_state_at_layer: tensor [d_model] — the residual stream at `source_layer_idx`
    and the token position of interest. final_norm_fn: the model's own final-layer norm
    module (callable). lm_head_weight: the model's own unembedding matrix [vocab, d_model].

    Returns (probs[vocab], topk_indices, topk_values). Use this when you want the full
    ranked-vocabulary readout (e.g. decoding top-k words for a sanity-check control prompt);
    for the concept-cosine probe (cosine of h_l against a specific token's J-lens vector v_t,
    without going through softmax/unembedding), pull the row directly from lens["J"] instead.
    """
    J = lens["J"]  # shape: [n_source_layers, d_model, d_model] (or similar; verified at runtime)
    source_layers = lens["source_layers"]
    if hasattr(source_layers, "tolist"):
        source_layers = source_layers.tolist()
    if source_layer_idx not in source_layers:
        raise ValueError(f"layer {source_layer_idx} not in lens source_layers {source_layers}")
    row = source_layers.index(source_layer_idx)
    J_l = J[row].to(device=hidden_state_at_layer.device, dtype=hidden_state_at_layer.dtype)

    transported = J_l @ hidden_state_at_layer  # [d_model]
    normed = final_norm_fn(transported)
    logits = (lm_head_weight.to(dtype=normed.dtype) @ normed).float()  # [vocab]
    probs = torch.softmax(logits, dim=-1)
    top = torch.topk(probs, topk)
    return probs, top.indices, top.values
