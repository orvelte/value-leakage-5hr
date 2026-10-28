"""vLLM offline batch generation for the Donation Bet giraffes task.

COPY of src/qual/sample.py for runs/releases (spec: copy, don't edit, the shared module).
Changes vs the original, and nothing else:
  - MODEL_ID is a module global set by set_model() so one driver serves 3.5 / 3.6 / 3.8,
    with each model pinned to an HF revision (MODELS below).
  - build_prompt_text passes reasoning_effort to the chat template when REASONING_EFFORT is
    set. Only the 3.8 template reads it; 3.5/3.6 templates have no effort knob (verified on the
    pinned revisions), so it is left None for them and their prompts are byte-identical to the
    original run's.

Writes append-only raw JSONL to runs/<stamp>/raw/<condition>.jsonl. Never overwrites or
filters in place — parsing/filtering happens downstream in parse.py.
"""
import json
import os
import time
from pathlib import Path

MODELS = {
    "q35": ("Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654", None),
    "q36": ("Qwen/Qwen3.6-27B", "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9", None),
    # 3.8's template default is xhigh; set explicitly so the choice is visible in config.json.
    "q38": ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", "xhigh"),
}
MODEL_ID, MODEL_REVISION, REASONING_EFFORT = MODELS["q35"]


def set_model(key):
    global MODEL_ID, MODEL_REVISION, REASONING_EFFORT
    MODEL_ID, MODEL_REVISION, REASONING_EFFORT = MODELS[key]
MAX_MODEL_LEN = 16384
MAX_TOKENS = 16000
TEMPERATURE = 1.0
ENGINE_SEED = 0  # model-init reproducibility only — NOT passed to SamplingParams.
# GOTCHA (found the hard way in the prior round of this project): passing the same explicit
# `seed` to every request's SamplingParams gives every request its own torch.Generator seeded
# to the SAME value (vllm/v1/worker/gpu_model_runner.py: generator.manual_seed(sampling_params.
# seed)), which correlates sampling across concurrent requests with similar prompts — observed
# as up to 43% byte-identical completions within a single condition. Requests must use
# seed=None (vLLM's default nondeterministic per-request RNG) to be independent draws. Verify
# with stats.effective_sample_size_row's dup_rate before trusting any n.


_ASSISTANT_HEADER = "<|im_start|>assistant\n"


def build_prompt_text(tokenizer, user_content, enable_thinking=True):
    """enable_thinking=False makes the chat template pre-fill an EMPTY, already-closed think
    block ('<think>\\n\\n</think>\\n\\n') so the model answers directly. Verified against
    Qwen/Qwen3.5-27B's tokenizer_config.json.

    Note this means "thinking off" is itself a prefill -- the model is not run in a different
    mode, it is handed a closed scratchpad. It is the same class of manipulation as the
    forced-denial prefill, so report the two together and apply the off-distribution caveat to
    both. Do NOT use a '/no_think' string in the user prompt: unlike Qwen3, this template has no
    soft-switch handling, so it would just be text the model may react to as an instruction.
    """
    kw = {"reasoning_effort": REASONING_EFFORT} if REASONING_EFFORT is not None else {}
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        **kw,
    )


def assistant_prefill(tokenizer, enable_thinking=True):
    """The part of the generation prompt the template puts INSIDE the assistant turn, which the
    model's own completion continues from. Completions must be reconstructed as
    prefill + completion.text so downstream parsing sees a well-formed turn.

    Derived from the template rather than hardcoded: with thinking on this is '<think>\\n', with
    it off '<think>\\n\\n</think>\\n\\n'. Hardcoding '<think>\\n' for both silently produces a
    completion with no closing </think>, which parse.parse_estimate correctly refuses to parse --
    i.e. every no-think rollout would come back as None.
    """
    text = build_prompt_text(tokenizer, "x", enable_thinking=enable_thinking)
    idx = text.rfind(_ASSISTANT_HEADER)
    if idx == -1:
        raise ValueError("chat template no longer emits an <|im_start|>assistant header")
    return text[idx + len(_ASSISTANT_HEADER):]


def load_engine(max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.95):
    from vllm import LLM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    llm = LLM(
        model=MODEL_ID,
        revision=MODEL_REVISION,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        seed=ENGINE_SEED,
        # vLLM 0.32's default (1024) exceeds the ~300 Mamba cache blocks this hybrid model gets
        # at 16k context on an H100, and engine init fails. Scheduling only; sampling unchanged.
        max_num_seqs=256,
        # Text-only task: don't load the vision tower, leaving its memory to the KV cache. At 0.90
        # with it loaded only ~15.7 GiB (~20 concurrent 9k-token traces) was left, which ran at
        # ~280 output tok/s. (limit_mm_per_prompt=0 crashes the offline engine at compile time on
        # vLLM 0.32.) Memory/scheduling only; the language model and sampling are unchanged.
        language_model_only=True,
    )
    return llm, tokenizer


def generate_batch(llm, tokenizer, prompts, n=1, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                    seed=None, enable_thinking=True):
    """prompts: list of (item_id, user_content) tuples. Returns (records, wall_clock_seconds).

    seed=None (default) is required for independent draws across requests — see the module
    docstring note on ENGINE_SEED above. Only pass an explicit seed for a single-request
    reproducibility check, never for a multi-rollout statistical sample.
    """
    from vllm import SamplingParams

    sp = SamplingParams(temperature=temperature, max_tokens=max_tokens, n=n, seed=seed)
    prompt_texts = [build_prompt_text(tokenizer, uc, enable_thinking=enable_thinking)
                    for _, uc in prompts]
    prefill = assistant_prefill(tokenizer, enable_thinking=enable_thinking)
    t0 = time.time()
    outputs = llm.generate(prompt_texts, sp)
    wall_clock = time.time() - t0

    records = []
    for (item_id, user_content), out in zip(prompts, outputs):
        for rollout_idx, completion in enumerate(out.outputs):
            raw_text = prefill + completion.text
            total_tokens = len(completion.token_ids)
            records.append({
                "item_id": item_id,
                "rollout_idx": rollout_idx,
                "prompt": user_content,
                "raw_completion": raw_text,
                "finish_reason": completion.finish_reason,
                "num_tokens": total_tokens,
                "enable_thinking": enable_thinking,
            })
    return records, wall_clock


def append_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def write_config(run_dir, extra):
    import subprocess
    import vllm as _vllm

    # The pod has no git checkout; the driver is launched with GIT_SHA set from the local repo.
    git_sha = os.environ.get("GIT_SHA") or subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=True, text=True).stdout.strip()
    cfg = {
        "git_sha": git_sha,
        "vllm_version": _vllm.__version__,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "reasoning_effort": REASONING_EFFORT,
        "sampling": {"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "seed": None,
                     "engine_init_seed": ENGINE_SEED},
        "wall_clock_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    cfg.update(extra)
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(run_dir) / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
