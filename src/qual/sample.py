"""vLLM offline batch generation for the Donation Bet giraffes task.

Writes append-only raw JSONL to runs/<stamp>/raw/<condition>.jsonl. Never overwrites or
filters in place — parsing/filtering happens downstream in parse.py.
"""
import json
import os
import time
from pathlib import Path

MODEL_ID = "Qwen/Qwen3.5-27B"
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
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
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


def load_engine(max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.90):
    from vllm import LLM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        seed=ENGINE_SEED,
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

    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__),
                              capture_output=True, text=True).stdout.strip()
    cfg = {
        "git_sha": git_sha,
        "vllm_version": _vllm.__version__,
        "model_id": MODEL_ID,
        "sampling": {"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "seed": None,
                     "engine_init_seed": ENGINE_SEED},
        "wall_clock_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    cfg.update(extra)
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(run_dir) / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
