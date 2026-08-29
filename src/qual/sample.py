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


def build_prompt_text(tokenizer, user_content):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )


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
                    seed=None):
    """prompts: list of (item_id, user_content) tuples. Returns (records, wall_clock_seconds).

    seed=None (default) is required for independent draws across requests — see the module
    docstring note on ENGINE_SEED above. Only pass an explicit seed for a single-request
    reproducibility check, never for a multi-rollout statistical sample.
    """
    from vllm import SamplingParams

    sp = SamplingParams(temperature=temperature, max_tokens=max_tokens, n=n, seed=seed)
    prompt_texts = [build_prompt_text(tokenizer, uc) for _, uc in prompts]
    t0 = time.time()
    outputs = llm.generate(prompt_texts, sp)
    wall_clock = time.time() - t0

    records = []
    for (item_id, user_content), out in zip(prompts, outputs):
        for rollout_idx, completion in enumerate(out.outputs):
            raw_text = "<think>\n" + completion.text
            total_tokens = len(completion.token_ids)
            records.append({
                "item_id": item_id,
                "rollout_idx": rollout_idx,
                "prompt": user_content,
                "raw_completion": raw_text,
                "finish_reason": completion.finish_reason,
                "num_tokens": total_tokens,
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
