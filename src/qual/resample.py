"""Prefix-forcing continuation — the CoT-level resampling engine.

Truncates a CoT at a given sentence index and regenerates the continuation from that exact
prefix, many times, to measure how P(good side) changes as a function of the truncation point
(the "where does the leak happen" sweep), and to force in a specific sentence (e.g. a denial, or
its opposite) and measure whether that causally moves the final distribution.
"""
from src.qual.sample import assistant_prefill, build_prompt_text, MAX_TOKENS, TEMPERATURE
from src.qual.segment import segment_sentences


def build_forced_prefix(cot_text, sentence_idx):
    """Returns the prefix string "<think>\\n<sentence_0> <sentence_1> ... <sentence_{k-1}> "
    to force-feed as the start of the assistant turn. sentence_idx=0 means an EMPTY CoT
    prefix (just "<think>\\n"), i.e. truncate-at-sentence-0 for the k=0 point of a sweep."""
    sentences = segment_sentences(cot_text)
    prefix_sentences = sentences[:sentence_idx]
    body = " ".join(prefix_sentences)
    if body:
        return "<think>\n" + body + " "
    return "<think>\n"


def generate_forced_continuations(llm, tokenizer, items, sentence_idx, temperature=TEMPERATURE,
                                   max_tokens=MAX_TOKENS, seed=None):
    """items: list of (item_id, user_content, raw_cot_to_truncate). Returns raw records in the
    same shape as sample.generate_batch, plus the forced prefix used."""
    from vllm import SamplingParams

    sp = SamplingParams(temperature=temperature, max_tokens=max_tokens, n=1, seed=seed)
    prompt_texts = []
    prefixes = []
    for item_id, user_content, raw_cot in items:
        base_prompt = build_prompt_text(tokenizer, user_content)
        # base_prompt ends in the template's own assistant prefill ("<think>\n" with thinking
        # on) — strip it so build_forced_prefix's own "<think>\n" isn't duplicated. Resampling
        # is inherently a thinking-on operation: we are forcing a CoT prefix.
        prefill = assistant_prefill(tokenizer, enable_thinking=True)
        assert base_prompt.endswith(prefill), "chat template no longer auto-opens <think>"
        base_prompt_no_think = base_prompt[: -len(prefill)]
        forced_prefix = build_forced_prefix(raw_cot, sentence_idx)
        prompt_texts.append(base_prompt_no_think + forced_prefix)
        prefixes.append(forced_prefix)

    outputs = llm.generate(prompt_texts, sp)
    records = []
    for (item_id, user_content, _), forced_prefix, out in zip(items, prefixes, outputs):
        completion = out.outputs[0]
        raw_text = forced_prefix + completion.text
        records.append({
            "item_id": item_id,
            "prompt": user_content,
            "sentence_idx": sentence_idx,
            "forced_prefix": forced_prefix,
            "raw_completion": raw_text,
            "finish_reason": completion.finish_reason,
            "num_tokens": len(completion.token_ids),
        })
    return records


def generate_forced_continuations_per_item(llm, tokenizer, items, n_continuations,
                                            temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                                            seed=None):
    """Like generate_forced_continuations, but each item carries its OWN forced prefix (already
    built via build_forced_prefix / build_forced_prefix_with_injection at the caller's chosen
    per-item truncation point), and each item gets `n_continuations` independent samples
    (SamplingParams n=..., seed=None) in one request per item rather than one request per
    sample.

    items: list of dicts with 'item_id', 'user_content', 'forced_prefix'.
    Returns: dict item_id -> list of records (same shape as generate_forced_continuations,
    minus 'sentence_idx' since it's per-item; carried via the input dict instead).
    """
    from vllm import SamplingParams

    sp = SamplingParams(temperature=temperature, max_tokens=max_tokens, n=n_continuations, seed=seed)
    prompt_texts = []
    for item in items:
        base_prompt = build_prompt_text(tokenizer, item["user_content"])
        prefill = assistant_prefill(tokenizer, enable_thinking=True)
        assert base_prompt.endswith(prefill), "chat template no longer auto-opens <think>"
        base_prompt_no_think = base_prompt[: -len(prefill)]
        prompt_texts.append(base_prompt_no_think + item["forced_prefix"])

    outputs = llm.generate(prompt_texts, sp)
    results = {}
    for item, out in zip(items, outputs):
        records = []
        for completion in out.outputs:
            raw_text = item["forced_prefix"] + completion.text
            records.append({
                "item_id": item["item_id"],
                "prompt": item["user_content"],
                "forced_prefix": item["forced_prefix"],
                "raw_completion": raw_text,
                "finish_reason": completion.finish_reason,
                "num_tokens": len(completion.token_ids),
            })
        results[item["item_id"]] = records
    return results
