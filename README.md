# Value Leakage — Donation Bet mechanism (fresh round)

Investigates the Donation Bet bias from Betley et al., *Value Leakage* (`papers/2607.14345v4.pdf`,
§3): a model asked to Fermi-estimate a quantity shifts its estimate toward whichever side of a
threshold a stated bet favors — sometimes while its own CoT states it is being unbiased. This
round's plan and the three candidate mechanisms (concealment / absent computation / selection
effect) are in `planning/hypotheses.md` — read that first, and keep it updated after each
experiment.

Background papers in `papers/`: the Value Leakage paper itself, Thought Anchors (sentence-level
CoT resampling methodology), and the J-lens / global-workspace paper (verbalizable-representation
readout methodology).

## What's here

- `src/qual/prompts.py` — the Donation Bet prompt templates (baseline / above_good / below_good),
  byte-verified against the paper.
- `src/qual/sample.py` — vLLM batch generation. **Read the ENGINE_SEED comment before running
  anything** — the prior round found that passing an explicit seed to every request correlates
  sampling across concurrent rollouts (up to 43% byte-identical completions). Always `seed=None`
  per request; verify with `stats.effective_sample_size_row`'s dup-rate.
- `src/qual/parse.py` — numeric-estimate extraction from a raw completion + the outlier filter.
- `src/qual/segment.py` — CoT → sentence list (the unit resampling/locating operates on).
- `src/qual/resample.py` — prefix-forcing continuation engine: truncate a CoT at sentence k,
  regenerate the rest, many times.
- `src/qual/judge.py` — the two paper judges used verbatim (App. E.2.1 covertness, App. E.5.1
  trajectory extraction), plus number-blurring so judges can't infer bias from where a number
  landed.
- `src/experiments/locate_disclaimer.py` — finds the sentence index where a CoT states its
  denial/admission, for judge-classified or unclassified text.
- `src/qual/lens.py` — J-lens/R-lens download + load + full-vocab readout, with the Qwen3.5-27B
  layer-indexing cross-check (`assert_layer_indexing`) already verified: the lens's
  `target_layer=62` is itself a `linear_attention` layer, one before the model's real final
  layer 63 — don't assume naive `model.layers[i]` correspondence on a different model.
- `src/qual/stats.py` — bootstrap CI for the bias metric, AUC, KS test, power calc.

No experiment-runner scripts yet — this is infra only, ported from a prior round of the same
project. Model/task-specific driver scripts (generation runs, the resampling sweep, the J-lens
concept-probe reads, ablation/steering) still need to be written against this round's actual
scope.

## Environment

GPU work (`sample.py`'s vLLM generation, `lens.py`'s forward passes) needs a CUDA machine — see
`requirements.txt`. vLLM and HF transformers both want the whole GPU; don't run them
concurrently.

Model: `Qwen/Qwen3.5-27B` (pin the exact revision once chosen and record it in each run's
`config.json`, via `sample.write_config`). Lens artifacts: `camilablank/workspace-lenses` on HF,
path `{model_dir}/{j-lens,r-lens}/lens.pt`.
