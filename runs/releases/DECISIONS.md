# Decisions and deviations: releases

| # | date | decision | why |
|---|---|---|---|
| 1 | 2026-10-28 | Thinking-off cells use the model's **thinking-on** baseline median as T, not a thinking-off median. The 150 thinking-off baselines give the distribution and anchoring only. | The original +0.52 used T = 75M from the thinking-on baseline (`runs/thinking_off.py`), while the thinking-off median is ~422M. A thinking-off threshold would make G-anchor not like for like, and "within 25% of T" would mostly measure baseline mass near its own median. Agreed with Olivia before generation. |
| 2 | 2026-10-28 | Added a thinking-on **turns baseline (n=30)** per model. | The turns cells need a threshold; the original derived 1600 this way (`a5_second_question.py`). The spec's cell table omitted it. |
| 3 | 2026-10-28 | 3.8 runs at `reasoning_effort=xhigh` (its template default, set explicitly). 3.5 and 3.6 run with no effort setting. | Only the 3.8 template reads `reasoning_effort` (verified on the pinned revisions). At xhigh it adds a system message the others don't get. This affects thinking-on cells only; the primary bias test is thinking off. |
| 4 | 2026-10-28 | Covertness and trajectory judges are Claude subagents with the verbatim prompts, as in the original. | Same method as the 3.5 numbers. OpenRouter stays available as a fallback. |
| 5 | 2026-10-28 | vLLM 0.32.0 (pod image) on H100, against the original's 0.27.1 on A100. | Current pod image; same dtype and sampling settings. G-anchor is the check. |
| 6 | 2026-10-28 | The threshold is written into the prompt as a Python float (`75000000.0`), as in the original. | Keeps prompts comparable with the 3.5 run. |
| 7 | 2026-10-28 | Ran `generate.py` and the original `runs/*.py` analysis scripts on the per-model dirs instead of copying them. Only `sample.py` is copied to `lib/` (model selection and effort). | Spec: reuse `src/qual`; copy only modules that need changes. |
