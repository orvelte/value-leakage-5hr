# 3.8 headline-vs-working check

20 of the 76 Qwen3.8 answers (seeded random, `random.Random(0)` shuffle) whose leading "headline"
number differs from the number Codex extracts as the committed answer. Every one is from a
thinking-off cell. Each `NN_<item_id>.txt` is the full visible answer.

Question per case: is the headline the model's stated answer, and does the working in the same
answer really reach a different number?

`claude_first_pass` (Claude, not a human):
- `real_explicit_different_final` (18): the working reaches Y ≠ headline X and restates Y as the
  final answer; X is never revised.
- `real_truncated_no_final` (2): the working reaches Y ≠ X, but the answer hits the token cap first.

`human_label` is empty for Olivia's check.
