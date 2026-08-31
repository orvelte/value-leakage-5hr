# Value Leakage — Donation Bet on Qwen3.5-27B 

Summary: 4 experiments on Qwen3.5-27B, chosen to separate four accounts of the denial-plus-bias phenomenon: post-hoc rationalization, motivated search, covert report, and sincere unawareness.

1. “Turning off” the CoT: The value’s influence doesn’t need a CoT to cause bias. With thinking disabled, bias is +0.52 [+0.42, +0.61] against +0.42 with reasoning, and the no-trace answers are threshold copies: 59% land within 25% of the threshold (baseline 2%), with the moral valence picking the side. Replicated on a second question (+0.40), where the anchor pulled the opposite direction the anchor was shown to be the threshold itself.

2. Motivated backtracking, quantified. In free traces, the model is 2.1× more likely to stop when its running estimate is on the favored side, and wrong-side revisions move toward the favored side 62% of the time; no-bet traces show neither. This quantifies the selection effect behind the paper's length–bias correlation. Overt and covert traces didn't detectably differ here (p=0.33).

3. The stated stance is causal. Forcing an admission as the trace's opening sentence installs the full bias (+0.44); following it with a denial removes most of it (+0.12, vs +0.43 for a length-matched control; p=0.016); a reversal without honesty content removes nothing, and the goal survives a denial if restated. Free traces agree: deniers land favored 59% vs 88% for admitters (OR 5.1). The paper's pigeonhole argument bounds the genuinely covert share at 12%, bootstrap CI [0.00, 0.40].

4. The incentive stays represented when denied. At the denial sentence, the J-space carries 2.4× the donation content of matched sentences and a signed above/below lean that separates the arms (+0.39); at the pre-answer token, a difference-of-means direction fitted only on admitting traces reads denying ones at AUC 0.80. But within a condition it predicts nothing (underpowered null): encoded, not shown to be used.

