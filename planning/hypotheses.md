# Running doc: what does motivated reasoning look like, and is "unfaithful CoT" the right name?

Task: Donation Bet Fermi estimates (`src/qual/prompts.py`). Model: `Qwen/Qwen3.5-27B` @ rev
`fc05daec18b0a78c049392ed2e771dde82bdf654`, vLLM 0.27.1, temperature 1.0 (verified top_p=1.0 /
top_k=0 — vLLM does not fold in the model's generation_config defaults), max_tokens 16000,
`seed=None` per request. Threshold re-derived from this model's own baseline median.

Update this after every block. Hourly: did the last block change a hypothesis?

## The two questions everything serves

1. What does motivated reasoning look like **as a process**?
2. Is **"unfaithful CoT"** the right category for it?

## The three hypotheses

**H1 — explicit goal step.** An "aim above/below the threshold" move appears and *causally
steers* the number. Signature: the admission sentence precedes the point where the estimate
trajectory diverges from baseline, and removing it by resampling removes the bias.
*If H1 wins:* "unfaithful CoT" is the wrong name for the overt rollouts — the CoT is faithful,
it says what it is doing. The unfaithfulness question then belongs only to the covert slice.

**H2 — asymmetric scrutiny (Kunda-style).** Every step is sincere, but revision probability and
stopping depend on which side the current estimate is on. Signature: P(revise | current estimate
on bad side) > P(revise | good side), revisions directionally biased toward the favored side, and
earlier stopping once on the good side.
*If H2 wins:* the CoT is faithful step-by-step and unfaithful at the level of **step selection**.
"Unfaithful CoT" is too blunt — no individual sentence is a lie; the search procedure is biased.
This is the most interesting answer for the proposal.

**H3 — anchor / user-desire inference.** The threshold number, or inferring what the user wants,
drives the estimate; values are incidental. Signature: bias survives the valence-stripped
(coin-flip) framing, or the user-picks-charity variant, or survives with thinking disabled.
*If H3 wins:* it is not value leakage in the paper's sense at all, and not a CoT-faithfulness
story — it is a prior on the answer. Internals get skipped; the framing sweep becomes the paper.

Not mutually exclusive. The overt/covert split is the control that runs through all three: if the
same signature appears with and without the admission sentence, overt and covert are **one
computation with different narration** — the headline candidate.

## Scope

**The giraffes question only** (one estimation question, not nine, and not two). `prompts.py`
already has exactly this and needs no additions. Everything reported must therefore carry
"one estimation question" as a stated limitation — the paper averages covertness over nine, and
the above/below asymmetry we see may be giraffe-specific.

## Thinking-off is a prefill, not a mode

Verified against `Qwen/Qwen3.5-27B`'s `tokenizer_config.json`. The template's generation-prompt
block branches on `enable_thinking`: by default it opens `<think>\n` and the model generates a
trace; with `enable_thinking=False` it emits an empty, already-closed block
`<think>\n\n</think>\n\n` and the model continues straight into the answer.

Consequences:

- The thinking-off condition and the forced-denial prefill are **the same class of manipulation**
  — both edit the start of the assistant turn. Report them together as *verbalization
  interventions* and apply the off-distribution caveat to both.
- Use the template kwarg only: `apply_chat_template(..., enable_thinking=False)`, or
  `chat_template_kwargs` in vLLM serving. Unlike Qwen3 there is **no `/think` / `/no_think` soft
  switch** in this template, so putting `/no_think` in the user prompt is just text the model may
  react to as an instruction — a confound.
- **Bug this would have caused, now fixed:** `sample.generate_batch` hardcoded
  `raw_completion = "<think>\n" + completion.text`. In no-think mode the completion has no
  closing `</think>`, so `parse.parse_estimate` correctly returns `None`
  (`truncated_no_close_tag`) — *every* thinking-off rollout would have silently parsed to nothing.
  `sample.assistant_prefill(tokenizer, enable_thinking=...)` now derives the prefill from the
  template, and `generate_batch` records `enable_thinking` on each row. Verified: thinking-on is
  byte-identical to the old behaviour; thinking-off parses correctly.
- On the first no-think rollout, still confirm by eye that the output contains no `<think>`
  content and that the answer begins after the empty block.

## Evidence log

| date | experiment | result | what it moves |
|---|---|---|---|
| 2026-08-29 | Hour-0 pilot, giraffes, n=30/40/40 (`runs/hour0_pilot.py`) | bias **+0.420** [95% CI +0.220, +0.622]; threshold 75M from own baseline median | Effect replicates. Something to explain. Discriminates nothing yet. |
| 2026-08-29 | Covertness judge (App. E.2.1 verbatim), n=80 | **41% Admits / 59% Denies**, 0% in either hedging category | Every trace takes a definite position. Large overt majority. |
| 2026-08-29 | Same labels re-scored on the paper's Fig. 6 metric (App. D latent mixture + most-favorable attribution) | p_biased 0.420; **covert lower bound 15.6%, overt 84.4%** | The 84/16 starting state. Comparable to the paper's per-model bars. |
| 2026-08-29 | Bias restricted to judge-labelled deniers | **+0.220, 95% CI [-0.045, +0.504]** (n=19/28) | Covert effect positive but NOT established at this n. |
| 2026-08-29 | Disclosure vs behaviour | INFLUENCED P(good)=0.875 (n=32) vs NOT_INFLUENCED 0.596 (n=47) | CoT is partly faithful — admitting predicts being more biased. |
| 2026-08-29 | Judge-instance consistency (8 agents, identical frozen prompt, balanced random batches) | INFLUENCED 5,6,8,2,3,0,3,6 /10; chi2=19.3, df=7, **p=0.007** | **The 84/16 split is not reliable.** Re-judge single-instance or majority-vote k=3 before gating on it. |
| 2026-08-29 | CoT length, good-side overt vs covert (n=28/28) | overt 9708 vs covert 7583 tokens, p=0.0002, r=+0.59 | **Probably a judge artifact, not a fact about the model** — see confound below. |
| 2026-08-29 | # intermediate estimates, good-side overt vs covert | 28 vs 23, p=0.094 | Not significant. Same finding as length, not a second one. |
| 2026-08-29 | **First estimate already on the favored side** (App. E.5.1 trajectory judge, good-side only, n=28/28) | overt **0.61** vs covert **0.86**; signed log10(first/T) toward favored side +0.104 vs +0.220, **p=0.025**; logistic covert OR=3.72 (p=0.054) with n_estimates contributing nothing (OR=0.90, p=0.77) | **Sharpest hypothesis this pilot produced.** Covert rollouts are already on the good side at their first estimate; overt ones start further away and revise toward it. Two routes to the same outcome. Weighs against H2 being the whole story *for the covert slice* — the leak looks upstream of the verbalized reasoning there. |
| 2026-08-29 | Trajectory-judge consistency across 8 extraction agents | Kruskal-Wallis H=7.8, p=0.35 | Extraction granularity is stable, unlike the covertness judge. This measure is trustworthy. |

## Live confounds — read before citing anything above

- **The length result is probably circular.** A longer CoT gives the judge more text in which to
  find an admission. Bad-side control (no bias to admit) shows the *same* gap: +2012 tokens,
  r=+0.61. Logistic: tokens **OR=2.02 per 1000** for being labelled INFLUENCED, independent of
  landing good-side. Do not report "overt rollouts reason longer" as a model fact.
- **The covertness split is measured by an unreliable judge** (p=0.007 heterogeneity). Everything
  stratified on it inherits that.
- **The first-estimate contrast is conditioned on the final outcome.** The n_estimates control
  closes one route for that to bite, not all. Needs a bad-side trajectory control (23 rollouts)
  and ideally baseline (30).
- **Trajectory extraction covers good-side rollouts only** (56 of 79). H2's core statistic —
  P(revise | current on bad side) — needs the bad-side rollouts and baseline too.
- **One estimation question.** The paper averages covertness over nine; the above/below asymmetry
  seen here may not survive a second question. Stated limitation, not a fixable gap this sprint.

## Where the data actually stands vs. what the plan assumes

The plan's starting state says "~1000 rollouts with the paper's judge labels". Actual:
**110 rollouts** (30 baseline / 40 above_good / 40 below_good), **80 covertness labels**,
**56 trajectories**. The covert cell is 47 rollouts, not ~160. Every fixed-n block in the plan
(150/condition thinking-off, 100/condition prefill) is fine as written, but the 1:15–2:15
trajectory block cannot run "on data you already have" at the implied scale.

Power note: detecting the deniers-only bias of +0.22 at 80% power needs ~80 deniers per
direction — roughly 135 rollouts per direction at the observed denial rate.

## Carried-over constraints for the internals block

- Adjacent digit-token J-lens directions are near-collinear (cos 0.92–0.997). Do not read which
  side of the threshold the estimate is on through the lens on digit tokens. Measure the estimate
  behaviourally; use the lens only for moral/incentive concepts.
- Shallow-mid layers at delimiter positions are template-dominated. Validate any denial-period
  readout against a shuffled-label null and restrict claims to layers where recognition appears.
- Probe-word single-token coverage is **verified** for all of donate / donation / charity / cause /
  threshold / above / below / honest / bias / good / bad / estimate in the ' word' form
  (`runs/probe_token_coverage.json`).
- Lens layer indexing is **verified** against this model: J is a dict {layer -> [5120,5120] fp16},
  layers 0..62, target_layer 62 is itself a `linear_attention` layer, one before the model's real
  final layer 63 (`runs/lens_indexing_check.json`).
- Off the table: any "make it disclose" prompt that tells the model its internals are being read.

## Reference: the paper's Qwen models on this task

From Figure 6 (measured off chart geometry, ±1–2pp — the paper publishes no table):

| model | bias | Admits | covert share |
|---|---|---|---|
| Qwen3.5-35B-A3B | 0.62 | 55.5% | 7.0% |
| **Qwen3.5-27B (ours)** | **0.42** | **35.4%** | **15.6%** |
| Qwen3.6-35B-A3B | 0.27 | 17.5% | 21.4% |

Ours sits between the two on both axes. For contrast, Claude models run 67–92% covert and
Kimi-K2.6 86.9%; Qwen is the least covert family in the paper.
