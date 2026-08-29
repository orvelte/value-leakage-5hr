# Results — Donation Bet value leakage in Qwen3.5-27B

Model `Qwen/Qwen3.5-27B @ fc05daec18b0a78c049392ed2e771dde82bdf654`. Task: Donation Bet, giraffes question only. Threshold **75,000,000**, re-derived from this model's own baseline median. Temperature 1, `seed=None` per request. Regenerate with `runs/collect_results.py`.

## The one-sentence answer

> Value leakage in Qwen3.5-27B is **not well described as unfaithful CoT**. The answer is already biased before any reasoning happens — the effect survives with the chain of thought switched off entirely — and what the reasoning adds on top is biased *step selection*, not biased step content: the model keeps searching while its estimate is on the unfavoured side and stops once it reaches the favoured one. No individual sentence needs to be false for the number to come out wrong.

## Headline numbers

| quantity | value |
|---|---|
| bias, thinking ON (n=39/40) | **+0.420**  [+0.220, +0.622] |
| bias, thinking OFF (n=148/146) | **+0.517**  [+0.415, +0.612] |
| bias, user-picks-charity framing | **+0.096**  [+0.001, +0.184] |
| bias among judge-labelled deniers only | **+0.220**  [-0.045, +0.504] |
| covert share of the bias (paper Fig. 6 metric, majority labels) | **12.6%** |
| disclosure split (majority of 3 judge passes) | **34/80 Admits**, 46/80 Denies, 0 hedging |

## Findings

### 1. The bias replicates, and does not need a chain of thought

Thinking on: **+0.420**  [+0.220, +0.622]. Thinking off: **+0.517**  [+0.415, +0.612] — larger, if anything. `enable_thinking=False` is a prefill (the template supplies an empty closed `<think>` block), so this is the same class of manipulation as the forced-denial prefill and carries the same off-distribution caveat.

**Kills:** H1 and H2 as *necessary* mechanisms. An explicit goal step and an asymmetric revision rule both need a reasoning trace to operate in, and there isn't one.

### 2. Naming a threshold anchors the estimate ~5×, with nothing at stake

| framing | median estimate | × threshold |
|---|---|---|
| baseline (no bet) | 422,750,000 | 5.64 |
| threshold_only | 83,100,000 | 1.11 |
| coin | 75,000,001 | 1.00 |
| userpick_above | 82,500,000 | 1.10 |
| above_good | 84,000,000 | 1.12 |
| below_good | 68,000,000 | 0.91 |

Anchoring pulls both directions toward the same number, so it **cancels in the bias metric by design** — it cannot produce a directional effect. Measurable only in the thinking-off arm, where the baseline sits far from the threshold.

**Kills:** H3's *"values are incidental"*. Confirms H3's anchoring claim.

### 3. Motivated reasoning here is a stopping rule

Controlling for step index, being on the favoured side makes the model **2.11× more likely to stop** at that step (p=0.0053; permutation null p=0.0030). The no-bet baseline shows no asymmetry (OR 1.08, p=0.84). Bad-side revisions move toward the favoured side 0.621 of the time [0.586, 0.655]; baseline nulls straddle 0.5.

Overt vs covert: stopping OR 3.48 vs 2.55, interaction p=0.33 — no *detectable* difference, though underpowered at n=34/46. The revision *direction* does differ (bad-side pull 0.714 [0.654, 0.768] vs 0.577 [0.533, 0.619], non-overlapping).

### 4. Covert rollouts are already on the favoured side at their first estimate

Among good-side rollouts, covert start favoured 0.86 of the time vs 0.61 for overt (signed distance p=0.025; logistic OR 3.72 with revision count contributing nothing). Two routes to the same outcome: covert arrive there, overt work their way over.

### 5. The judge is usable, with caveats

Three passes with fresh item→instance pairings. Test-retest 85.0%, Cohen's κ=0.688. Pass 1 showed instance heterogeneity (χ²=19.3, p=0.007) but pass 2 showed none (p=0.71) — a bad draw, not a broken judge. ~15% of items are genuinely borderline; use `covertness_majority.json` for anything label-conditioned.

## Figures

### f2_thinking_off

![f2_thinking_off](figures/f2_thinking_off.png)

### f3_framing_controls

![f3_framing_controls](figures/f3_framing_controls.png)

### f4_revision_asymmetry

![f4_revision_asymmetry](figures/f4_revision_asymmetry.png)

### f5_covertness

![f5_covertness](figures/f5_covertness.png)

## Standing limitations

- **One estimation question** (giraffes). The paper averages covertness over nine.
- **Prefill is off-distribution**, and that includes the thinking-off arm.
- **Anchoring is thinking-off only** and cannot be measured thinking-on as set up, since there the threshold *is* the baseline median by construction.
- **Covert n is small** (46 rollouts); the deniers-only bias CI includes zero.
- **LLM judges** for both covertness and trajectory extraction. Covertness κ=0.688; trajectory extraction showed no instance effect (Kruskal-Wallis p=0.35).
- A few extracted trajectories contain a spurious `0` from the extractor; negligible against ~2,100 steps but uncleaned.

See `planning/hypotheses.md` for the running evidence log and `results/snippets.md` for raw CoT excerpts.
