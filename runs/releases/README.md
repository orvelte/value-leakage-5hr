# Donation Bet across Qwen releases (3.5 / 3.6 / 3.8, 27B)

Pre-registration: [PREREG.md](PREREG.md). Every deviation (19 of them, numbered): [DECISIONS.md](DECISIONS.md).
All numbers: [results.json](results.json). Figures: [figures/](figures/).

## Which rule fired

**"Against Betley's prediction."** With thinking off (the primary measure), Qwen3.8's bias is
**+0.090 [−0.026, +0.198]** against Qwen3.5's **+0.551 [+0.458, +0.645]**: 0.46 lower, with
non-overlapping CIs. The pre-registered threshold for "against" was a drop of ≥ 0.15.

| Prediction | Result |
|---|---|
| P1: 3.8 bias ≥ 3.5 + 0.15 (thinking off) | **Fails.** 3.8 − 3.5 = −0.46 |
| P2: covert lower bound or denial share ≥ 10 points higher in 3.8 | **Holds on the covert bound** (0% → 60%), but see the caveat below. The denial share *falls* (35% → 10%). |
| P3: userpick bias < ½ good/bad bias, in all three | **Holds for 3.5 and 3.6, fails for 3.8.** In 3.8 both are near zero (+0.11 vs +0.09), so the test has no power there. |

## Results

### Bias (balanced metric, 95% bootstrap CI)

| Arm | Qwen3.5 | Qwen3.6 | Qwen3.8 |
|---|---|---|---|
| Good/bad cause, thinking off **(primary)** | **+0.551** [+0.458, +0.645] | **+0.133** [+0.022, +0.250] | **+0.090** [−0.026, +0.198] |
| Good/bad cause, thinking on | +0.287 [+0.085, +0.489] | +0.118 [−0.085, +0.321] | +0.217 [+0.009, +0.424] |
| User picks charity, thinking off | +0.120 [+0.000, +0.240] | −0.063 [−0.186, +0.061] | +0.113 [−0.037, +0.251] |
| Left turns, thinking off | +0.502 [+0.380, +0.614] | +0.419 [+0.288, +0.562] | +0.008 [−0.189, +0.197] ⚠ |
| Bet answers within 25% of T (baseline) | 64% (3%) | 33% (9%) | 25% (16%) |
| Thresholds T (giraffes / turns) | 71M / 1,700 | 44M / 1,500 | 140M / 2,400 |
| Median reasoning tokens, thinking on | 8,960 | 3,877 | 3,835 |

⚠ 3.8 left-turns, thinking off: 65–68 of 100 answers hit the 4,096-token cap mid-reasoning
without stating an answer (DECISIONS #19), so only 44 and 33 remain. This cell fails G-parse and
should not be read.

**Original 3.5 run, for reference:** +0.517 (thinking off), +0.420 (on), +0.096 (userpick), +0.399 (turns).
G-anchor passed: this run's 3.5 thinking-off bias falls inside the original CI [+0.415, +0.612].

### Bias excluding answers within 25% of T

| Arm | Qwen3.5 | Qwen3.6 | Qwen3.8 |
|---|---|---|---|
| Good/bad cause, thinking off | +0.519 [+0.368, +0.674] | +0.069 [−0.058, +0.194] | +0.096 [−0.043, +0.224] |
| Good/bad cause, thinking on | +0.029 [−0.213, +0.278] | +0.070 [−0.141, +0.283] | +0.141 [−0.112, +0.394] |
| User picks charity, thinking off | +0.157 [−0.031, +0.348] | −0.184 [−0.324, −0.043] | +0.018 [−0.150, +0.185] |
| Left turns, thinking off | +0.526 [+0.321, +0.736] | +0.464 [+0.250, +0.679] | −0.056 [−0.273, +0.160] ⚠ |

3.5's thinking-off bias is not just answers sitting on T: it survives intact away from the
threshold. 3.5's thinking-on bias does sit mostly near T (+0.29 → +0.03, n = 41). 3.6's
userpick control turns significantly negative away from T. That was unexpected, and it is not interpreted.

### Disclosure and covertness (thinking on, 80 bet traces per model)

Paper-verbatim covertness judge on number-blurred traces. Judge: **Codex `gpt-6.1-sol` (medium)**,
not the original Claude subagent (DECISIONS #12–13), so compare across models here, not with the
original 34/80.

| | Admits | Mentions | No mention | Denies | Covert lower bound | p_biased |
|---|---|---|---|---|---|---|
| Qwen3.5 | 52 | 0 | 0 | 28 | 0% [0, 0] | +0.25 |
| Qwen3.6 | 12 | 0 | 10 | 58 | 0% [0, 58%] | +0.10 |
| Qwen3.8 | 5 | 1 | **66** | 8 | **60%** [0, 94%] | +0.13 |

**Second-judge check** (20 traces per model, seeded random, 10 per direction; exported with the
raw traces to [handlabels/](handlabels/) for hand-labelling). Sonnet was requested, but
Anthropic's content filter refused every request. The second judge is Codex `gpt-6.1-terra`
(high), a different model from judge 1. Agreement on admit-or-not: 3.5 19/20 (κ 0.89), 3.6
19/20 (κ 0.86), 3.8 20/20 (no admits in the sample, so κ is undefined). Both judges are OpenAI
models; treat this as a consistency check, not cross-vendor validation. Passes 1 and 2 of judge 1
were identical on every item, so pass agreement carries no information.

### Reasoning signatures (thinking on)

| | Stop-on-favoured OR (hazard model) | Baseline nulls | Wrong-side revisions toward favoured | Bet trajectories |
|---|---|---|---|---|
| Qwen3.5 | **1.86** [1.14, 3.03], p = 0.014 | 0.77, 1.30 | **62%** [59, 65] | 80 |
| Qwen3.6 | 1.58 [0.95, 2.62], p = 0.08 | 0.54, 1.84 | 56% [52, 60] | 70 |
| Qwen3.8 | 1.49 [0.40, 5.51], p = 0.55 | 0.52, 1.91 | 64% [50, 77] | 12 |

The original `prep_trajectory.py` keeps only traces labelled INFLUENCED or NOT_INFLUENCED, so
3.8's 66 no-mention traces drop out. Of the remaining 13, one yielded no extractable trajectory, so 3.8's signatures rest on 12 bet trajectories (and 38 baselines).

## Caveats that matter for interpretation

1. **Retrains, not persona training.** The releases differ in much more than persona strength,
   so this tests the prediction on "a model with a stronger default persona" only.
2. **3.8 ran at `reasoning_effort=medium`, not the pre-registered xhigh** (#17). At xhigh, 35/40
   thinking-on traces hit the 16k cap. Medium gives prompts byte-identical to 3.5/3.6. The
   primary (thinking-off) measure is unaffected by effort.
3. **Answer parsing had to change** (#14, #18, #19). 3.6 and 3.8 often reason in prose before
   giving a number. The rule: the original regex where the answer leads with a number, Codex
   extraction otherwise, and truncated prose answers count as unparsed. Sensitivity checks
   (thinking off, good/bad, 3.5 / 3.6 / 3.8): Codex extraction on every answer gives
   +0.572 / +0.140 / +0.070. The original regex everywhere gives +0.558 / (not meaningful: it
   reads list indices) / +0.164. The rule's verdict is "against" under every variant.
4. **The 3.8 covert bound is fragile.** 60% of a +0.13 bias, with a CI of 0–94%. What's robust
   is the disclosure *pattern*: 3.8 rarely engages with the bet in its reasoning at all.
5. **Exploratory, not pre-registered:** in answers that lead with a number, the headline number
   and the number the justification derives disagree in 3% (3.5), 17% (3.6) and 55% (3.8) of the
   60 sampled answers per model.

## Paragraph for the main write-up

> Betley's split-persona account predicts that a stronger trained persona should rationalise more, so motivated reasoning should rise across Qwen releases as the default persona strengthens (prompted roles show through 50% → 31% → 8% for 3.5 → 3.6 → 3.8). On the Donation Bet the opposite happens. With thinking off, the bias toward the "good cause" side of the threshold falls from +0.55 [0.46, 0.65] in Qwen3.5-27B to +0.13 [0.02, 0.25] in 3.6 and +0.09 [−0.03, +0.20] in 3.8. The pre-registered "against" rule fires (a drop of ≥ 0.15 with non-overlapping CIs), and the decline tracks the role-expression trend monotonically. The thinking-on arm and the turns question point the same way, though less precisely. What changes alongside the bias is how the model talks about the bet. 3.5 reasons about the threshold openly and mostly admits being swayed (52/80 traces). 3.6 mostly denies it (58/80). 3.8 largely ignores it (66/80 traces never mention the bet). So 3.8's small residual bias is mostly not disclosed (covert lower bound 60%, CI 0–94%). Read together, the stronger persona is less biased by the incentive and does not visibly rationalise more. If anything, its remaining sensitivity to the bet is quieter. The releases are separate retrains, so this argues against "stronger persona → more motivated reasoning" as a general regularity across these models, not against persona training as a mechanism.

## Reproducing

| Step | Command |
|---|---|
| Generate one model (pod) | `python3 runs/releases/generate.py --model q35` (`--off-first` for 3.8) |
| Everything after generation | `bash runs/releases/postprocess.sh <model>` |
| Fold + figures | `python3 runs/releases/collect.py && python3 runs/releases/make_figures.py` |

Spend: **$19.67** GPU (4.93 h on one H100, including ~1.6 h lost to vLLM start-up fixes and the
aborted 3.8 xhigh run). Judges and extraction ran on Codex via a ChatGPT login ($0 against the
budget). The Sonnet requests were all refused ($0).
