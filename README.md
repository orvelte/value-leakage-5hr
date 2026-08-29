# Value Leakage — Donation Bet on Qwen3.5-27B

Investigating the Donation Bet bias from Betley et al., *Value Leakage* (`papers/2607.14345v4.pdf`,
§3): a model asked to Fermi-estimate a quantity shifts its estimate toward whichever side of a
threshold a stated bet favours. Two questions drive the work — **what does motivated reasoning look
like as a process**, and **is "unfaithful CoT" the right category for it?**

The three live hypotheses (H1 explicit goal step / H2 asymmetric scrutiny / H3 anchor or
user-desire inference), the evidence log, and the open confounds are in
**`planning/hypotheses.md` — read that first** and update it after every block. The current
sprint plan is `New Plan.md`.

## Where the data stands

**Hour-0 pilot** (`runs/hour0/`, thinking on): 110 rollouts, threshold **75M** re-derived from
this model's own baseline median. Bias **+0.420** [+0.220, +0.622]. Covertness judged over three
passes and majority-voted: **42.5% Admits / 57.5% Denies**, covert share **12.6%** on the paper's
Fig. 6 metric (`covertness_majority.json` — use these labels, not pass 1's). Test-retest
kappa = 0.688. Trajectory extraction on the 56 good-side rollouts.

**Thinking-off** (`runs/thinking_off/`): 450 rollouts. Bias **+0.517** [+0.415, +0.612] — the
effect survives with no CoT at all, and is if anything larger. The no-think baseline median is
**422M** against the 75M threshold.

**Framing controls** (`runs/framing_controls/`): 600 rollouts, thinking off. Naming a threshold
with **no stake whatsoever** collapses the median estimate from 422M to 83M (coin variant:
75,000,001). Anchoring is large but orthogonal to the bias metric, which the symmetric
two-direction design makes anchor-proof. The user-picks-charity framing yields only **+0.096**
[+0.001, +0.184] against +0.517 for the moral framing.

Reading: a values-driven prior on the answer that the CoT narrates rather than produces. Neither
an explicit goal step nor an asymmetric revision rule can be necessary, since the bias needs no
CoT; and "values are incidental" is false, since anchoring cannot produce a directional effect
and sycophancy explains only ~19% of it.

Live caveats: one estimation question only; the anchoring result is thinking-off only and cannot
be measured thinking-on as set up; trajectory coverage is good-side only.

## Results

Everything lands in `results/`, regenerated from the run JSONs so it cannot drift from the data:

- **`results/RESULTS.md`** — the report, structured by finding, with figures embedded. Start here.
- `results/results.json` — the same numbers, machine-readable.
- `results/figures/` — one figure per finding, shared style from `src/qual/figures.py`.
- `results/snippets.md` — CoT excerpts with full provenance, curated and random kept separate.

**Verification.** `runs/verify_headline.py` recomputes every headline number from raw JSONL
while importing nothing from `src/qual` — its own parser, bootstrap, logistic and AUC. Run it
before quoting a number anywhere. Currently 14/14 reproduce.

```bash
source env.sh
python3 runs/verify_headline.py
python3 runs/analyze_covertness.py --run hour0
python3 runs/analyze_revision.py   --run hour0
python3 runs/make_figures.py
python3 runs/collect_snippets.py
python3 runs/collect_results.py
```

## Environment

`source env.sh` before any GPU work — this pod's r550 driver needs the CUDA-13 forward-compat
libs that vLLM's torch build requires, and it points HF_HOME at the `/workspace` volume. See
`requirements.txt`. vLLM and HF transformers both want the whole GPU; don't run them concurrently.

Model: `Qwen/Qwen3.5-27B` @ `fc05daec18b0a78c049392ed2e771dde82bdf654`.
Lens artifacts: `camilablank/workspace-lenses`, path `qwen3.5-27b/{j-lens,r-lens}/lens.pt`.

## Library — `src/qual/`

- `prompts.py` — Donation Bet templates (baseline / above_good / below_good), byte-verified
  against the paper. Giraffes question only, by design — the plan is scoped to one question.
- `sample.py` — vLLM batch generation, with `enable_thinking` support. Thinking-off is a
  **prefill**, not a mode: the template emits an empty closed `<think>\n\n</think>\n\n` block.
  Reconstruct completions with `assistant_prefill()`, never a hardcoded `"<think>\n"`, or every
  no-think rollout parses to `None`. **Read the ENGINE_SEED comment before running anything**:
  passing an explicit per-request seed correlates sampling across concurrent rollouts (up to 43%
  byte-identical completions in a prior round). Always `seed=None`; verify with
  `stats.effective_sample_size_row`'s dup rate.
- `parse.py` — final-estimate extraction and the `[T/10, 10T]` outlier filter.
- `segment.py` — CoT → sentences, the unit resampling and the locator operate on.
- `resample.py` — prefix-forcing continuation engine (truncate at sentence k, regenerate). The
  prefill/forced-denial block runs on this.
- `judge.py` — the two paper judges used verbatim (App. E.2.1 covertness, App. E.5.1 trajectory),
  plus number-blurring so the covertness judge cannot infer bias from where the number landed.
- `lens.py` — J-lens/R-lens download, load, readout, and the layer-indexing cross-check. `J` is a
  dict `{layer -> [5120,5120] fp16}`, **not** a stacked tensor; `target_layer=62` is itself a
  `linear_attention` layer, one before the model's real final layer 63.
- `stats.py` — bias bootstrap CI, the paper's latent-mixture covertness quantification (App. D +
  the Fig. 6 favorable attribution), AUC, KS, and a dispersion test for judge-fan-out consistency.

`src/experiments/locate_disclaimer.py` finds the sentence index stating a denial or admission —
needed for H1's "does the admission precede divergence" test and for resampling from just before
it. It has no caller yet; that is by design, not rot.

## Drivers — `runs/`

| script | what it does |
|---|---|
| `check_setup.py` | pre-flight: CUDA forward-compat, chat template opens `<think>`, probe-word tokenisation, lens layer indexing |
| `hour0_pilot.py` | generation: baseline → threshold → both bet directions, writes covertness judge inputs |
| `analyze_covertness.py` | tallies judge outputs; reports both the paper's Fig. 6 lower bound and the deniers-only bias, plus the judge-consistency test |
| `prep_trajectory.py` | writes App. E.5.1 trajectory-judge inputs and batch lists |
| `analyze_trajectory.py` | covert vs overt on CoT length, revision count, and where the first estimate fell, with the confound controls |
| `show_cot.py` | read or export full transcripts with estimate + judge verdict attached |
| `plot_bias.py` | the bias-replication figure (`runs/figs/`) |

The judge steps are Claude-subagent fan-outs run outside these scripts; the scripts prepare inputs
and parse outputs. Batch assignments are persisted under `runs/<run>/{judge,trajectory}_batches/`
so the consistency check stays reproducible.

Raw generation JSONL and judge *inputs* are gitignored — both are regenerable. Judge and
trajectory *outputs* are kept: they cost real money to recreate.
