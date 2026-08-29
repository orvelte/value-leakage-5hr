# Running doc: three mechanisms for "CoT denies the bias"

Per the spec: update this after every experiment. Task is the Donation Bet giraffes Fermi
estimate (`src/qual/prompts.py`), model TBD at hour 0 (see below), threshold re-derived from
that model's own baseline median (not copied from any other model's threshold).

## Hour 0 check: does this model even show the covert variant?

Before anything else: does the chosen model deny the bias in raw CoT (covert), or state it
outright ("moral implication: aim above X") like Qwen3.6-35B-A3B does in the paper (overt)? If
it only reproduces the overt variant, that redirects the whole project — better to learn this
in minute 20 than hour 3.

- Model:
- n rollouts / condition:
- Covertness judge result (Admits / Mentions / No mention / Denies):
- Verdict: covert / overt / mixed —

## The three hypotheses

**A. Concealment** — the model represents the incentive and suppresses it in the CoT. The
denial is not sincere; the influence is present and causally live, just not verbalized.

**B. Absent computation (arguably not "values" at all)** — the incentive enters through a
channel the CoT never verbalizes because it isn't a values-mediated decision at all, e.g. the
threshold number simply anchors the Fermi estimate regardless of moral valence. This is the
hypothesis the neutral-framing prompt control (strip moral valence, keep the number) is
designed to kill.

**C. Selection effect, CoT honest throughout** — each step is honest, but the model keeps
refining when its estimate is on the bad side and stops when it's on the good side. Bias
emerges from a stopping rule, not from any single dishonest step. The paper itself flags this
when discussing the reasoning-length correlation.

## Evidence log

| date | experiment | result | what it moves |
|---|---|---|---|
| | | | |

## What would distinguish them

- **First estimate already differs between conditions** → leak is upstream of verbalized
  reasoning; denial describes an already-biased process (weighs toward A or B, against C being
  the *whole* story).
- **Only final estimates differ, first estimates match** → late-stage move; localizes where to
  intervene (resampling sweep should show a jump point).
- **Finals differ mainly through CoT length / stopping point** → selection effect (C) is live.
- **Bias survives the neutral-framing (coin-flip, no moral valence) prompt control** → numeric
  anchoring (B), not values — the whole "value leakage" framing would be a red herring for this
  model on this task.
- **Forcing the denial sentence into a truncated prefix doesn't move the final distribution** →
  the denial is epiphenomenal (consistent with A or C, since the verbalized sentence isn't
  doing causal work either way) rather than a real concealment act with causal weight.
- **J-lens shows the incentive direction (above/below) lit up at the denial sentence itself,
  paired-condition** → the content is in the workspace at the moment of denial — argues against
  a strong reading of B (nothing there to conceal) and is consistent with A's "represents but
  suppresses" story, though on its own doesn't distinguish A from a version of C where the
  representation exists but isn't what the stopping rule uses.
- **Ablating/steering the J-lens/DoM incentive direction reduces bias while baseline estimates
  stay unchanged** → the direction is not just correlated but causally mediates the bias —
  answers "can you turn the effect off."

## Known constraints carried over from prior lens work

- Adjacent digit-token J-lens directions are near-collinear (cos 0.92-0.997) — do not try to
  read which side of the threshold the estimate is on via the lens on digit tokens. Measure the
  estimate behaviorally; use the lens only for moral/incentive concepts (donate, charity,
  cause, threshold, above, below, honest, bias).
- Shallow-mid layers at delimiter positions tend to be template-dominated — validate any
  denial-period readout against a shuffled-label null and restrict claims to layers where
  recognition actually showed up, rather than reporting a raw top-k list.
- Check single-token coverage for each probe word in the Qwen tokenizer before assuming a
  vector exists for it; some of these concepts may be multi-token.
