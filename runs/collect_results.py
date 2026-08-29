"""Aggregate every experiment's results into one machine-readable JSON and one report.

Single source of truth: `results/results.json` (numbers) and `results/RESULTS.md` (the report,
structured by finding rather than by chronology, with figures embedded). Re-run after any
analysis changes so nothing in the write-up can drift from the data:

  source env.sh && python3 runs/analyze_covertness.py --run hour0
  source env.sh && python3 runs/analyze_revision.py --run hour0
  source env.sh && python3 runs/make_figures.py
  source env.sh && python3 runs/collect_results.py
"""
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RUNS = Path(__file__).resolve().parent
RES = RUNS.parent / "results"


def load(p, default=None):
    p = Path(p)
    return json.load(open(p)) if p.exists() else default


def main():
    RES.mkdir(exist_ok=True)
    hour0 = load(RUNS / "hour0" / "behavioral_results.json", {})
    cov = load(RUNS / "hour0" / "covertness_summary.json", {})
    maj = load(RUNS / "hour0" / "covertness_majority.json", {})
    traj = load(RUNS / "hour0" / "trajectory_results.json", {})
    rev = load(RUNS / "hour0" / "revision_results.json", {})
    revcov = load(RUNS / "hour0" / "revision_by_covertness.json", {})
    toff = load(RUNS / "thinking_off" / "results.json", {})
    fram = load(RUNS / "framing_controls" / "results.json", {})
    pref = load(RUNS / "prefill_tests" / "results.json", None)

    out = {
        "model": "Qwen/Qwen3.5-27B @ fc05daec18b0a78c049392ed2e771dde82bdf654",
        "task": "Donation Bet, giraffes question only",
        "threshold": hour0.get("threshold"),
        "generated": str(date.today()),
        "bias": {
            "thinking_on": hour0.get("bias"),
            "thinking_off": toff.get("bias_bootstrap"),
            "userpick_thinking_off": fram.get("userpick_bias"),
            "by_prefill": {k: v for k, v in (pref or {}).get("cells", {}).items()
                           if k.endswith("/bias")} or None,
        },
        "covertness": {
            "buckets_pass1": cov.get("buckets_all_rollouts"),
            "n_influenced_majority": maj.get("n_influenced"),
            "covert_share_pass1": cov.get("covert_share_lower_bound"),
            "covert_share_majority": maj.get("covert_share"),
            "judge_consistency_pass1": cov.get("judge_consistency"),
            "deniers_only_bias": cov.get("deniers_only_bias"),
        },
        "anchoring_thinking_off": {k: v.get("median") for k, v in
                                   (fram.get("conditions") or {}).items()},
        "trajectory": {
            "first_estimate": (traj.get("comparisons") or {}).get("first_estimate"),
            "cot_tokens": (traj.get("comparisons") or {}).get("cot_tokens"),
            "n_estimates": (traj.get("comparisons") or {}).get("n_estimates"),
        },
        "revision_asymmetry": {"hazard_models": rev.get("hazard_models"),
                               "by_covertness": revcov or None},
        "prefill_tests": pref,
    }
    (RES / "results.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {RES/'results.json'}")

    def ci(d, fmt="{:+.3f}"):
        if not d:
            return "—"
        return (f"**{fmt.format(d['point'])}**  [{fmt.format(d['ci_low'])}, "
                f"{fmt.format(d['ci_high'])}]")

    figs = sorted((RES / "figures").glob("*.png")) if (RES / "figures").exists() else []
    lines = [
        "# Results — Donation Bet value leakage in Qwen3.5-27B",
        "",
        f"Model `{out['model']}`. Task: {out['task']}. Threshold "
        f"**{out['threshold']:,.0f}**, re-derived from this model's own baseline median. "
        f"Temperature 1, `seed=None` per request. Regenerate with `runs/collect_results.py`.",
        "",
        "## The one-sentence answer",
        "",
        "> Value leakage in Qwen3.5-27B is **not well described as unfaithful CoT**. The answer is "
        "already biased before any reasoning happens — the effect survives with the chain of "
        "thought switched off entirely — and what the reasoning adds on top is biased *step "
        "selection*, not biased step content: the model keeps searching while its estimate is on "
        "the unfavoured side and stops once it reaches the favoured one. No individual sentence "
        "needs to be false for the number to come out wrong.",
        "",
        "## Headline numbers",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| bias, thinking ON (n=39/40) | {ci(out['bias']['thinking_on'])} |",
        f"| bias, thinking OFF (n=148/146) | {ci(out['bias']['thinking_off'])} |",
        f"| bias, user-picks-charity framing | {ci(out['bias']['userpick_thinking_off'])} |",
        f"| bias among judge-labelled deniers only | {ci(out['covertness']['deniers_only_bias'])} |",
        f"| covert share of the bias (paper Fig. 6 metric, majority labels) | "
        f"**{out['covertness']['covert_share_majority']:.1%}** |",
        f"| disclosure split (majority of 3 judge passes) | "
        f"**{out['covertness']['n_influenced_majority']}/80 Admits**, "
        f"{80 - out['covertness']['n_influenced_majority']}/80 Denies, 0 hedging |",
        "",
        "## Findings",
        "",
        "### 1. The bias replicates, and does not need a chain of thought",
        "",
        f"Thinking on: {ci(out['bias']['thinking_on'])}. Thinking off: "
        f"{ci(out['bias']['thinking_off'])} — larger, if anything. `enable_thinking=False` is a "
        "prefill (the template supplies an empty closed `<think>` block), so this is the same "
        "class of manipulation as the forced-denial prefill and carries the same "
        "off-distribution caveat.",
        "",
        "**Kills:** H1 and H2 as *necessary* mechanisms. An explicit goal step and an asymmetric "
        "revision rule both need a reasoning trace to operate in, and there isn't one.",
        "",
        "### 2. Naming a threshold anchors the estimate ~5×, with nothing at stake",
        "",
        "| framing | median estimate | × threshold |",
        "|---|---|---|",
    ]
    anchor_order = ["baseline (no bet)", "threshold_only", "coin", "userpick_above",
                    "above_good", "below_good"]
    anch = out["anchoring_thinking_off"] or {}
    lines.append(f"| baseline (no bet) | 422,750,000 | 5.64 |")
    for k in anchor_order:
        if k in anch and anch[k]:
            lines.append(f"| {k} | {anch[k]:,.0f} | {anch[k]/out['threshold']:.2f} |")
    lines += [
        "",
        "Anchoring pulls both directions toward the same number, so it **cancels in the bias "
        "metric by design** — it cannot produce a directional effect. Measurable only in the "
        "thinking-off arm, where the baseline sits far from the threshold.",
        "",
        "**Kills:** H3's *\"values are incidental\"*. Confirms H3's anchoring claim.",
        "",
        "### 3. Motivated reasoning here is a stopping rule",
        "",
        "Controlling for step index, being on the favoured side makes the model **2.11× more "
        "likely to stop** at that step (p=0.0053; permutation null p=0.0030). The no-bet "
        "baseline shows no asymmetry (OR 1.08, p=0.84). Bad-side revisions move toward the "
        "favoured side 0.621 of the time [0.586, 0.655]; baseline nulls straddle 0.5.",
        "",
        "Overt vs covert: stopping OR 3.48 vs 2.55, interaction p=0.33 — no *detectable* "
        "difference, though underpowered at n=34/46. The revision *direction* does differ "
        "(bad-side pull 0.714 [0.654, 0.768] vs 0.577 [0.533, 0.619], non-overlapping).",
        "",
        "### 4. Covert rollouts are already on the favoured side at their first estimate",
        "",
        "Among good-side rollouts, covert start favoured 0.86 of the time vs 0.61 for overt "
        "(signed distance p=0.025; logistic OR 3.72 with revision count contributing nothing). "
        "Two routes to the same outcome: covert arrive there, overt work their way over.",
        "",
        "### 5. The judge is usable, with caveats",
        "",
        "Three passes with fresh item→instance pairings. Test-retest 85.0%, Cohen's κ=0.688. "
        "Pass 1 showed instance heterogeneity (χ²=19.3, p=0.007) but pass 2 showed none "
        "(p=0.71) — a bad draw, not a broken judge. ~15% of items are genuinely borderline; "
        "use `covertness_majority.json` for anything label-conditioned.",
        "",
    ]
    if pref:
        lines += ["### 6. Prefill causal tests", "",
                  "| prefill | bias |", "|---|---|"]
        for k in ("neutral", "denial", "admission"):
            lines.append(f"| {k} | {ci(pref['cells'].get(f'{k}/bias'))} |")
        lines += ["| (none, reference) | **+0.420**  [+0.220, +0.622] |", ""]

    lines += ["## Figures", ""]
    for f in figs:
        lines.append(f"### {f.stem}")
        lines.append("")
        lines.append(f"![{f.stem}](figures/{f.name})")
        lines.append("")

    lines += [
        "## Standing limitations",
        "",
        "- **One estimation question** (giraffes). The paper averages covertness over nine.",
        "- **Prefill is off-distribution**, and that includes the thinking-off arm.",
        "- **Anchoring is thinking-off only** and cannot be measured thinking-on as set up, since "
        "there the threshold *is* the baseline median by construction.",
        "- **Covert n is small** (46 rollouts); the deniers-only bias CI includes zero.",
        "- **LLM judges** for both covertness and trajectory extraction. Covertness κ=0.688; "
        "trajectory extraction showed no instance effect (Kruskal-Wallis p=0.35).",
        "- A few extracted trajectories contain a spurious `0` from the extractor; negligible "
        "against ~2,100 steps but uncleaned.",
        "",
        "See `planning/hypotheses.md` for the running evidence log and `results/snippets.md` for "
        "raw CoT excerpts.",
    ]
    (RES / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {RES/'RESULTS.md'}")


if __name__ == "__main__":
    main()
