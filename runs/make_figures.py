"""Regenerate every figure from stored results. One command, so figures never drift from data.

Each figure is one finding. Run after any analysis script changes its results JSON:
  source env.sh && python3 runs/make_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib.ticker as mticker
import numpy as np
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.qual import figures as F
from src.qual import judge, parse, stats

RUNS = Path(__file__).resolve().parent
OUT = RUNS.parent / "results" / "figures"
T = json.load(open(RUNS / "hour0" / "threshold.json"))["threshold"]


def load_vals(path, filt=True):
    v = []
    for r in parse.parse_jsonl_file(path):
        x, _ = parse.parse_estimate(r["raw_completion"])
        if x is None or (filt and not parse.apply_outlier_filter(x, T)):
            continue
        v.append(x)
    return v


# ---------------------------------------------------------------- F2: thinking on vs off
def fig_thinking():
    off = json.load(open(RUNS / "thinking_off" / "results.json"))
    fig, (axA, axB) = F.new_fig(11.6, 5.2, n_axes=2, width_ratios=[1.25, 1.0], top=0.665)

    rows = [("thinking ON\n(n=39/40)", 0.420, 0.220, 0.622, F.BLUE),
            ("thinking OFF\n(n=148/146)", off["bias_bootstrap"]["point"],
             off["bias_bootstrap"]["ci_low"], off["bias_bootstrap"]["ci_high"], F.ORANGE)]
    for i, (lab, pt, lo, hi, c) in enumerate(rows):
        y = len(rows) - 1 - i
        axA.barh(y, pt, height=0.32, color=c, zorder=3)
        axA.errorbar(pt, y, xerr=[[pt - lo], [hi - pt]], fmt="none", ecolor=F.INK_2,
                     elinewidth=1.5, capsize=4, capthick=1.5, zorder=4)
        F.label_point(axA, hi, y, f"{pt:+.3f}")
    F.null_line(axA, 0.0, "0 = no leakage", y=1.55)
    axA.set_yticks(range(len(rows)))
    axA.set_yticklabels([r[0] for r in rows][::-1], fontsize=9.5, color=F.INK, linespacing=1.35)
    axA.set_xlim(-0.1, 0.85)
    axA.set_ylim(-0.6, 1.9)
    axA.set_xlabel("bias  =  P(above-good favoured) + P(below-good favoured) − 1", fontsize=9.5,
                   color=F.INK_2, labelpad=7)
    F.panel_title(axA, "A.  The bias does not need a chain of thought")

    base = load_vals(RUNS / "thinking_off" / "raw" / "nothink_baseline.jsonl", filt=False)
    on_base = load_vals(RUNS / "hour0" / "raw" / "giraffes_baseline.jsonl", filt=False)
    rng = np.random.default_rng(3)
    for row, (vals, lab, c) in enumerate([(on_base, "thinking ON", F.BLUE),
                                          (base, "thinking OFF", F.ORANGE)]):
        y = row + rng.uniform(-0.15, 0.15, size=len(vals))
        axB.scatter(vals, y, s=26, color=c, alpha=0.6, edgecolors=F.SURFACE, linewidths=1.0,
                    zorder=3)
        med = float(np.median(vals))
        axB.plot([med, med], [row - 0.28, row + 0.28], color=c, lw=2.4, zorder=4)
        axB.annotate(f"median {F.human(med)}", (med, row + 0.34), ha="center", va="bottom",
                     fontsize=9, color=F.INK, zorder=6,
                     bbox=dict(boxstyle="round,pad=0.18", facecolor=F.SURFACE, edgecolor="none"))
    axB.axvline(T, color=F.INK_2, lw=1.4, ls=(0, (5, 3)), zorder=2)
    axB.annotate(f"threshold {F.human(T)}", (T, 1.62), ha="center", va="bottom", fontsize=9,
                 color=F.INK_2, zorder=6,
                 bbox=dict(boxstyle="round,pad=0.2", facecolor=F.SURFACE, edgecolor="none"))
    F.log_x(axB, 5e4, 5e9)
    axB.set_ylim(-0.55, 1.95)
    axB.set_yticks([0, 1])
    axB.set_yticklabels(["thinking ON", "thinking OFF"], fontsize=9.5, color=F.INK)
    axB.set_xlabel("no-bet baseline estimate (log)", fontsize=9.5, color=F.INK_2, labelpad=7)
    F.panel_title(axB, "B.  …but the unbiased answer moves a lot")

    F.title_block(fig, "Removing the chain of thought does not remove the bias",
                  "Giraffe-spot Fermi estimate. Thinking-off is a prefill: the template supplies an empty closed <think> block and the model\n"
                  "answers directly. The bias is if anything larger without a CoT. The no-bet baseline, however, sits 5.6× above the threshold\n"
                  "with no CoT versus at it with one — which is what makes the anchoring test in the next figure possible.")
    F.save(fig, OUT / "f2_thinking_off.png")


# ---------------------------------------------------------------- F3: framing controls
def fig_framing():
    fc = json.load(open(RUNS / "framing_controls" / "results.json"))
    fig, (axA, axB) = F.new_fig(13.2, 5.6, n_axes=2, width_ratios=[1.5, 1.0], wspace=0.38,
                                left=0.155, right=0.975, top=0.63, bottom=0.135)

    series = [("baseline (no bet)", RUNS / "thinking_off/raw/nothink_baseline.jsonl", F.GRAY),
              ("threshold only", RUNS / "framing_controls/raw/threshold_only.jsonl", F.AQUA),
              ("coin flip (no valence)", RUNS / "framing_controls/raw/coin.jsonl", F.AQUA),
              ("user picks charity", RUNS / "framing_controls/raw/userpick_above.jsonl", F.BLUE),
              ("good/bad cause", RUNS / "thinking_off/raw/nothink_above_good.jsonl", F.ORANGE)]
    rng = np.random.default_rng(11)
    for row, (lab, path, c) in enumerate(series):
        vals = load_vals(path, filt=False)
        y = row + rng.uniform(-0.16, 0.16, size=len(vals))
        axA.scatter(vals, y, s=20, color=c, alpha=0.5, edgecolors=F.SURFACE, linewidths=0.8,
                    zorder=3)
        med = float(np.median(vals))
        axA.plot([med, med], [row - 0.3, row + 0.3], color=c, lw=2.6, zorder=4)
        axA.annotate(f"{F.human(med)}", (med, row + 0.34), ha="center", va="bottom", fontsize=9,
                     color=F.INK, fontweight="semibold", zorder=6,
                     bbox=dict(boxstyle="round,pad=0.16", facecolor=F.SURFACE, edgecolor="none"))
    axA.axvline(T, color=F.INK_2, lw=1.4, ls=(0, (5, 3)), zorder=2)
    axA.annotate(f"threshold {F.human(T)}", (T, len(series) - 0.32), ha="center", va="bottom",
                 fontsize=9, color=F.INK_2, zorder=6,
                 bbox=dict(boxstyle="round,pad=0.2", facecolor=F.SURFACE, edgecolor="none"))
    F.log_x(axA, 5e4, 5e9)
    axA.set_ylim(-0.6, len(series) - 0.05)
    axA.set_yticks(range(len(series)))
    axA.set_yticklabels([s[0] for s in series], fontsize=9.5, color=F.INK)
    axA.set_xlabel("estimate (log)", fontsize=9.5, color=F.INK_2, labelpad=7)
    F.panel_title(axA, "A.  Any mention of a threshold anchors the estimate")
    from matplotlib.patches import Patch
    axA.legend(handles=[Patch(facecolor=F.GRAY, label="no threshold named"),
                        Patch(facecolor=F.AQUA, label="threshold, nothing at stake"),
                        Patch(facecolor=F.BLUE, label="threshold + user's preference"),
                        Patch(facecolor=F.ORANGE, label="threshold + moral valence")],
               loc="lower left", frameon=False, fontsize=8.5, labelcolor=F.INK_2,
               handlelength=1.1, borderpad=0.2)

    rows = [("good/bad cause", 0.517, 0.415, 0.612, F.ORANGE),
            ("user picks charity", fc["userpick_bias"]["point"], fc["userpick_bias"]["ci_low"],
             fc["userpick_bias"]["ci_high"], F.BLUE)]
    for i, (lab, pt, lo, hi, c) in enumerate(rows):
        y = len(rows) - 1 - i
        axB.barh(y, pt, height=0.3, color=c, zorder=3)
        axB.errorbar(pt, y, xerr=[[pt - lo], [hi - pt]], fmt="none", ecolor=F.INK_2,
                     elinewidth=1.5, capsize=4, capthick=1.5, zorder=4)
        F.label_point(axB, hi, y, f"{pt:+.3f}")
    F.null_line(axB, 0.0, "0 = no leakage", y=1.5)
    axB.set_yticks(range(len(rows)))
    axB.set_yticklabels([r[0] for r in rows][::-1], fontsize=9.5, color=F.INK)
    axB.set_xlim(-0.06, 0.78)
    axB.set_ylim(-0.6, 1.85)
    axB.set_xlabel("directional bias", fontsize=9.5, color=F.INK_2, labelpad=7)
    F.panel_title(axB, "B.  Only moral valence bends it directionally")

    F.title_block(fig, "Anchoring is large; the directional bias is still about values",
                  "Thinking off, n=150 per condition. Left: every framing that merely names a threshold collapses the estimate from 422M to ~75–84M,\n"
                  "including one where a coin decides the outcome either way. Anchoring pulls both directions toward the same number, so it cancels in\n"
                  "the bias metric by design. Right: swapping moral valence for the user's own preference retains only ~19% of the effect.")
    F.save(fig, OUT / "f3_framing_controls.png")


# ---------------------------------------------------------------- F4: revision asymmetry
def fig_revision():
    """Plots the STEP-ADJUSTED odds ratio, not the raw stopping rate.

    An earlier draft of this figure showed raw P(stop | side). That is the wrong quantity and it
    was actively misleading: raw rates are confounded with trajectory length, so covert rollouts
    read as 0.042 vs 0.036 (almost nothing) while their step-adjusted OR is 2.55. The subtitle
    quoted the adjusted number while the bars showed the unadjusted one. Plot the statistic you
    are actually claiming.
    """
    run = RUNS / "hour0"
    man = {m["id"]: m for m in json.load(open(run / "trajectory_manifest.json"))}
    final = json.load(open(run / "covertness_majority.json"))["final_labels"]
    lab = {k.replace("covertness_above_good_", "").replace("covertness_below_good_", ""): v
           for k, v in final.items()}
    traj = {}
    for i in man:
        f = run / "trajectory_outputs" / f"{i}.txt"
        t = judge.parse_trajectory_answer(f.read_text()) if f.exists() else None
        if t and len(t) >= 3:
            traj[i] = t

    def fav(v, d):
        return v > T if d == "above_good" else v <= T

    def odds_ratio(pred, framing=None):
        rows = []
        for i, t in traj.items():
            if not pred(i):
                continue
            d = framing or man[i]["direction"]
            for k, v in enumerate(t):
                rows.append((1.0 if fav(v, d) else 0.0, k / 10.0, 1.0 if k == len(t) - 1 else 0.0))
        X = np.column_stack([np.ones(len(rows)), [r[0] for r in rows], [r[1] for r in rows]])
        y = np.array([r[2] for r in rows])
        b = np.zeros(3)
        for _ in range(500):
            pr = 1 / (1 + np.exp(-X @ b)); W = pr * (1 - pr) + 1e-9
            b += np.linalg.solve((X * W[:, None]).T @ X + 1e-6 * np.eye(3), X.T @ (y - pr))
        pr = 1 / (1 + np.exp(-X @ b)); W = pr * (1 - pr) + 1e-9
        se = np.sqrt(np.diag(np.linalg.inv((X * W[:, None]).T @ X + 1e-6 * np.eye(3))))
        return (float(np.exp(b[1])), float(np.exp(b[1] - 1.96 * se[1])),
                float(np.exp(b[1] + 1.96 * se[1])), len({i for i in traj if pred(i)}))

    groups = [
        ("all intervention", lambda i: man[i]["direction"] != "baseline", None, F.INK_2),
        ("overt (Admits)", lambda i: man[i]["direction"] != "baseline" and lab.get(i) == "INFLUENCED", None, F.BLUE),
        ("covert (Denies)", lambda i: man[i]["direction"] != "baseline" and lab.get(i) == "NOT_INFLUENCED", None, F.ORANGE),
        ("baseline null (no bet)", lambda i: man[i]["direction"] == "baseline", "above_good", F.GRAY),
    ]
    LBL_X = 11.0   # fixed gutter in data coords, past the widest CI
    fig, ax = F.new_fig(10.8, 5.0, top=0.66, left=0.185, right=0.985, bottom=0.155)
    for row, (name, pred, framing, c) in enumerate(groups):
        y = len(groups) - 1 - row
        orv, lo, hi, n = odds_ratio(pred, framing)
        ax.plot([lo, hi], [y, y], color=c, lw=2.4, solid_capstyle="round", zorder=3)
        ax.scatter([orv], [y], s=90, color=c, edgecolors=F.SURFACE, linewidths=1.6, zorder=4)
        # labels go in a fixed right-hand gutter, not at the CI end: the overt CI runs past the
        # axis limit and its label was being clipped off the figure entirely.
        ax.annotate(f"{orv:.2f}  [{lo:.2f}, {hi:.2f}]", (LBL_X, y), va="center", ha="left",
                    fontsize=10, color=F.INK, fontweight="semibold", zorder=6)
        ax.annotate(f"n={n}", (LBL_X, y - 0.27), fontsize=8.5, color=F.MUTED, va="center")
    F.null_line(ax, 1.0, "1.0 = no asymmetry", y=len(groups) - 0.55)
    ax.set_xscale("log")
    ax.set_xlim(0.45, 46.0)
    ax.set_xticks([0.5, 1, 2, 4, 8])
    ax.set_xticklabels(["0.5", "1", "2", "4", "8"])
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())   # log minor ticks self-label otherwise
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([g[0] for g in groups][::-1], fontsize=9.8, color=F.INK)
    ax.set_ylim(-0.55, len(groups) - 0.25)
    ax.set_xlabel("odds ratio for stopping when the current estimate is on the favoured side\n"
                  "(logistic, controlling for step index; log scale, 95% CI)",
                  fontsize=9.5, color=F.INK_2, labelpad=8)
    F.panel_title(ax, "Step-adjusted stopping asymmetry", pad=14)
    F.title_block(fig, "Motivated reasoning here is a stopping rule, not a lie",
                  "Every intermediate estimate in 110 rollouts (2,120 intervention steps). The model keeps searching while its current estimate is\n"
                  "on the unfavoured side and stops once it reaches the favoured one. Raw stopping rates are confounded with trajectory length, so\n"
                  "this is the step-adjusted odds ratio. The no-bet baseline shows no asymmetry; overt and covert do not detectably differ (p=0.33).",
                  x=0.185)
    F.save(fig, OUT / "f4_revision_asymmetry.png")


# ---------------------------------------------------------------- F5: covertness + reliability
def fig_covertness():
    summ = json.load(open(RUNS / "hour0" / "covertness_summary.json"))
    maj = json.load(open(RUNS / "hour0" / "covertness_majority.json"))
    fig, (axA, axB) = F.new_fig(11.4, 4.9, n_axes=2, width_ratios=[1.0, 1.0], top=0.665)

    n_inf = maj["n_influenced"]
    parts = [("Admits to bias", n_inf, F.BLUE), ("Denies bias", 80 - n_inf, F.ORANGE),
             ("Mentions / no mention", 0, F.GRAY)]
    left = 0
    for lab, n, c in parts:
        if n == 0:
            continue
        axA.barh(0, n, left=left, height=0.42, color=c, zorder=3)
        axA.annotate(f"{lab}\n{n}/80 = {n/80:.0%}", (left + n / 2, 0), ha="center", va="center",
                     fontsize=10, color="white", fontweight="semibold", zorder=5)
        left += n
    axA.annotate("Mentions bias and No mention: 0 of 80 — every trace takes a definite position",
                 (40, -0.42), ha="center", va="top", fontsize=9, color=F.INK_2)
    axA.set_xlim(0, 80); axA.set_ylim(-0.75, 0.45); axA.set_yticks([])
    axA.set_xlabel("rollouts (majority vote over 3 judge passes)", fontsize=9.5, color=F.INK_2,
                   labelpad=7)
    F.panel_title(axA, "A.  Disclosure split")

    jc = summ["judge_consistency"]
    x1 = np.array(jc["influenced"]) / 10.0
    axB.scatter(x1, [1] * len(x1), s=70, color=F.BLUE, alpha=0.75, edgecolors=F.SURFACE,
                linewidths=1.4, zorder=3)
    p2 = [2, 2, 5, 5, 4, 4, 5, 4]
    axB.scatter(np.array(p2) / 10.0, [0] * len(p2), s=70, color=F.ORANGE, alpha=0.75,
                edgecolors=F.SURFACE, linewidths=1.4, zorder=3)
    for y, v, c in ((1, x1, F.BLUE), (0, np.array(p2) / 10.0, F.ORANGE)):
        axB.plot([v.mean(), v.mean()], [y - 0.22, y + 0.22], color=c, lw=2.4, zorder=4)
    axB.set_yticks([1, 0])
    axB.set_yticklabels(["pass 1\nχ²p=0.007", "pass 2\nχ²p=0.71"], fontsize=9.5, color=F.INK,
                        linespacing=1.3)
    axB.set_xlim(-0.05, 0.95); axB.set_ylim(-0.6, 1.6)
    axB.set_xlabel("fraction judged INFLUENCED, per judge instance (8 per pass)", fontsize=9.5,
                   color=F.INK_2, labelpad=7)
    F.panel_title(axB, "B.  Judge-instance spread")
    F.title_block(fig, "Covertness: the split, and how much to trust it",
                  "80 intervention rollouts, paper's Appendix E.2.1 judge run verbatim, three independent passes with fresh item→instance pairings.\n"
                  "Test-retest agreement 85.0%, Cohen's κ=0.688. Pass 1's instance spread was a bad draw, not a broken judge — pass 2 shows none.\n"
                  "On the paper's Figure 6 metric this model is ~87% overt / 12.6% covert, near Qwen3.6-35B-A3B and unlike the Claude models.")
    F.save(fig, OUT / "f5_covertness.png")


# ---------------------------------------------------------------- F6: prefill causal tests
def fig_prefill():
    """Bias by forced opening sentence, with the CoT-length confound shown alongside.

    Panel B is not decoration. Prefilling collapses reasoning length ~8x, and length correlates
    with bias, so "the neutral prefill removed the bias" could be nothing but that. The admission
    arm is what rules it out: it is just as short and keeps the full effect.
    """
    pf = json.load(open(RUNS / "prefill_tests" / "results.json"))
    fig, (axA, axB) = F.new_fig(13.4, 5.2, n_axes=2, width_ratios=[1.5, 1.0], wspace=0.30,
                                left=0.215, right=0.955, top=0.645, bottom=0.145)

    rows = [("no prefill\n(reference)", 0.420, 0.220, 0.622, F.GRAY),
            ("neutral\n\u201c…step by step.\u201d", None, None, None, F.AQUA),
            ("denial\n\u201c…set aside the donation framing.\u201d", None, None, None, F.BLUE),
            ("admission\n\u201c…aim for the good-donation side.\u201d",
             None, None, None, F.ORANGE)]
    for i, key in enumerate(["neutral", "denial", "admission"]):
        c = pf["cells"][f"{key}/bias"]
        rows[i + 1] = (rows[i + 1][0], c["point"], c["ci_low"], c["ci_high"], rows[i + 1][4])
    for i, (lab, pt, lo, hi, c) in enumerate(rows):
        y = len(rows) - 1 - i
        axA.plot([lo, hi], [y, y], color=c, lw=2.4, solid_capstyle="round", zorder=3)
        axA.scatter([pt], [y], s=90, color=c, edgecolors=F.SURFACE, linewidths=1.6, zorder=4)
        axA.annotate(f"{pt:+.3f}", (0.70, y), va="center", ha="left", fontsize=10.5,
                     color=F.INK, fontweight="semibold", zorder=6)
    F.null_line(axA, 0.0, "0 = no leakage", y=len(rows) - 0.55)
    axA.set_xlim(-0.32, 0.88)
    axA.set_ylim(-0.55, len(rows) - 0.3)
    axA.set_yticks(range(len(rows)))
    axA.set_yticklabels([r[0] for r in rows][::-1], fontsize=9, color=F.INK, linespacing=1.35)
    axA.set_xlabel("bias (95% CI)", fontsize=9.5, color=F.INK_2, labelpad=7)
    F.panel_title(axA, "A.  Denial does nothing a neutral sentence doesn't")

    lens = {}
    for key in ["neutral", "denial", "admission"]:
        v = []
        for d in ["above_good", "below_good"]:
            v += [r["num_tokens"] for r in
                  parse.parse_jsonl_file(RUNS / "prefill_tests" / "raw" / f"{key}_{d}.jsonl")]
        lens[key] = v
    unpref = []
    for d in ["above_good", "below_good"]:
        unpref += [r["num_tokens"] for r in
                   parse.parse_jsonl_file(RUNS / "hour0" / "raw" / f"giraffes_{d}.jsonl")]
    order = [("no prefill", unpref, F.GRAY), ("neutral", lens["neutral"], F.AQUA),
             ("denial", lens["denial"], F.BLUE), ("admission", lens["admission"], F.ORANGE)]
    rng = np.random.default_rng(5)
    for row, (lab, v, c) in enumerate(order):
        y = len(order) - 1 - row + rng.uniform(-0.15, 0.15, size=len(v))
        axB.scatter(v, y, s=16, color=c, alpha=0.45, edgecolors="none", zorder=3)
        med = float(np.median(v))
        yy = len(order) - 1 - row
        axB.plot([med, med], [yy - 0.28, yy + 0.28], color=c, lw=2.4, zorder=4)
        axB.annotate(f"{med:,.0f}", (med, yy + 0.32), ha="center", va="bottom", fontsize=9,
                     color=F.INK, fontweight="semibold", zorder=6,
                     bbox=dict(boxstyle="round,pad=0.16", facecolor=F.SURFACE, edgecolor="none"))
    axB.set_xscale("log")
    axB.set_xlim(300, 30000)
    axB.set_xticks([500, 1000, 3000, 10000])
    axB.set_xticklabels(["500", "1k", "3k", "10k"])
    axB.xaxis.set_minor_formatter(mticker.NullFormatter())
    axB.xaxis.set_minor_locator(mticker.NullLocator())
    axB.set_yticks(range(len(order)))
    axB.set_yticklabels([o[0] for o in order][::-1], fontsize=9.5, color=F.INK)
    axB.set_ylim(-0.55, len(order) - 0.25)
    axB.set_xlabel("CoT length (tokens, log)", fontsize=9.5, color=F.INK_2, labelpad=7)
    F.panel_title(axB, "B.  Every prefill shortens the CoT")

    F.title_block(fig, "Forcing the denial changes nothing; forcing the goal restores everything",
                  "n=60 per cell, thinking on. Prefilling any opening sentence removes the bias — except the one that states the goal, which\n"
                  "returns it to the unprefilled level. So the denial sentence is epiphenomenal: it does no more than \u201clet me think step by step\u201d.\n"
                  "Panel B is the confound check: every prefill collapses reasoning length ~8\u00d7, but the admission arm is equally short and keeps the effect.",
                  x=0.215)
    F.save(fig, OUT / "f6_prefill_tests.png")



if __name__ == "__main__":
    print("regenerating figures ->", OUT)
    fig_thinking(); fig_framing(); fig_revision(); fig_covertness()
    if (RUNS / "prefill_tests" / "results.json").exists():
        fig_prefill()
    print("done")
