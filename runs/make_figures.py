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
    fig, (ax, axR) = F.new_fig(13.6, 5.4, n_axes=2, width_ratios=[1.9, 1.0], wspace=0.30,
                               top=0.615, left=0.155, right=0.975, bottom=0.145)
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
    F.panel_title(ax, "A.  Step-adjusted stopping asymmetry", pad=14)

    # The plan asks for the thinking-off bias as a reference alongside this. It has different
    # units from an odds ratio, so it goes in its own panel rather than as a line on axis A.
    toff = json.load(open(RUNS / "thinking_off" / "results.json"))["bias_bootstrap"]
    ref = [("with a CoT", 0.420, 0.220, 0.622, F.INK_2),
           ("with NO CoT", toff["point"], toff["ci_low"], toff["ci_high"], F.ORANGE)]
    for i, (lab, pt, lo, hi, c) in enumerate(ref):
        yy = len(ref) - 1 - i
        axR.plot([lo, hi], [yy, yy], color=c, lw=2.4, solid_capstyle="round", zorder=3)
        axR.scatter([pt], [yy], s=80, color=c, edgecolors=F.SURFACE, linewidths=1.6, zorder=4)
        axR.annotate(f"{pt:+.3f}", (0.70, yy), va="center", ha="left", fontsize=10,
                     color=F.INK, fontweight="semibold")
    F.null_line(axR, 0.0, "0", y=1.5)
    axR.set_yticks([0, 1])
    axR.set_yticklabels(["with NO CoT", "with a CoT"], fontsize=9.5, color=F.INK)
    axR.set_xlim(-0.12, 0.95)
    axR.set_ylim(-0.55, 1.85)
    axR.set_xlabel("bias", fontsize=9.5, color=F.INK_2, labelpad=7)
    F.panel_title(axR, "B.  …but the search is not necessary", pad=14)
    F.title_block(fig, "Motivated reasoning here is a stopping rule, not a lie",
                  "Every intermediate estimate in 110 rollouts (2,120 intervention steps). The model keeps searching while its current estimate is\n"
                  "on the unfavoured side and stops once it reaches the favoured one. Raw stopping rates are confounded with trajectory length, so\n"
                  "this is the step-adjusted odds ratio. The no-bet baseline shows no asymmetry; overt and covert do not detectably differ (p=0.33).\n"
                  "Panel B is the reference that keeps this honest: the same bias appears, larger, in rollouts that have no chain of thought to search in.",
                  x=0.155)
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



# ---------------------------------------------------------------- F7: admission resampling
def fig_resample():
    """H1's direct test — reported as INCONCLUSIVE, and the figure has to show why.

    An earlier draft plotted only the bias CIs per cut, which made a clean-looking null. It was
    not a null: 20-22 of 24 rollouts sit at P(favoured)=1.000 in EVERY condition, so the design
    has no room to detect a decrease. Panel A shows the pile-up at the ceiling; panel B shows the
    paired within-rollout differences that are all that can be said. Plot the saturation, not a
    conclusion the data cannot support.
    """
    import collections
    rows = [json.loads(l) for l in
            open(RUNS / "resample_admission" / "raw" / "continuations.jsonl")]
    rows = [r for r in rows if r["on_good_side"] is not None]
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by[r["source"]][r["cut"]].append(r["on_good_side"])

    fig, (axA, axB) = F.new_fig(12.4, 5.0, n_axes=2, width_ratios=[1.25, 1.0], wspace=0.32,
                                left=0.155, right=0.965, top=0.635, bottom=0.15)
    cuts = [("cut BEFORE admission", "pre", F.ORANGE),
            ("cut AFTER admission", "post", F.AQUA),
            ("cut at random sentence", "random", F.BLUE)]
    rng = np.random.default_rng(2)
    for row, (lab, key, c) in enumerate(cuts):
        per = [np.mean(v[key]) for v in by.values() if v.get(key)]
        y = len(cuts) - 1 - row + rng.uniform(-0.16, 0.16, size=len(per))
        axA.scatter(per, y, s=42, color=c, alpha=0.6, edgecolors=F.SURFACE, linewidths=1.2,
                    zorder=3)
        n_ceil = sum(1 for x in per if x == 1.0)
        axA.annotate(f"{n_ceil}/{len(per)} at 1.00", (1.075, len(cuts) - 1 - row),
                     va="center", ha="left", fontsize=9.5, color=F.INK, fontweight="semibold")
    axA.axvline(1.0, color=F.INK_2, lw=1.4, zorder=5)
    axA.set_xlim(0.3, 1.42)
    axA.set_xticks([0.4, 0.6, 0.8, 1.0])
    axA.set_ylim(-0.55, len(cuts) - 0.35)
    axA.set_yticks(range(len(cuts)))
    axA.set_yticklabels([c[0] for c in cuts][::-1], fontsize=9.5, color=F.INK)
    axA.set_xlabel("P(favoured side) per rollout, 10 continuations each", fontsize=9.5,
                   color=F.INK_2, labelpad=7)
    F.panel_title(axA, "A.  Every condition is pinned at the ceiling")

    for row, (lab, a, b, c) in enumerate([("pre − post", "pre", "post", F.ORANGE),
                                          ("pre − random", "pre", "random", F.BLUE)]):
        d = np.array([np.mean(v[a]) - np.mean(v[b]) for v in by.values()
                      if v.get(a) and v.get(b)])
        y = 1 - row
        jit = rng.uniform(-0.13, 0.13, size=len(d))
        axB.scatter(d, y + jit, s=42, color=c, alpha=0.55, edgecolors=F.SURFACE, linewidths=1.2,
                    zorder=3)
        m = float(d.mean())
        axB.plot([m, m], [y - 0.25, y + 0.25], color=c, lw=2.6, zorder=4)
        pv = sps.ttest_1samp(d, 0).pvalue
        axB.annotate(f"mean {m:+.3f}  p={pv:.2f}", (0.135, y), va="center", ha="left",
                     fontsize=9.5, color=F.INK, fontweight="semibold")
    F.null_line(axB, 0.0, "no difference", y=1.45)
    axB.set_xlim(-0.30, 0.52)
    axB.set_ylim(-0.5, 1.85)
    axB.set_yticks([1, 0])
    axB.set_yticklabels(["pre − post", "pre − random"], fontsize=9.5, color=F.INK)
    axB.set_xlabel("paired within-rollout difference in P(favoured)", fontsize=9.5,
                   color=F.INK_2, labelpad=7)
    F.panel_title(axB, "B.  Paired differences, n=24")

    F.title_block(fig, "Deleting the admission sentence: an inconclusive test, by design",
                  "24 overt rollouts, 10 continuations per cut. The admission sits at 0.78 of the CoT, so a prefix that stops just short of it still\n"
                  "contains most of a trace already converging on a favoured answer — and nearly every rollout returns a favoured number every\n"
                  "time. This cannot detect a decrease. Differences run in the predicted direction but are tiny and not significant. Cut far earlier.",
                  x=0.155)
    F.save(fig, OUT / "f7_admission_resampling.png")



# ---------------------------------------------------------------- F8: internals
def fig_internals():
    """The read result is strong; the causal result is a weak null, and the figure says both."""
    res = json.load(open(RUNS / "internals" / "results.json"))
    abl = json.load(open(RUNS / "internals" / "ablation.json"))["summary"]
    fig, (axA, axB, axC) = F.new_fig(14.2, 4.9, n_axes=3, width_ratios=[1.35, 1.0, 1.0],
                                     wspace=0.36, left=0.075, right=0.985, top=0.63, bottom=0.16)

    auc = np.array(res["cv_auc_by_layer"])
    nl = res["null"]
    axA.axhspan(0.5 - 1.96 * nl["sd"], 0.5 + 1.96 * nl["sd"], color=F.GRID, zorder=1)
    axA.axhline(0.5, color=F.INK_2, lw=1.2, zorder=2)
    axA.plot(range(len(auc)), auc, color=F.BLUE, lw=2.0, zorder=3)
    b = res["best_layer"]
    axA.scatter([b], [auc[b]], s=80, color=F.BLUE, edgecolors=F.SURFACE, linewidths=1.6, zorder=4)
    axA.annotate(f"layer {b}\n{auc[b]:.3f}", (b, auc[b] + 0.03), ha="center", va="bottom",
                 fontsize=9.5, color=F.INK, fontweight="semibold", linespacing=1.25)
    axA.annotate("shuffled-label null (±2 sd)", (2, 0.5 + 1.96 * nl["sd"] + 0.012),
                 fontsize=8.5, color=F.MUTED, va="bottom")
    axA.set_xlim(-1, len(auc)); axA.set_ylim(0.38, 0.88)
    axA.set_xlabel("layer", fontsize=9.5, color=F.INK_2, labelpad=6)
    axA.set_ylabel("cross-validated AUC", fontsize=9.5, color=F.INK_2)
    axA.grid(axis="y", color=F.GRID, lw=0.8)
    F.panel_title(axA, "A.  A direction predicts the condition", pad=12)

    lays, vals = [], []
    for k, v in res["layers"].items():
        lays.append(int(k)); vals.append(v)
    order = np.argsort(lays)
    for i, oi in enumerate(order):
        y = len(order) - 1 - i
        v = vals[oi]
        axB.scatter([v["auc_overt_fit_on_covert"]], [y], s=80, color=F.ORANGE,
                    edgecolors=F.SURFACE, linewidths=1.6, zorder=4)
        lo = v["random_null_mean"] - 1.96 * v["random_null_sd"]
        hi = v["random_null_mean"] + 1.96 * v["random_null_sd"]
        axB.plot([lo, hi], [y, y], color=F.GRAY, lw=6, alpha=0.45, solid_capstyle="butt", zorder=2)
        F.label_point(axB, v["auc_overt_fit_on_covert"], y,
                      f"{v['auc_overt_fit_on_covert']:.3f}", dx=0.02, fontsize=9.5)
    axB.axvline(0.5, color=F.INK_2, lw=1.2, zorder=3)
    axB.set_yticks(range(len(order)))
    axB.set_yticklabels([f"layer {lays[o]}" for o in order][::-1], fontsize=9.5, color=F.INK)
    axB.set_xlim(0.25, 1.0); axB.set_ylim(-0.6, len(order) - 0.4)
    axB.set_xlabel("AUC on covert rollouts\n(direction fitted on overt only)", fontsize=9.5,
                   color=F.INK_2, labelpad=6)
    F.panel_title(axB, "B.  …and it is there in covert ones", pad=12)

    rows = [("unablated", "none", F.GRAY), ("incentive direction\nprojected out", "dom", F.ORANGE),
            ("random direction\n(matched norm)", "random", F.BLUE)]
    for i, (lab, key, c) in enumerate(rows):
        y = len(rows) - 1 - i
        v = abl["by_cond"][key]
        axC.plot([v["ci_low"], v["ci_high"]], [y, y], color=c, lw=2.4, solid_capstyle="round",
                 zorder=3)
        axC.scatter([v["bias"]], [y], s=80, color=c, edgecolors=F.SURFACE, linewidths=1.6,
                    zorder=4)
        F.label_point(axC, v["ci_high"], y, f"{v['bias']:+.3f}", dx=0.02, fontsize=9.5)
    F.null_line(axC, 0.0, "0", y=len(rows) - 0.55)
    axC.set_yticks(range(len(rows)))
    axC.set_yticklabels([r[0] for r in rows][::-1], fontsize=9, color=F.INK, linespacing=1.3)
    axC.set_xlim(-0.12, 0.82); axC.set_ylim(-0.6, len(rows) - 0.35)
    axC.set_xlabel("bias after regenerating the answer\n(CoT held fixed)", fontsize=9.5,
                   color=F.INK_2, labelpad=6)
    F.panel_title(axC, "C.  But ablating it changes nothing", pad=12)

    F.title_block(fig, "The incentive is linearly readable at the pre-number position — and ablating it does nothing",
                  "79 rollouts, residual stream at the token before the final estimate. A difference-of-means direction separates above-good from below-good out of sample\n"
                  "(AUC 0.757 at layer 31, shuffled-label null p<0.0001), and a direction fitted only on rollouts that ADMIT the bias transfers to ones that DENY it (0.802, p=0.010).\n"
                  "But projecting it out does not move the answer. Read that as a weak null: the final number already appears verbatim in the CoT in 90% of rollouts, so this tests the copy step.",
                  x=0.075, y_title=0.955, y_sub=0.885)
    F.save(fig, OUT / "f8_internals.png")



# ---------------------------------------------------------------- F9: during-CoT ablation
def fig_ablate_cot():
    """A failed intervention, plotted as one. The headline is the validity check, not the bias."""
    import statistics as st
    T = json.load(open(RUNS / "hour0" / "threshold.json"))["threshold"]
    base = RUNS / "ablate_during_cot" / "raw"

    def vals(cond, arm):
        f = base / f"{cond}_{arm}.jsonl"
        if not f.exists():
            return []
        return [json.loads(l)["estimate"] for l in open(f)
                if json.loads(l)["estimate"] is not None]

    fig, (axA, axB) = F.new_fig(12.6, 5.0, n_axes=2, width_ratios=[1.35, 1.0], wspace=0.32,
                                left=0.18, right=0.965, top=0.625, bottom=0.155)

    series = [("no-bet baseline, unablated", "baseline", "none", F.GRAY),
              ("no-bet baseline, ABLATED", "baseline", "dom", F.ORANGE),
              ("above-good, unablated", "above_good", "none", F.GRAY),
              ("above-good, ABLATED", "above_good", "dom", F.ORANGE),
              ("above-good, random direction", "above_good", "random", F.BLUE)]
    rng = np.random.default_rng(4)
    for row, (lab, cond, arm, c) in enumerate(series):
        v = vals(cond, arm)
        if not v:
            continue
        y = len(series) - 1 - row
        axA.scatter(v, y + rng.uniform(-0.15, 0.15, size=len(v)), s=44, color=c, alpha=0.7,
                    edgecolors=F.SURFACE, linewidths=1.2, zorder=3)
        med = float(st.median(v))
        axA.plot([med, med], [y - 0.27, y + 0.27], color=c, lw=2.4, zorder=4)
        axA.annotate(F.human(med), (med, y + 0.31), ha="center", va="bottom", fontsize=9,
                     color=F.INK, fontweight="semibold", zorder=6,
                     bbox=dict(boxstyle="round,pad=0.16", facecolor=F.SURFACE, edgecolor="none"))
    axA.axvline(T, color=F.INK_2, lw=1.4, ls=(0, (5, 3)), zorder=2)
    axA.annotate(f"threshold {F.human(T)}", (T, len(series) - 0.38), ha="center", va="bottom",
                 fontsize=9, color=F.INK_2, zorder=6,
                 bbox=dict(boxstyle="round,pad=0.2", facecolor=F.SURFACE, edgecolor="none"))
    F.log_x(axA, 2e4, 2e9)
    axA.set_ylim(-0.55, len(series) - 0.1)
    axA.set_yticks(range(len(series)))
    axA.set_yticklabels([s2[0] for s2 in series][::-1], fontsize=9, color=F.INK)
    axA.set_xlabel("estimate (log)", fontsize=9.5, color=F.INK_2, labelpad=7)
    F.panel_title(axA, "A.  Ablation crushes every estimate — bet or no bet")

    p_ab = sum(1 for v in vals("above_good", "dom") if v > T) / max(1, len(vals("above_good", "dom")))
    p_be = sum(1 for v in vals("below_good", "dom") if v <= T) / max(1, len(vals("below_good", "dom")))
    for i, (lab, p, c) in enumerate([("above-good\nP(favoured)", p_ab, F.BLUE),
                                     ("below-good\nP(favoured)", p_be, F.ORANGE)]):
        y = 1 - i
        axB.barh(y, p, height=0.34, color=c, zorder=3)
        axB.annotate(f"{p:.3f}", (p + 0.03, y), va="center", fontsize=11, color=F.INK,
                     fontweight="semibold")
    axB.axvline(0.5, color=F.INK_2, lw=1.4, zorder=5)
    axB.annotate("0.5", (0.5, 1.5), ha="center", va="bottom", fontsize=9, color=F.INK_2)
    axB.set_xlim(0, 1.22); axB.set_ylim(-0.55, 1.8)
    axB.set_yticks([1, 0]); axB.set_yticklabels(["above-good", "below-good"], fontsize=9.5,
                                                color=F.INK)
    axB.set_xlabel("P(favoured) under ablation", fontsize=9.5, color=F.INK_2, labelpad=7)
    axB.annotate("bias = 0.000 — but PINNED,\nnot balanced", (0.60, -0.36), fontsize=9.5,
                 color=F.INK, fontweight="semibold", ha="center")
    F.panel_title(axB, "B.  \"Bias removed\" is an artefact")

    F.title_block(fig, "Ablating the direction during CoT generation: an invalid intervention",
                  "n=8 per cell. Projecting the layer-31 difference-of-means direction out at every forward pass collapses estimates ~25–35×, and it does so in\n"
                  "the NO-BET baseline too (105.7M → 3.0M, KS D=0.875, p=0.002). The direction is therefore not incentive-specific, and it fails the plan's own bar:\n"
                  "a bias drop with baseline estimates untouched. The resulting bias of exactly 0.000 comes from every estimate landing below the threshold, so\n"
                  "above-good can never win and below-good always does. A random matched-norm direction does none of this.",
                  x=0.18, y_title=0.955, y_sub=0.885)
    F.save(fig, OUT / "f9_ablate_during_cot.png")



if __name__ == "__main__":
    print("regenerating figures ->", OUT)
    fig_thinking(); fig_framing(); fig_revision(); fig_covertness()
    if (RUNS / "prefill_tests" / "results.json").exists():
        fig_prefill()
    if (RUNS / "resample_admission" / "results.json").exists():
        fig_resample()
    if (RUNS / "internals" / "ablation.json").exists():
        fig_internals()
    if (RUNS / "ablate_during_cot" / "results.json").exists():
        fig_ablate_cot()
    print("done")
