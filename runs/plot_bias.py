"""Hour-0 figure: the Donation Bet bias replicates on Qwen3.5-27B.

Two panels, because the claim has two halves that a single panel would conflate:
  A. the estimate distributions actually move across the counterfactual pair, and
  B. P(good side) exceeds the 0.5 null in BOTH directions -- which is what makes it
     value leakage rather than a one-sided anchoring artifact.

Colors are categorical slots 1 (blue) and 2 (orange) from the dataviz reference palette;
validated all-pairs on the light surface (CVD dE 24.7, normal-vision 33.6, contrast >=3:1).
Baseline is de-emphasis gray: it is context, not a third series.

Run with: python3 runs/plot_bias.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.qual import parse, stats  # noqa: E402

RUN = Path(__file__).resolve().parent / "hour0"
OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(exist_ok=True)

SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
MUTED     = "#8a8983"
GRID      = "#e6e5e1"
BLUE      = "#2a78d6"   # above_good
ORANGE    = "#eb6834"   # below_good
GRAY      = "#9d9c95"   # baseline = context

LABEL = {"above_good": "above-good\n(>T → good cause)",
         "below_good": "below-good\n(≤T → good cause)",
         "baseline":   "baseline\n(no bet)"}
COLOR = {"above_good": BLUE, "below_good": ORANGE, "baseline": GRAY}


def load():
    T = json.load(open(RUN / "threshold.json"))["threshold"]
    d = {}
    for cond in ["baseline", "above_good", "below_good"]:
        recs = parse.parse_jsonl_file(RUN / "raw" / f"giraffes_{cond}.jsonl")
        vals, kept = [], []
        for r in recs:
            v, _ = parse.parse_estimate(r["raw_completion"])
            if v is None:
                continue
            vals.append(v)
            kept.append(parse.apply_outlier_filter(v, T))
        d[cond] = (np.array(vals), np.array(kept))
    return T, d


def boot_p(outcomes, n_resamples=2000, seed=0):
    """Percentile CI for a single proportion, same binomial resampling as
    stats.balanced_bias_bootstrap uses per direction -- so panel B's whiskers and the
    pooled bias in the subtitle come from one procedure, not two."""
    o = np.asarray(outcomes, dtype=float)
    p = o.mean()
    rng = np.random.default_rng(seed)
    draws = rng.binomial(len(o), p, size=n_resamples) / len(o)
    return p, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def human(v, _=None):
    if v >= 1e9:
        return f"{v/1e9:g}B"
    if v >= 1e6:
        return f"{v/1e6:g}M"
    if v >= 1e3:
        return f"{v/1e3:g}K"
    return f"{v:g}"


def main():
    T, d = load()
    outcomes = {}
    for cond in ["above_good", "below_good"]:
        vals, kept = d[cond]
        outcomes[cond] = [1 if parse.classify_on_good_side(v, T, cond) else 0
                          for v, k in zip(vals, kept) if k]
    bias, blo, bhi = stats.balanced_bias_bootstrap(outcomes["above_good"],
                                                   outcomes["below_good"])

    fig = plt.figure(figsize=(12.6, 5.7), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.52, 1.0], wspace=0.30,
                          left=0.105, right=0.965, top=0.655, bottom=0.125)
    axA, axB = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for ax in (axA, axB):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=9.5, length=3, color=GRID)

    # ---------- Panel A: where the estimates actually land ----------
    order = ["baseline", "above_good", "below_good"]
    rng = np.random.default_rng(7)
    for row, cond in enumerate(order):
        vals, kept = d[cond]
        y = row + rng.uniform(-0.17, 0.17, size=len(vals))
        c = COLOR[cond]
        axA.scatter(vals[kept], y[kept], s=42, color=c, alpha=0.75,
                    edgecolors=SURFACE, linewidths=1.4, zorder=3)
        if (~kept).any():   # outlier-filtered points: shown, but hollow
            axA.scatter(vals[~kept], y[~kept], s=42, facecolors="none",
                        edgecolors=c, linewidths=1.4, alpha=0.9, zorder=3)
        med = float(np.median(vals[kept]))
        axA.plot([med, med], [row - 0.30, row + 0.30], color=c, lw=2.4,
                 solid_capstyle="round", zorder=4)
        axA.annotate(f"median {human(med)}", (med, row + 0.36), ha="center", va="bottom",
                     fontsize=9, color=INK, fontweight="medium", zorder=6,
                     bbox=dict(boxstyle="round,pad=0.18", facecolor=SURFACE, edgecolor="none"))

    axA.axvline(T, color=INK_2, lw=1.4, ls=(0, (5, 3)), zorder=2)
    axA.annotate(f"threshold  {human(T)}\n(this model's own baseline median)",
                 (T, 2.62), ha="center", va="bottom", fontsize=9, color=INK_2, linespacing=1.35,
                 zorder=6, bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none"))
    axA.set_xscale("log")
    axA.set_xlim(6e4, 3e9)
    axA.xaxis.set_major_formatter(FuncFormatter(human))
    axA.set_ylim(-0.65, 2.95)
    axA.set_yticks(range(len(order)))
    axA.set_yticklabels([LABEL[c] for c in order], fontsize=9.5, color=INK, linespacing=1.35)
    axA.set_xlabel("estimated total giraffe spots (log scale)", fontsize=10, color=INK_2, labelpad=7)
    axA.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    axA.set_axisbelow(True)
    axA.set_title("A.  Each rollout's final estimate", fontsize=11.5, color=INK,
                  fontweight="semibold", loc="left", pad=30)

    # ---------- Panel B: P(good side) against the 0.5 null ----------
    conds = ["above_good", "below_good"]
    ys = [1, 0]
    for cond, y in zip(conds, ys):
        p, lo, hi = boot_p(outcomes[cond])
        n = len(outcomes[cond])
        axB.barh(y, p - 0.5, left=0.5, height=0.34, color=COLOR[cond], zorder=3)
        axB.errorbar(p, y, xerr=[[p - lo], [hi - p]], fmt="none", ecolor=INK_2,
                     elinewidth=1.5, capsize=4, capthick=1.5, zorder=4)
        axB.annotate(f"{p:.2f}", (hi + 0.022, y), va="center", ha="left",
                     fontsize=11, color=INK, fontweight="semibold")
        axB.annotate(f"n={n}", (hi + 0.022, y - 0.20), va="center", ha="left",
                     fontsize=8.5, color=MUTED)

    axB.axvline(0.5, color=INK_2, lw=1.4, zorder=5)
    axB.annotate("0.5 = no leakage", (0.5, 1.62), ha="center", va="bottom",
                 fontsize=9, color=INK_2, zorder=6,
                 bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none"))
    axB.set_xlim(0.0, 1.0)
    axB.set_ylim(-0.72, 1.9)
    axB.set_yticks(ys)
    axB.set_yticklabels([LABEL[c] for c in conds], fontsize=9.5, color=INK, linespacing=1.35)
    axB.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axB.set_xlabel("P(estimate lands on the good-cause side)", fontsize=10, color=INK_2, labelpad=7)
    axB.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    axB.set_axisbelow(True)
    axB.set_title("B.  Both point estimates sit above the null", fontsize=11.5, color=INK,
                  fontweight="semibold", loc="left", pad=30)
    axB.annotate(f"pooled bias = {bias:+.2f}   95% CI [{blo:+.2f}, {bhi:+.2f}]",
                 (0.5, -0.62), ha="center", va="center", fontsize=10, color=INK,
                 fontweight="semibold",
                 bbox=dict(boxstyle="round,pad=0.42", facecolor="#f0efec", edgecolor="none"))

    fig.suptitle("The Donation Bet bias replicates on Qwen3.5-27B", x=0.105, y=0.955,
                 ha="left", fontsize=14.5, color=INK, fontweight="bold")
    fig.text(0.105, 0.888,
             "Giraffe-spot Fermi estimate, temperature 1, n = 30 / 40 / 40. Swapping which side of the threshold funds the good cause shifts the whole\n"
             "estimate distribution with it. Pooled over both framings the bias is +0.42 with the CI excluding 0; the below-good direction alone does\n"
             "not reach significance at n=40 (CI 0.48\u20130.78). Hollow = outside the [T/10, 10T] window; one above-good rollout hit the 16k cap unfinished.",
             ha="left", va="top", fontsize=9.5, color=INK_2, linespacing=1.55)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"hour0_bias_replication.{ext}", dpi=200, facecolor=SURFACE)
    print(f"bias={bias:+.3f} [{blo:+.3f},{bhi:+.3f}]")
    for c in conds:
        p, lo, hi = boot_p(outcomes[c])
        print(f"{c}: p={p:.3f} [{lo:.3f},{hi:.3f}] n={len(outcomes[c])}")
    print("wrote", OUT / "hour0_bias_replication.png")


if __name__ == "__main__":
    main()
