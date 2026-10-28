"""Figures for runs/releases, from results.json -> runs/releases/figures/*.png.

Style per the spec: Liberation Serif (bundled in fonts/), inward ticks on all four sides, faint
horizontal grid, white background. Colour = arm (or disclosure bucket), hatch = version
(3.5 solid, 3.6 '..', 3.8 '//').

Run with: python3 runs/releases/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
for f in (HERE / "fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))

PALETTE = ["#3B73C8", "#E07B39", "#4CAF7D", "#E5A93B", "#D9769C", "#8E6BBF", "#3BA7A7"]
HATCH = {"q35": "", "q36": "..", "q38": "//"}
VLABEL = {"q35": "3.5", "q36": "3.6", "q38": "3.8"}
plt.rcParams.update({
    "font.family": "Liberation Serif", "font.size": 11, "axes.titlesize": 12,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": "#222222", "axes.linewidth": 0.8, "hatch.linewidth": 0.7,
    "hatch.color": "#ffffff",
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 4, "ytick.major.size": 4, "xtick.minor.size": 2, "ytick.minor.size": 2,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#000000", "grid.alpha": 0.08,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
})

# Original 3.5 run (results/RESULTS.md, framing_controls, second_question), drawn as hollow markers.
ORIGINAL_Q35 = {"giraffes_off": 0.517, "giraffes_on": 0.420, "userpick_off": 0.096, "turns_off": 0.399}

ARMS = [("giraffes_off", "Good/bad cause\nthinking off"), ("giraffes_on", "Good/bad cause\nthinking on"),
        ("userpick_off", "User picks charity\nthinking off"), ("turns_off", "Left turns\nthinking off")]


def version_legend(models):
    return [Patch(facecolor="#9a9a9a", edgecolor="#222222", hatch=HATCH[m], label=VLABEL[m])
            for m in models]


def fig_bias(res):
    models = [m for m in ("q35", "q36", "q38") if m in res["models"]]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    w = 0.26
    for ai, (arm, lab) in enumerate(ARMS):
        for mi, m in enumerate(models):
            b = res["models"][m]["bias"].get(arm)
            if not b:
                continue
            x = ai + (mi - (len(models) - 1) / 2) * w
            ax.bar(x, b["point"], w * 0.92, color=PALETTE[ai], hatch=HATCH[m], edgecolor="#222222",
                   linewidth=0.6)
            ax.errorbar(x, b["point"], yerr=[[b["point"] - b["ci_low"]], [b["ci_high"] - b["point"]]],
                        color="#222222", capsize=2.5, lw=0.9)
        x0 = ai - (len(models) - 1) / 2 * w if "q35" in models else ai
        ax.plot(x0, ORIGINAL_Q35[arm], marker="D", ms=6, mfc="white", mec="#111111", mew=1.1,
                ls="none", zorder=5)
    ax.axhline(0, color="#222222", lw=0.8)
    ax.set_xticks(range(len(ARMS)), [a[1] for a in ARMS])
    ax.set_ylim(-0.1, 1.0)
    ax.set_ylabel("Bias toward favoured side (95% CI)")
    ax.set_title("Donation Bet bias by model and arm", loc="left")
    handles = version_legend(models) + [plt.Line2D([], [], marker="D", ms=6, mfc="white",
                                                   mec="#111111", ls="none", label="original 3.5 run")]
    ax.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "bias_by_model.png", dpi=200)
    plt.close(fig)


def fig_disclosure(res):
    models = [m for m in ("q35", "q36", "q38") if "disclosure" in res["models"].get(m, {})]
    if not models:
        return
    buckets = ["Admits to bias", "Mentions bias", "No mention of bias", "Denies bias"]
    colors = [PALETTE[2], PALETTE[3], PALETTE[6], PALETTE[1]]
    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    for yi, m in enumerate(models):
        d = res["models"][m]["disclosure"]
        left = 0.0
        for b, c in zip(buckets, colors):
            share = d["split"][b] / d["n_labelled"]
            ax.barh(yi, share, left=left, color=c, hatch=HATCH[m], edgecolor="#222222", lw=0.6)
            if share >= 0.06:
                ax.text(left + share / 2, yi, f"{d['split'][b]}", ha="center", va="center",
                        fontsize=10, color="#111111",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
            left += share
    ax.set_yticks(range(len(models)), [f"Qwen{VLABEL[m]}" for m in models])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of thinking-on bet rollouts (majority judge label; counts on bars)")
    ax.set_title("What the reasoning trace discloses about the threshold", loc="left")
    ax.grid(False)
    ax.legend(handles=[Patch(facecolor=c, edgecolor="#222222", label=b) for b, c in zip(buckets, colors)],
              loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=4, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT / "disclosure_by_model.png", dpi=200)
    plt.close(fig)


def fig_stopping(res):
    models = [m for m in ("q35", "q36", "q38") if "stopping" in res["models"].get(m, {})]
    if not models:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for i, m in enumerate(models):
        s = res["models"][m]["stopping"]
        ax.bar(i, s["or"], 0.6, color=PALETTE[0], hatch=HATCH[m], edgecolor="#222222", lw=0.6)
        ax.errorbar(i, s["or"], yerr=[[s["or"] - s["ci"][0]], [s["ci"][1] - s["or"]]],
                    color="#222222", capsize=3, lw=0.9)
        nulls = [v for k, v in s.items() if k.startswith("null_or")]
        ax.plot([i + 0.18] * len(nulls), nulls, ls="none", marker="o", ms=5, mfc="#9a9a9a",
                mec="#222222", mew=0.6, zorder=5)
    ax.axhline(1, color="#222222", lw=0.8, ls="--")
    top = max(res["models"][m]["stopping"]["ci"][1] for m in models)
    ax.set_ylim(0, top * 1.3)
    ax.set_xticks(range(len(models)), [f"Qwen{VLABEL[m]}" for m in models])
    ax.set_ylabel("Odds ratio: stop on favoured side\n(hazard model, 95% CI)")
    ax.legend(handles=[plt.Line2D([], [], ls="none", marker="o", ms=5, mfc="#9a9a9a", mec="#222222",
                                  label="no-bet baseline, scored\nunder each framing (null)")],
              loc="upper right", fontsize=9)
    ax.set_title("Stopping asymmetry (thinking on)", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "stopping_by_model.png", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    res = json.load(open(HERE / "results.json"))
    fig_bias(res)
    fig_disclosure(res)
    fig_stopping(res)
    print("figures ->", OUT)


if __name__ == "__main__":
    main()
