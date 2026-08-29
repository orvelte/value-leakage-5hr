"""Shared figure style, so every result in this project reads as one system.

Palette is the dataviz reference instance, categorical slots 1-3, validated all-pairs on the
light surface: CVD dE 9.2 worst pair (deutan), normal-vision 24.0, lightness band and chroma
floor pass. Slot 3 (aqua) sits at 2.74:1 against the surface, below the 3:1 bar, so the RELIEF
RULE applies -- every mark carries a visible direct label. That is the house style here anyway:
these are research figures and the numbers must be readable off the page.

Conventions used throughout:
  - medians and CIs, never bare means, because these distributions are heavy-tailed
  - the null (0.5 for probabilities, 0 for bias, 1.0 for odds ratios) is always drawn
  - a control/baseline series is de-emphasis gray; it is context, not a competing series
  - log x-axis whenever estimates are plotted, since they span four orders of magnitude
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8983"
GRID = "#e6e5e1"
BAND = "#f0efec"

BLUE = "#2a78d6"     # slot 1 — first/primary series (thinking on, overt, above-good)
ORANGE = "#eb6834"   # slot 2 — second series (thinking off, covert, below-good)
AQUA = "#1baf7a"     # slot 3 — third series; direct-label it (contrast 2.74:1)
GRAY = "#9d9c95"     # de-emphasis: baselines, nulls, context


def human(v, _=None):
    """Compact number formatter for axes: 4.2e8 -> 420M."""
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= div:
            return f"{v/div:g}{suf}"
    return f"{v:g}"


def new_fig(width=9.0, height=4.6, n_axes=1, width_ratios=None, wspace=0.30,
            left=0.115, right=0.965, top=0.70, bottom=0.145):
    fig = plt.figure(figsize=(width, height), facecolor=SURFACE)
    gs = fig.add_gridspec(1, n_axes, width_ratios=width_ratios, wspace=wspace,
                          left=left, right=right, top=top, bottom=bottom)
    axes = [fig.add_subplot(gs[i]) for i in range(n_axes)]
    for ax in axes:
        style_axis(ax)
    return fig, (axes[0] if n_axes == 1 else axes)


def style_axis(ax, xgrid=True):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9.5, length=3, color=GRID)
    if xgrid:
        ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def title_block(fig, title, subtitle, x=0.115, y_title=0.945, y_sub=0.875):
    fig.suptitle(title, x=x, y=y_title, ha="left", fontsize=13.5, color=INK, fontweight="bold")
    fig.text(x, y_sub, subtitle, ha="left", va="top", fontsize=9.3, color=INK_2, linespacing=1.55)


def panel_title(ax, text, pad=26):
    ax.set_title(text, fontsize=11, color=INK, fontweight="semibold", loc="left", pad=pad)


def null_line(ax, x, label, y=None, vertical=True):
    """Draw the null and label it with a surface-backed box so the rule never strikes text."""
    (ax.axvline if vertical else ax.axhline)(x, color=INK_2, lw=1.4, zorder=5)
    if label:
        ax.annotate(label, (x, y), ha="center", va="bottom", fontsize=9, color=INK_2, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none"))


def label_point(ax, x, y, text, dx=0.012, fontsize=10.5, weight="semibold", color=None):
    ax.annotate(text, (x + dx, y), va="center", ha="left", fontsize=fontsize,
                color=color or INK, fontweight=weight, zorder=6)


def log_x(ax, lo, hi):
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_formatter(FuncFormatter(human))


def save(fig, path, note=None):
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {p}" + (f"  ({note})" if note else ""))
