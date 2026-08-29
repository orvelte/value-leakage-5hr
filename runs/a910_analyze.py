"""A9 + A10 — between-arm contrasts for the second batch of prefill arms.

Same machinery as A3/A4: load_arm and bias_draws are imported from a34_analyze rather than
reimplemented, and the arm registry there is extended in place, so every bias and every contrast
in the project comes from one parse path and one resampling scheme.

Run with: python3 runs/a910_analyze.py   (no GPU)
"""
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import a34_analyze as base                                    # noqa: E402

for arm in ["neutral_b", "neutral_c", "admit_then_reverse", "admit_deny_admit"]:
    base.ARMS[arm] = "prefill_extra2"

CONTRASTS = [
    # --- A9: which half of "let me work through this step by step" is active? ---
    ("neutral_b", "neutral", "A9: does 'be careful' alone reproduce the neutral null?"),
    ("neutral_c", "neutral", "A9: does naming the decomposition procedure reproduce it?"),
    ("neutral_b", "neutral_c", "A9: deliberation vs procedure, against each other"),
    ("neutral_b", "minimal", "A9: is 'be careful' distinguishable from a contentless prefill?"),
    # --- A10: is it the denial's content, or any reversal? ---
    ("admit_then_reverse", "admit_then_deny",
     "A10: does a non-honesty reversal act like the denial?"),
    ("admit_then_reverse", "admit_twice", "A10: or like no reversal at all?"),
    ("admit_deny_admit", "admit_then_deny", "last-stance control: re-admitting after the denial"),
    ("admit_deny_admit", "admit_twice", "does re-admitting fully restore the goal?"),
]


def main():
    arms = {a: base.load_arm(a) for a in base.ARMS}
    res = {"arms": {}, "contrasts": {}}

    print(f"{'arm':20s}{'bias':>9s}{'95% CI':>20s}{'med tok':>9s}{'n_valid':>9s}{'drop':>7s}")
    for arm, (ab, be, toks, n_raw, n_trunc) in arms.items():
        from src.qual import stats
        pt, lo, hi = stats.balanced_bias_bootstrap(ab, be)
        n_valid = len(ab) + len(be)
        res["arms"][arm] = {"bias": pt, "ci": [lo, hi], "median_tokens": st.median(toks),
                            "n_raw": n_raw, "n_valid": n_valid, "n_truncated": n_trunc,
                            "drop_rate": 1 - n_valid / n_raw}
        print(f"{arm:20s}{pt:+9.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>20s}"
              f"{st.median(toks):9.0f}{n_valid:9d}{1 - n_valid/n_raw:7.2f}")

    print(f"\n{'contrast':44s}{'diff':>9s}{'95% CI':>20s}{'boot p':>9s}")
    for a, b, why in CONTRASTS:
        da = base.bias_draws(arms[a][0], arms[a][1], seed=1)
        db = base.bias_draws(arms[b][0], arms[b][1], seed=2)
        d = da - db
        pt = (np.mean(arms[a][1]) + np.mean(arms[a][0]) - 1) - \
             (np.mean(arms[b][1]) + np.mean(arms[b][0]) - 1)
        lo, hi = np.quantile(d, (0.025, 0.975))
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        res["contrasts"][f"{a} - {b}"] = {"diff": float(pt), "ci": [float(lo), float(hi)],
                                          "boot_p": float(p), "why": why}
        print(f"{a+' - '+b:44s}{pt:+9.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>20s}{p:9.4f}")
        print(f"{'':44s}{why}")

    out = HERE / "prefill_extra2" / "a910_contrasts.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
