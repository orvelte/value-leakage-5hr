"""A12.2 — does the layer-31 direction track the OUTCOME, or only the prompt?

The standing internals conclusion is "encoded, not shown to be used": the direction separates
above_good from below_good (CV AUC 0.757), but ablating it moved nothing, and A7 showed that on
thinking-off data an equivalent-looking separation is just the prompt.

This is the test that distinguishes the two readings without any generation. Hold the CONDITION
fixed — above_good rollouts only — and ask whether the projection predicts where that rollout's
final estimate actually landed (> T or <= T). Within a condition the prompt is identical, so
prompt information cannot contribute. If the projection predicts the outcome, the direction
carries something about the answer the model is going to give. If it is at chance, "encoded, not
shown to be used" stands and should be stated that way in the write-up.

Nulls: random matched-norm directions (the strict one, per A7) and shuffled outcome labels.

Run with: python3 runs/a12_within_condition_auc.py   (no GPU; needs acts.pt)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from src.qual import stats                                     # noqa: E402

BEST_LAYER = 31
N_NULL = 2000


def auc(scores, labels):
    return stats.roc_auc(list(scores), list(labels))


def main():
    d = torch.load(HERE / "internals" / "acts.pt", map_location="cpu", weights_only=False)
    A = d["acts"].numpy()
    rows = d["rows"]
    T = d["threshold"]
    y_cond = np.array([1 if r["direction"] == "above_good" else 0 for r in rows])

    # the direction, fit exactly as before: difference of means across conditions
    w = A[y_cond == 1, BEST_LAYER, :].mean(0) - A[y_cond == 0, BEST_LAYER, :].mean(0)
    w = w / (np.linalg.norm(w) or 1.0)

    rng = np.random.default_rng(0)
    res = {"layer": BEST_LAYER, "n": len(rows), "threshold": T, "conditions": {}}

    print(f"acts {A.shape}; direction = DoM at layer {BEST_LAYER}, fit across conditions")
    print(f"\n{'condition':14s}{'n':>5s}{'n(>T)':>7s}{'AUC(final>T)':>14s}"
          f"{'rand-dir null':>18s}{'p':>8s}{'shuf p':>9s}")
    for cond, name in ((1, "above_good"), (0, "below_good")):
        idx = np.where(y_cond == cond)[0]
        X = A[idx, BEST_LAYER, :]
        est = np.array([rows[i]["estimate"] for i in idx])
        y_out = (est > T).astype(int)
        if len(set(y_out.tolist())) < 2:
            print(f"{name:14s}{len(idx):5d}   degenerate outcome labels, skipped")
            continue
        a = auc(X @ w, y_out)

        rand = []
        for _ in range(N_NULL):
            r = rng.normal(size=w.shape)
            r /= np.linalg.norm(r)
            rand.append(auc(X @ r, y_out))
        rand = np.array(rand)
        p_rand = float((np.abs(rand - 0.5) >= abs(a - 0.5)).mean())

        s = X @ w
        shuf = np.array([auc(s, rng.permutation(y_out)) for _ in range(N_NULL)])
        p_shuf = float((np.abs(shuf - 0.5) >= abs(a - 0.5)).mean())

        res["conditions"][name] = {
            "n": int(len(idx)), "n_above_T": int(y_out.sum()), "auc_outcome": float(a),
            "rand_null_mean": float(rand.mean()), "rand_null_sd": float(rand.std()),
            "p_random_dir": p_rand, "p_shuffled": p_shuf}
        print(f"{name:14s}{len(idx):5d}{int(y_out.sum()):7d}{a:14.3f}"
              f"{f'{rand.mean():.3f}+-{rand.std():.3f}':>18s}{p_rand:8.3f}{p_shuf:9.3f}")

    print(f"\n  reference: across-condition CV AUC 0.757, overt->covert 0.802 (both at L{BEST_LAYER})")
    sig = [c for c, v in res["conditions"].items() if v["p_random_dir"] < 0.05]
    res["verdict"] = ("tracks the OUTCOME within condition in: " + ", ".join(sig)
                      if sig else
                      "at chance within condition against a random-direction null — "
                      "'encoded, not shown to be used' stands")
    print(f"\nVERDICT: {res['verdict']}")

    out = HERE / "internals" / "a12_within_condition_auc.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
