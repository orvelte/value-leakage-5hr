"""Is there a linear incentive direction at the pre-number position, and is it in covert rollouts?

Three questions, in the order the plan asks them:

  1. Does a difference-of-means direction between above-good and below-good separate the two
     conditions OUT OF SAMPLE? Reported as cross-validated AUC per layer, against a
     shuffled-label null. In-sample separation is guaranteed and meaningless at d=5120, n=79.
  2. Is that direction present in COVERT rollouts? Fit on overt only, test on covert only, so
     the covert score is fully out of sample.
  3. Does the J-lens say the model is poised to verbalize the incentive there? Concept logits
     for above/below/donate/... transported through the lens, paired across conditions.

Constraints carried from prior lens work and applied here: digit tokens are never read through
the lens (adjacent digit directions are near-collinear), and every lens claim is checked against
a shuffled-label null rather than reported as a raw top-k.

Run with: source env.sh && python3 runs/internals_analyze.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.qual import lens as qual_lens
from src.qual import stats


def dom_auc_cv(X, y, folds=8, seed=0):
    """Cross-validated AUC of the difference-of-means projection. y in {0,1}."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    scores = np.zeros(len(y))
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        if len(np.unique(y[tr])) < 2:
            return float("nan")
        d = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)
        n = np.linalg.norm(d)
        if n == 0:
            return float("nan")
        scores[te] = X[te] @ (d / n)
    return stats.roc_auc(scores, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="runs/internals/acts.pt")
    ap.add_argument("--probes", default="runs/internals/probe_weights.pt")
    ap.add_argument("--n-null", type=int, default=200)
    args = ap.parse_args()

    d = torch.load(args.acts, weights_only=False)
    A, rows = d["acts"].numpy(), d["rows"]
    y = np.array([1 if r["direction"] == "above_good" else 0 for r in rows])
    covert = np.array([r["label"] == "NOT_INFLUENCED" for r in rows])
    print(f"{A.shape[0]} rollouts, {A.shape[1]} layers, d={A.shape[2]}")
    print(f"  above_good {int(y.sum())} / below_good {int((1-y).sum())} | "
          f"overt {int((~covert).sum())} / covert {int(covert.sum())}\n")

    res = {"n": int(A.shape[0]), "layers": {}}

    print("=== 1. Cross-validated AUC of the DoM direction, per layer ===")
    print("   (in-sample separation at d=5120, n=79 is guaranteed; only CV means anything)")
    aucs, nulls = [], []
    rng = np.random.default_rng(0)
    for L in range(A.shape[1]):
        X = A[:, L, :]
        a = dom_auc_cv(X, y)
        aucs.append(a)
    aucs = np.array(aucs)
    best = int(np.nanargmax(aucs))
    print(f"  best layer {best}: CV AUC {aucs[best]:.3f}")
    for L in range(0, A.shape[1], 8):
        print(f"    layer {L:2d}: {aucs[L]:.3f}")
    print(f"    layer 62 (lens target): {aucs[62]:.3f}   layer 63 (final): {aucs[63]:.3f}")

    print(f"\n=== shuffled-label null at layer {best} ({args.n_null} permutations) ===")
    Xb = A[:, best, :]
    for _ in range(args.n_null):
        nulls.append(dom_auc_cv(Xb, rng.permutation(y), seed=int(rng.integers(1e6))))
    nulls = np.array([x for x in nulls if np.isfinite(x)])
    pv = float((nulls >= aucs[best]).mean())
    print(f"  observed {aucs[best]:.3f} | null mean {nulls.mean():.3f} sd {nulls.std():.3f} "
          f"| p = {pv:.4f}")
    res["cv_auc_by_layer"] = aucs.tolist()
    res["best_layer"] = best
    res["null"] = {"mean": float(nulls.mean()), "sd": float(nulls.std()), "p": pv}

    print("\n=== 2. Fit the direction on OVERT rollouts, test on COVERT ===")
    for L in [best, 48, 56, 62]:
        X = A[:, L, :]
        tr, te = ~covert, covert
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        dvec = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)
        dvec = dvec / np.linalg.norm(dvec)
        auc_cov = stats.roc_auc(X[te] @ dvec, y[te])
        rand = [stats.roc_auc(X[te] @ (lambda v: v / np.linalg.norm(v))(
            rng.standard_normal(X.shape[1])), y[te]) for _ in range(200)]
        rand = np.array(rand)
        print(f"  layer {L:2d}: overt-fit direction on covert rollouts AUC = {auc_cov:.3f}   "
              f"random-direction null {rand.mean():.3f}±{rand.std():.3f}  "
              f"p={float((np.abs(rand-0.5) >= abs(auc_cov-0.5)).mean()):.3f}")
        res["layers"][str(L)] = {"auc_overt_fit_on_covert": float(auc_cov),
                                 "random_null_mean": float(rand.mean()),
                                 "random_null_sd": float(rand.std())}

    print("\n=== 3. J-lens concept probes at the same position ===")
    pw = torch.load(args.probes, weights_only=False)
    W = pw["W_probe"].numpy()             # [n_probes, d_model]
    gw = pw["final_norm_w"].numpy()       # RMSNorm gain
    words = pw["probe_words"]
    lens_path = qual_lens.download_lens("qwen3.5-27b", "j-lens")
    L_ = qual_lens.load_lens(lens_path)
    src_layers = list(L_["source_layers"])

    def transported_logits(h, layer):
        J = L_["J"][layer].numpy().astype(np.float32)
        t = J @ h
        t = t / (np.sqrt((t ** 2).mean()) + 1e-6) * gw     # RMSNorm
        return W @ t

    probe_layers = [L for L in (40, 48, 56, 62) if L in src_layers]
    print(f"  {'layer':>6s} {'probe':>10s} {'above_good':>12s} {'below_good':>12s} "
          f"{'diff':>9s} {'perm p':>8s}")
    res["jlens"] = {}
    for L in probe_layers:
        S = np.stack([transported_logits(A[i, L, :].astype(np.float32), L)
                      for i in range(A.shape[0])])
        for wi, w in enumerate(words):
            if w not in ("above", "below", "donate", "threshold", "honest"):
                continue
            a, b = S[y == 1, wi], S[y == 0, wi]
            diff = a.mean() - b.mean()
            perm = []
            for _ in range(500):
                yy = rng.permutation(y)
                perm.append(S[yy == 1, wi].mean() - S[yy == 0, wi].mean())
            perm = np.array(perm)
            p = float((np.abs(perm) >= abs(diff)).mean())
            print(f"  {L:6d} {w:>10s} {a.mean():12.3f} {b.mean():12.3f} {diff:+9.3f} {p:8.3f}"
                  + ("  *" if p < 0.05 else ""))
            res["jlens"][f"L{L}/{w}"] = {"above": float(a.mean()), "below": float(b.mean()),
                                         "diff": float(diff), "perm_p": p}

    out = Path("runs/internals/results.json")
    out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
