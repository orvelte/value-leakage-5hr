"""A7 step 1b — does the thinking-ON layer-31 incentive direction transfer to thinking-OFF?

The gate for the rest of A7. The direction is the difference of means (above_good minus
below_good) fitted on the THINKING-ON activations, applied to the THINKING-OFF activations with
NO refit. If it reads the condition there, the direction is not an artefact of having a chain of
thought and ablation/steering are worth running. If AUC ~ 0.5, A7 stops here and that is the
result.

Three things this reports that a bare AUC would hide:
  - the in-sample AUC on the thinking-on set the direction came from, as a check that the
    regenerated direction reproduces the one reported earlier (0.853 in-sample, CV 0.757)
  - a shuffled-label null on the thinking-off set
  - a random matched-norm direction null, which is the stricter of the two: high-dimensional
    residuals can separate almost anything, so beating label-shuffling is not enough

Note there are no covertness labels thinking-off — with no CoT there is nothing to admit or
deny — so the overt->covert transfer question does not exist on this set.

Run with: python3 runs/a7_transfer.py   (no GPU; needs both acts files)
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
N_NULL = 1000


def dom_direction(A, y):
    """difference of means, unit-normalised. A: [n, d], y: 1 = above_good."""
    d = A[y == 1].mean(0) - A[y == 0].mean(0)
    n = np.linalg.norm(d)
    return d / n if n > 0 else d


def auc(scores, labels):
    return stats.roc_auc(list(scores), list(labels))


def main():
    on = torch.load(HERE / "internals" / "acts.pt", map_location="cpu", weights_only=False)
    off = torch.load(HERE / "internals" / "acts_nothink.pt", map_location="cpu",
                     weights_only=False)

    A_on = on["acts"].numpy()
    y_on = np.array([1 if r["direction"] == "above_good" else 0 for r in on["rows"]])
    A_off = off["acts"].numpy()
    y_off = np.array([1 if r["direction"] == "above_good" else 0 for r in off["rows"]])

    print(f"thinking-ON  : {A_on.shape}, {y_on.sum()} above / {len(y_on)-y_on.sum()} below")
    print(f"thinking-OFF : {A_off.shape}, {y_off.sum()} above / {len(y_off)-y_off.sum()} below")

    res = {"n_on": int(len(y_on)), "n_off": int(len(y_off)), "best_layer": BEST_LAYER,
           "layers": {}}
    rng = np.random.default_rng(0)

    n_layers = A_on.shape[1]
    per_layer = []
    for L in range(n_layers):
        w = dom_direction(A_on[:, L, :], y_on)
        per_layer.append(auc(A_off[:, L, :] @ w, y_off))
    res["transfer_auc_by_layer"] = [float(x) for x in per_layer]

    for L in (BEST_LAYER, int(np.argmax(per_layer)), n_layers - 1):
        if str(L) in res["layers"]:
            continue
        w = dom_direction(A_on[:, L, :], y_on)
        a_in = auc(A_on[:, L, :] @ w, y_on)
        s_off = A_off[:, L, :] @ w
        a_off = auc(s_off, y_off)

        shuf = np.array([auc(s_off, rng.permutation(y_off)) for _ in range(N_NULL)])
        p_shuf = float((np.abs(shuf - 0.5) >= abs(a_off - 0.5)).mean())

        norm = np.linalg.norm(A_off[:, L, :].std(0)) or 1.0
        rand = []
        for _ in range(N_NULL):
            r = rng.normal(size=w.shape)
            r /= np.linalg.norm(r)
            rand.append(auc(A_off[:, L, :] @ r, y_off))
        rand = np.array(rand)
        p_rand = float((np.abs(rand - 0.5) >= abs(a_off - 0.5)).mean())

        res["layers"][str(L)] = {
            "auc_in_sample_thinkon": float(a_in), "auc_transfer_thinkoff": float(a_off),
            "shuffled_null_mean": float(shuf.mean()), "shuffled_null_sd": float(shuf.std()),
            "p_shuffled": p_shuf,
            "random_dir_null_mean": float(rand.mean()), "random_dir_null_sd": float(rand.std()),
            "p_random_dir": p_rand}
        print(f"\n--- layer {L} ---")
        print(f"  in-sample AUC (thinking-on, the set it was fit on): {a_in:.3f}")
        print(f"  TRANSFER AUC (thinking-off, no refit):              {a_off:.3f}")
        print(f"  shuffled-label null   {shuf.mean():.3f} +- {shuf.std():.3f}   p={p_shuf:.4f}")
        print(f"  random-direction null {rand.mean():.3f} +- {rand.std():.3f}   p={p_rand:.4f}")

    # --- validity diagnostics, without which the AUCs above cannot be read ---
    # 1. Layer 0 is essentially the token embeddings. If the condition is already separable
    #    there, nothing downstream is evidence of a learned incentive representation: the two
    #    prompts differ in wording ("good cause"/"bad cause" swapped) and thinking-off leaves
    #    only ~200 tokens between the prompt and the target position, so the prompt dominates.
    # 2. The thinking-on set gets the same random-direction null, because if ITS null is also
    #    wide then the published 0.757 / 0.802 numbers inherit the same problem.
    w31_on = dom_direction(A_on[:, BEST_LAYER, :], y_on)
    rand_on = []
    for _ in range(N_NULL):
        r = rng.normal(size=w31_on.shape)
        r /= np.linalg.norm(r)
        rand_on.append(auc(A_on[:, BEST_LAYER, :] @ r, y_on))
    rand_on = np.array(rand_on)
    res["diagnostics"] = {
        "layer0_transfer_auc": float(per_layer[0]),
        "thinkon_random_dir_null_mean": float(rand_on.mean()),
        "thinkon_random_dir_null_sd": float(rand_on.std()),
        "thinkoff_random_dir_null_sd": res["layers"][str(BEST_LAYER)]["random_dir_null_sd"]}
    print("\n=== validity diagnostics ===")
    print(f"  layer-0 (embeddings) transfer AUC on thinking-off: {per_layer[0]:.3f}")
    print(f"  random-direction null SD, thinking-OFF: "
          f"{res['layers'][str(BEST_LAYER)]['random_dir_null_sd']:.3f}")
    print(f"  random-direction null SD, thinking-ON : {rand_on.std():.3f} "
          f"(mean {rand_on.mean():.3f})")

    best = int(np.argmax(per_layer))
    print(f"\nbest transferring layer: {best} (AUC {per_layer[best]:.3f});"
          f"  layer {BEST_LAYER} = {per_layer[BEST_LAYER]:.3f}")
    print(f"  reference: thinking-on CV AUC 0.757 at layer {BEST_LAYER}, "
          f"overt->covert 0.802")

    a = res["layers"][str(BEST_LAYER)]
    if per_layer[0] > 0.75:
        verdict = ("INVALID TEST — the condition is separable at layer 0 (embeddings), so the "
                   "thinking-off activations encode which PROMPT was used, not an incentive "
                   "representation. The high transfer AUC is not evidence either way, and a "
                   "random direction does as well (p={:.3f}). A7 stops, but this is NOT the "
                   "planned null: the measurement has no discriminative power here."
                   ).format(a["p_random_dir"])
    elif a["p_random_dir"] < 0.05 and abs(a["auc_transfer_thinkoff"] - 0.5) > 0.1:
        verdict = "TRANSFERS — ablation/steering worth running"
    else:
        verdict = "DOES NOT TRANSFER — A7 stops here, and that is the result"
    res["verdict"] = verdict
    print(f"\nVERDICT at layer {BEST_LAYER}: {verdict}")

    out = HERE / "internals" / "a7_transfer.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
