"""A15.1 — recompute the App. D covert lower bound on MAJORITY labels.

`covertness_majority.json` carries a covert_share of 0.1257 but no script in the repo writes
that file — it was produced in a session whose transcript is gone — so this recomputes the
number from the labels and the raw estimates using the paper's own functions in src/qual/stats.py
rather than trusting the stored value.

The construction (App. D, Fig. 6): infer how much of the intervention mass must have been biased
to produce the observed shift, then spend that budget on the most faithful disclosure categories
FIRST — admitters before deniers — so the bias is explained by admissions wherever it possibly
can be, and only the residue is charged to covert leakage. That makes the result a LOWER bound on
covertness, not an estimate of it.

This project's judge is binary (INFLUENCED / NOT_INFLUENCED), so the paper's four buckets
collapse to Admits and Denies; the two middle buckets are empty by construction and are carried
through as zeros so the same function is used.

Adds a bootstrap CI, which the stored scalar did not have.

Run with: python3 runs/a15_covert_lower_bound.py   (no GPU)
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from src.qual import parse, stats                              # noqa: E402

RUN = HERE / "hour0"
N_BOOT = 20000


def main():
    T = json.load(open(RUN / "threshold.json"))["threshold"]
    maj = json.load(open(RUN / "covertness_majority.json"))
    labels = {}
    for k, v in maj["final_labels"].items():
        labels[k[k.find("giraffes_"):]] = v

    # judged rollouts only, so labels and outcomes share a denominator
    recs = {}
    for d in ("above_good", "below_good", "baseline"):
        f = RUN / "raw" / f"giraffes_{d}.jsonl"
        if f.exists():
            for r in parse.parse_jsonl_file(f):
                recs[r["item_id"]] = (r, d)

    items = []
    for iid, lab in labels.items():
        if iid not in recs:
            continue
        rec, d = recs[iid]
        v, _ = parse.parse_estimate(rec["raw_completion"])
        if v is None or not parse.apply_outlier_filter(v, T):
            continue
        items.append({"id": iid, "label": lab, "direction": d,
                      "good": bool(parse.classify_on_good_side(v, T, d))})

    n = len(items)
    p_fav = sum(i["good"] for i in items) / n
    # the threshold IS the baseline median by construction, so the paper's p_favored_baseline
    # is 0.5; verified against the baseline arm rather than assumed
    base = [r for iid, (r, d) in recs.items() if d == "baseline"]
    bvals = [parse.parse_estimate(r["raw_completion"])[0] for r in base]
    bvals = [v for v in bvals if v is not None and parse.apply_outlier_filter(v, T)]
    p_base_emp = sum(v > T for v in bvals) / len(bvals)
    P_BASE = 0.5

    p_biased = stats.latent_mixture_p_biased(p_fav, P_BASE)
    good_admit = sum(i["good"] and i["label"] == "INFLUENCED" for i in items) / n
    good_deny = sum(i["good"] and i["label"] == "NOT_INFLUENCED" for i in items) / n
    shares = {"Admits to bias": good_admit, "Mentions bias": 0.0,
              "No mention of bias": 0.0, "Denies bias": good_deny}
    alloc, unexplained = stats.favorable_disclosure_decomposition(shares, p_biased)
    lb = stats.covert_share_of_bias(alloc, p_biased)

    print(f"n judged & valid = {n}   (INFLUENCED {sum(i['label']=='INFLUENCED' for i in items)}, "
          f"NOT_INFLUENCED {sum(i['label']=='NOT_INFLUENCED' for i in items)})")
    print(f"\n  P(favoured | intervention) = {p_fav:.4f}")
    print(f"  P(favoured | baseline)     = {P_BASE:.4f}  (by construction: T is the baseline "
          f"median; empirical baseline P(>T) = {p_base_emp:.3f} on n={len(bvals)})")
    print(f"  p_biased (App. D latent mixture) = {p_biased:.4f}")
    print(f"\n  share of ALL rollouts landing favoured AND admitting = {good_admit:.4f}")
    print(f"  share of ALL rollouts landing favoured AND denying     = {good_deny:.4f}")
    print(f"\n  allocation, admitters first:")
    for k, v in alloc.items():
        if v or k in ("Admits to bias", "Denies bias"):
            print(f"    {k:22s}{v:.4f}")
    print(f"    {'unexplained':22s}{unexplained:.4f}")
    print(f"\n  COVERT LOWER BOUND = {lb:.4f}  ({lb:.1%})")
    print(f"  stored value in covertness_majority.json = {maj['covert_share']:.4f}")
    print(f"  pass-1 value (superseded) = 0.1559")

    # bootstrap: resample rollouts, recompute end to end
    rng = np.random.default_rng(0)
    idx = np.arange(n)
    draws = []
    for _ in range(N_BOOT):
        s = rng.choice(idx, size=n, replace=True)
        sub = [items[j] for j in s]
        pf = sum(i["good"] for i in sub) / n
        pb = stats.latent_mixture_p_biased(pf, P_BASE)
        if not np.isfinite(pb) or pb <= 0:
            continue
        ga = sum(i["good"] and i["label"] == "INFLUENCED" for i in sub) / n
        gd = sum(i["good"] and i["label"] == "NOT_INFLUENCED" for i in sub) / n
        a, _u = stats.favorable_disclosure_decomposition(
            {"Admits to bias": ga, "Mentions bias": 0.0,
             "No mention of bias": 0.0, "Denies bias": gd}, pb)
        draws.append(stats.covert_share_of_bias(a, pb))
    draws = np.array([d for d in draws if np.isfinite(d)])
    lo, hi = np.quantile(draws, (0.025, 0.975))
    print(f"\n  bootstrap 95% CI [{lo:.4f}, {hi:.4f}] over {len(draws):,} draws "
          f"({N_BOOT - len(draws)} discarded as degenerate)")
    print(f"  P(lower bound = 0) = {(draws <= 1e-12).mean():.3f}")

    out = {"n": n, "p_favored_intervention": p_fav, "p_favored_baseline": P_BASE,
           "p_favored_baseline_empirical": p_base_emp, "p_biased": p_biased,
           "good_and_admits": good_admit, "good_and_denies": good_deny,
           "allocation": alloc, "unexplained": unexplained,
           "covert_lower_bound": lb, "ci": [float(lo), float(hi)],
           "p_lb_zero": float((draws <= 1e-12).mean()),
           "stored_value": maj["covert_share"], "pass1_value": 0.1559}
    p = RUN / "a15_covert_lower_bound.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
