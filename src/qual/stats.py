"""Statistics helpers: bias bootstrap CI (matching donation_bet/bias_metrics.py's procedure,
the paper's own method), KS test wrapper, power calc, AUC.
"""
import math

import numpy as np


def effective_sample_size_row(records, parse_fn):
    """A dup-rate check worth running before trusting any n: guards against the correlated-seed
    gotcha noted in sample.py (duplicate completions across "independent" rollouts).

    records: list of raw rollout dicts with 'raw_completion'.
    parse_fn: callable(raw_completion) -> (value_or_None, reason) e.g. parse.parse_estimate.
    Returns dict with n_raw, n_parsed, n_unique_completions, dup_rate, n_unique_estimates.
    """
    n_raw = len(records)
    texts = [r["raw_completion"] for r in records]
    n_unique_completions = len(set(texts))
    parsed_values = []
    for r in records:
        val, _ = parse_fn(r["raw_completion"])
        if val is not None:
            parsed_values.append(val)
    n_parsed = len(parsed_values)
    dup_rate = (1 - n_unique_completions / n_parsed) if n_parsed > 0 else float("nan")
    n_unique_estimates = len(set(parsed_values))
    return {
        "n_raw": n_raw,
        "n_parsed": n_parsed,
        "n_unique_completions": n_unique_completions,
        "dup_rate": dup_rate,
        "n_unique_estimates": n_unique_estimates,
    }


def balanced_bias_bootstrap(above_good_outcomes, below_good_outcomes, n_resamples=2000, seed=0):
    """bias = p_below_good_on_good_side + p_above_good_on_good_side - 1, independent
    per-direction binomial resampling, percentile CI. This is the paper's own bias metric
    (spec: bias = 2(p_favored - 0.5), pooled over above-good/below-good).

    *_outcomes: array-like of 0/1 (on_good_side) for each direction.
    Returns (point_estimate, ci_low, ci_high).
    """
    above = np.asarray(above_good_outcomes, dtype=float)
    below = np.asarray(below_good_outcomes, dtype=float)
    if len(above) == 0 or len(below) == 0:
        return float("nan"), float("nan"), float("nan")

    p_above = above.mean()
    p_below = below.mean()
    point = p_below + p_above - 1.0

    rng = np.random.default_rng(seed)
    draws_above = rng.binomial(len(above), p_above, size=n_resamples) / len(above)
    draws_below = rng.binomial(len(below), p_below, size=n_resamples) / len(below)
    draws = draws_below + draws_above - 1.0

    ci_low, ci_high = np.quantile(draws, (0.025, 0.975))
    return point, float(ci_low), float(ci_high)


def two_sample_ks(sample_a, sample_b):
    from scipy import stats as scipy_stats

    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    result = scipy_stats.ks_2samp(a, b)
    return float(result.statistic), float(result.pvalue)


def n_for_bootstrap_cell(observed_rate, target_count, min_n=1):
    """Rollouts needed per direction for `target_count` expected hits at `observed_rate`."""
    if observed_rate <= 0:
        return float("inf")
    return max(min_n, math.ceil(target_count / observed_rate))


def n_for_ratio_power_ttest(observed_ratio, pooled_cv, power=0.8, alpha=0.05):
    """Approximate per-group n needed to detect `observed_ratio` != 1 at `power`, given the
    pooled coefficient of variation `pooled_cv` (std/mean) of the underlying measure (CoT
    length or # in-CoT estimates), via a two-sample log-ratio z-test approximation:
    n ~= 2 * (cv^2) * ((z_alpha/2 + z_power) / log(ratio))^2
    """
    from scipy import stats as scipy_stats

    if observed_ratio <= 1.0 or observed_ratio == 1.0:
        return float("inf")
    z_alpha = scipy_stats.norm.ppf(1 - alpha / 2)
    z_power = scipy_stats.norm.ppf(power)
    log_ratio = math.log(observed_ratio)
    n = 2 * (pooled_cv ** 2) * ((z_alpha + z_power) / log_ratio) ** 2
    return math.ceil(n)


def rollout_multiplier_for_marginal_gate(observed_bias, target_bias=0.30):
    """(target/observed)^2 multiplier for scaling up n when an observed effect is real but
    underpowered relative to a target effect size."""
    if observed_bias <= 0:
        return float("inf")
    return (target_bias / observed_bias) ** 2


def roc_auc(scores, labels):
    """AUC via the Mann-Whitney U statistic (no sklearn dependency). scores: array-like of
    the scalar predictor; labels: array-like of 0/1 (1=positive class, e.g. 'favored').
    Returns nan if either class is empty."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    i = 0
    r = 1
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (r + (r + (j - i))) / 2.0
        ranks[order[i : j + 1]] = avg_rank
        r += j - i + 1
        i = j + 1
    sum_ranks_pos = ranks[labels == 1].sum()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))
