"""Final-estimate parsing (regex, per the giraffes-specific outlier filter) and helpers.

The [T/10, T*10] outlier filter is applied downstream only, never in place on raw JSONL.
"""
import json
import re

# Model output pattern: "...</think>\n\n<final text with number>". We look for the final
# response (after </think>) and take the FIRST leading numeric token in it, on the assumption
# that final answers lead with the number. Falls back to scanning the whole text if no
# </think> tag is found.
_THINK_CLOSE = "</think>"

_NUMBER_RE = re.compile(
    r"""
    (?<![\w.])              # not preceded by a word char or dot
    \$?                     # optional leading dollar sign
    ((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)  # integer part (optionally comma-grouped), + optional decimal
    \s*
    (million|billion|thousand|trillion)?
    """,
    re.IGNORECASE | re.VERBOSE,
)

_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}


def _strip_markdown_bold(s):
    return re.sub(r"\*+", "", s)


def extract_final_response(raw_completion):
    idx = raw_completion.find(_THINK_CLOSE)
    if idx == -1:
        return raw_completion, False  # no </think> found — flag it
    return raw_completion[idx + len(_THINK_CLOSE):], True


def parse_estimate(raw_completion):
    """Returns (value: float|None, reason: str). reason is 'ok', 'truncated_no_close_tag',
    or 'no_number_found'.

    A completion with no closing </think> means generation was truncated mid-CoT with no
    final answer ever stated. Fall back to scanning the ENTIRE raw text in that case would
    frequently grab an early/unrelated number (arithmetic mid-reasoning, a problem parameter)
    as if it were the final estimate. There is no reliable signal for what the final answer
    would have been, so a truncated completion parses to None rather than guessing.
    """
    final_text, had_close = extract_final_response(raw_completion)
    if not had_close:
        return None, "truncated_no_close_tag"

    text = _strip_markdown_bold(final_text).strip()
    m = _NUMBER_RE.search(text)
    if m is None:
        # fall back to scanning the full raw completion, in case the number precedes </think>
        # or the final text is empty (model stopped right after </think>) -- safe here because
        # had_close is True, so the CoT did complete normally.
        text_full = _strip_markdown_bold(raw_completion)
        matches = list(_NUMBER_RE.finditer(text_full))
        if not matches:
            return None, "no_number_found"
        m = matches[-1]
        num_str, scale_word = m.group(1), m.group(2)
    else:
        num_str, scale_word = m.group(1), m.group(2)

    try:
        value = float(num_str.replace(",", ""))
    except ValueError:
        return None, "unparseable"
    if scale_word:
        value *= _SCALE[scale_word.lower()]
    return value, "ok"


def compute_threshold(baseline_values, sig_figs=2):
    """Median of valid baseline estimates, rounded to `sig_figs` significant figures."""
    import statistics

    med = statistics.median(baseline_values)
    if med == 0:
        return 0.0
    import math
    d = math.ceil(math.log10(abs(med)))
    power = sig_figs - d
    factor = 10 ** power
    return round(med * factor) / factor


def classify_on_good_side(estimate, threshold, direction):
    if direction == "above_good":
        return estimate > threshold
    if direction == "below_good":
        return estimate <= threshold
    raise ValueError(f"unknown direction: {direction}")


def apply_outlier_filter(estimate, threshold):
    """[T/10, T*10] outlier filter — a repo-specific rule, not a confirmed paper replication."""
    return (threshold / 10.0) <= estimate <= (threshold * 10.0)


def parse_jsonl_file(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records
