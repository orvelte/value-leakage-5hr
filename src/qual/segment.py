"""CoT sentence segmentation — the resampling/locator unit throughout this project.

Simple regex-based sentence splitter, consistent with the "Thought Anchors"-style treatment of a
CoT as a sequence of sentences (Bogdan et al., papers/) — not a full NLP sentence tokenizer, since
CoTs are informal, number-heavy, and full of trailing ellipses/parentheticals that break
spaCy/NLTK defaults about as often as a simple heuristic does.
"""
import re

# Split on sentence-final punctuation followed by whitespace and a capital/number/quote start,
# but don't split on periods that are part of decimals, abbreviations-with-digits, or ellipses.
# KNOWN LIMITATION (found 2026-08-29 while locating admission sentences): a bare markdown bullet
# marker "*" is not in the lookahead class below, so a numbered heading and all of its
# sub-bullets collapse into a single "sentence". Median sentence is 90 chars but the maximum
# observed is 1,571. Anything that treats a sentence index as a fine-grained position -- the
# disclaimer locator, prefix-forcing resampling -- is therefore working at block granularity,
# not clause granularity. Adding "*" to the class would renumber every existing index, so any
# fix has to invalidate and re-run the locator outputs that depend on the current numbering.
_SPLIT_RE = re.compile(
    r"""
    (?<!\.\.)              # not part of an ellipsis
    (?<=[.!?])             # after sentence-final punctuation
    (?<!\d\.\d)            # not a decimal point (heuristic: preceding char isn't digit.digit)
    \s+
    (?=[A-Z0-9"'(—])  # followed by a capital, digit, quote, paren, or em-dash
    """,
    re.VERBOSE,
)


def segment_sentences(cot_text):
    """Extract the reasoning portion (inside <think>...</think>, if present) and split into
    sentences. Also splits on double-newlines (paragraph breaks) as a stronger boundary."""
    text = cot_text
    if "<think>" in text and "</think>" in text:
        text = text.split("<think>", 1)[1].split("</think>", 1)[0]
    text = text.strip()
    if not text:
        return []
    paragraphs = re.split(r"\n\s*\n", text)
    sentences = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        parts = _SPLIT_RE.split(para)
        sentences.extend(p.strip() for p in parts if p.strip())
    return sentences
