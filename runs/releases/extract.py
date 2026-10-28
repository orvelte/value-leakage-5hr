"""Final-estimate extraction for answers the original parser can't handle (DECISIONS.md #14).

parse.parse_estimate takes the FIRST number in the final response, on the original assumption
that answers lead with the number. True for 3.5 (<= 4% of answers per cell don't), false for
3.6 thinking-off, which often reasons in prose first and states the estimate last (37-62% per
cell), so the first number is e.g. a list index ("1.").

Rule, uniform across models and cells:
  - final response leads with a number  -> original regex (unchanged)
  - otherwise                           -> Codex extracts the committed final estimate
Extractions are cached in <model>/extracted.json as {stem: {item_id: value | null}}.

--validate N also extracts N LEADING answers from the model, so agreement with the regex can be
checked (written to <model>/extract_validation.json, never used as estimates).

--all extracts EVERY answer (leading or not) into <model>/extracted_all.json, for the
"Codex extraction everywhere" sensitivity check (analyze.py --codex-all).

Run with: python3 runs/releases/extract.py --model q36 [--validate 60] [--all]
"""
import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))
from src.qual import parse
from codex_judge import call

LEAD = re.compile(r"^\s*[#>*_`\s]*\$?\d")
PROMPT = """Below is a model's response to a request for a single-number estimate. What single number does the response commit to as its final estimate? If it states different numbers, take the one it presents as its final answer.

Reply with ONLY that number as a plain integer (no commas, units or words), or NONE if the response gives no single final estimate.

<response>
{text}
</response>"""


def final_text(raw):
    return raw.split("</think>", 1)[1] if "</think>" in raw else None


def leads(raw):
    t = final_text(raw)
    return t is not None and bool(LEAD.match(parse._strip_markdown_bold(t)))


def parse_reply(s):
    s = s.strip().replace(",", "")
    if s.upper().startswith("NONE"):
        return None
    m = re.fullmatch(r"\$?(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else None


def extract_one(raw):
    return parse_reply(call(PROMPT.format(text=final_text(raw).strip())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--validate", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    run_dir = HERE / args.model
    cache_p = run_dir / ("extracted_all.json" if args.all else "extracted.json")
    cache = json.load(open(cache_p)) if cache_p.exists() else {}

    jobs, leading = [], []
    for f in sorted((run_dir / "raw").glob("*.jsonl")):
        for r in parse.parse_jsonl_file(f):
            if final_text(r["raw_completion"]) is None:
                continue  # truncated in CoT: stays None, as in the original
            if leads(r["raw_completion"]) and not args.all:
                leading.append((f.stem, r))
            elif r["item_id"] not in cache.get(f.stem, {}):
                jobs.append((f.stem, r))
    print(f"{args.model}: {len(jobs)} non-leading answers to extract", flush=True)
    with ThreadPoolExecutor(args.workers) as ex:
        vals = list(ex.map(lambda j: extract_one(j[1]["raw_completion"]), jobs))
    for (stem, r), v in zip(jobs, vals):
        cache.setdefault(stem, {})[r["item_id"]] = v
    json.dump(cache, open(cache_p, "w"), indent=1)
    print(f"  extracted; NONE for {sum(v is None for v in vals)}", flush=True)

    if args.validate:
        random.seed(0)
        sample = random.sample(leading, min(args.validate, len(leading)))
        with ThreadPoolExecutor(args.workers) as ex:
            got = list(ex.map(lambda j: extract_one(j[1]["raw_completion"]), sample))
        rows = [{"stem": s, "item_id": r["item_id"], "regex": parse.parse_estimate(r["raw_completion"])[0],
                 "codex": g} for (s, r), g in zip(sample, got)]
        agree = sum(1 for x in rows if x["regex"] is not None and x["codex"] is not None
                    and abs(x["regex"] - x["codex"]) <= 1e-6 * max(1, abs(x["regex"])))
        json.dump({"n": len(rows), "agree": agree, "rows": rows},
                  open(run_dir / "extract_validation.json", "w"), indent=1)
        print(f"  validation on leading answers: codex == regex for {agree}/{len(rows)}")


if __name__ == "__main__":
    main()
