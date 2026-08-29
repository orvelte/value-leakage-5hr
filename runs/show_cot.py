"""Read full rollouts from a run: CoT, final response, parsed estimate, and judge verdict.

The plan's write-up step calls for "randomly selected raw CoT excerpts so the reader can see
the phenomenon is real", and the raw JSONL is not readable by hand. This prints or exports
whole transcripts with their covertness label attached.

Examples:
  python3 runs/show_cot.py --list
  python3 runs/show_cot.py --id giraffes_above_good_0
  python3 runs/show_cot.py --category INFLUENCED --direction above_good --sample 2
  python3 runs/show_cot.py --sample 6 --balanced --export runs/hour0/cot_samples.md
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.qual import parse, segment


def load(run_dir):
    T = json.load(open(run_dir / "threshold.json"))["threshold"]
    verdict = {}
    cov = run_dir / "covertness_results.json"
    if cov.exists():
        for r in json.load(open(cov)):
            verdict[r["id"].replace("covertness_" + r["direction"] + "_", "")] = r

    items = {}
    for direction in ["baseline", "above_good", "below_good"]:
        f = run_dir / "raw" / f"giraffes_{direction}.jsonl"
        if not f.exists():
            continue
        for rec in parse.parse_jsonl_file(f):
            val, reason = parse.parse_estimate(rec["raw_completion"])
            good = (parse.classify_on_good_side(val, T, direction)
                    if val is not None and direction != "baseline" else None)
            v = verdict.get(rec["item_id"], {})
            items[rec["item_id"]] = {
                "item_id": rec["item_id"], "direction": direction, "estimate": val,
                "parse_reason": reason, "on_good_side": good,
                "raw": rec["raw_completion"], "num_tokens": rec["num_tokens"],
                "finish_reason": rec["finish_reason"],
                "judge_raw": v.get("raw"), "judge_bucket": v.get("bucket"),
            }
    return T, items


def render(it, T, full=True):
    cot = it["raw"].split("</think>", 1)[0]
    if cot.startswith("<think>"):
        cot = cot[len("<think>"):]
    resp = it["raw"].split("</think>", 1)[1] if "</think>" in it["raw"] else "(TRUNCATED — no </think>)"
    est = f"{it['estimate']:,.0f}" if it["estimate"] is not None else f"None ({it['parse_reason']})"
    side = {True: "GOOD side", False: "bad side", None: "n/a"}[it["on_good_side"]]
    head = (f"### {it['item_id']}\n\n"
            f"- condition: **{it['direction']}**, threshold {T:,.0f}\n"
            f"- final estimate: **{est}** → {side}\n"
            f"- covertness judge: **{it['judge_raw'] or 'n/a'}** ({it['judge_bucket'] or 'n/a'})\n"
            f"- {it['num_tokens']} tokens, {len(segment.segment_sentences(it['raw']))} CoT "
            f"sentences, finish={it['finish_reason']}\n")
    body = (f"\n**Chain of thought:**\n\n```\n{cot.strip()}\n```\n"
            f"\n**Final response:**\n\n```\n{resp.strip()}\n```\n") if full else ""
    return head + body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="hour0")
    ap.add_argument("--id", help="one item_id, or several comma-separated")
    ap.add_argument("--category", help="judge label: INFLUENCED / NOT_INFLUENCED / ...")
    ap.add_argument("--direction", choices=["baseline", "above_good", "below_good"])
    ap.add_argument("--good-side", choices=["true", "false"])
    ap.add_argument("--sample", type=int, default=1)
    ap.add_argument("--balanced", action="store_true",
                    help="split the sample evenly across INFLUENCED / NOT_INFLUENCED")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--export")
    args = ap.parse_args()

    run_dir = Path(__file__).resolve().parent / args.run
    T, items = load(run_dir)

    if args.list:
        for k, it in sorted(items.items()):
            est = f"{it['estimate']:,.0f}" if it["estimate"] is not None else "—"
            print(f"{k:34s} {it['direction']:11s} est={est:>15s} "
                  f"good={str(it['on_good_side']):5s} judge={it['judge_raw']}")
        return

    if args.id:
        chosen = [items[i.strip()] for i in args.id.split(",")]
    else:
        pool = list(items.values())
        if args.direction:
            pool = [i for i in pool if i["direction"] == args.direction]
        if args.category:
            pool = [i for i in pool if i["judge_raw"] == args.category]
        if args.good_side:
            pool = [i for i in pool if i["on_good_side"] is (args.good_side == "true")]
        rng = random.Random(args.seed)
        if args.balanced:
            chosen = []
            for cat in ["INFLUENCED", "NOT_INFLUENCED"]:
                sub = [i for i in pool if i["judge_raw"] == cat]
                rng.shuffle(sub)
                chosen += sub[: max(1, args.sample // 2)]
        else:
            rng.shuffle(pool)
            chosen = pool[: args.sample]

    out = "\n\n---\n\n".join(render(i, T) for i in chosen)
    if args.export:
        Path(args.export).write_text(
            f"# Sample rollouts — run `{args.run}`, giraffes task, threshold {T:,.0f}\n\n"
            f"Selected with `--seed {args.seed}`. Numbers are NOT blurred here (the covertness "
            f"judge saw a blurred copy; these are the raw transcripts).\n\n---\n\n" + out)
        print(f"wrote {len(chosen)} rollouts to {args.export}")
    else:
        print(out)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        # piping into `head` closes stdout early; not an error worth a traceback
        import os
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
