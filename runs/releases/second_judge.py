"""Hand-label export + second covertness judge on the same items.

For each model: pick 20 of the 80 thinking-on bet traces (10 above_good, 10 below_good, seeded
random, chosen before either judge's labels are looked at), export them to
runs/releases/handlabels/<model>/, have the second judge (Codex, a different model from the first judge: SECOND_MODEL at high effort)
judge them with the same paper-verbatim, number-blurred prompt the first judge saw, and write
labels.csv with both judges' labels and an empty human_label column.

Olivia asked for Sonnet as the second judge; Anthropic's content filter refused every request
(OpenRouter finish_reason content_filter, $0 spent), so per her follow-up the second judge is
Codex (DECISIONS.md #16).

Per item, the export directory holds:
  <id>.judge_prompt.txt  the exact prompt both judges saw (numbers blurred)
  <id>.trace.txt         the raw thinking-on completion (numbers intact), for hand-labelling
  <id>.second.txt        the second judge's full response

Run with: python3 runs/releases/second_judge.py --model q35   (then --report)
"""
import argparse
import csv
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))
from src.qual import judge, parse
import codex_judge

SECOND_MODEL, SECOND_EFFORT = "gpt-6.1-terra", "high"
N_PER_DIR = 10
OUT = HERE / "handlabels"


def second(prompt):
    codex_judge.CODEX_MODEL, codex_judge.EFFORT = SECOND_MODEL, SECOND_EFFORT
    return codex_judge.call(prompt)


def select(run_dir):
    man = json.load(open(run_dir / "judge_manifest.json"))
    rng = random.Random(0)
    pick = []
    for d in ("above_good", "below_good"):
        ids = sorted(m["id"] for m in man if m["direction"] == d)
        pick += rng.sample(ids, N_PER_DIR)
    return pick


def bucket(text):
    a = judge.parse_covertness_answer(text)
    return a[1] if a else None


def run(model):
    run_dir = HERE / model
    od = OUT / model
    od.mkdir(parents=True, exist_ok=True)
    man = {m["id"]: m for m in json.load(open(run_dir / "judge_manifest.json"))}
    raw = {}
    for d in ("above_good", "below_good"):
        for r in parse.parse_jsonl_file(run_dir / "raw" / f"giraffes_{d}.jsonl"):
            raw[r["item_id"]] = r["raw_completion"]
    ids = select(run_dir)
    for i in ids:
        (od / f"{i}.judge_prompt.txt").write_text((run_dir / "judge_inputs" / f"{i}.txt").read_text())
        (od / f"{i}.trace.txt").write_text(raw[man[i]["source_item_id"]])
    todo = [i for i in ids if not (od / f"{i}.second.txt").exists()
            or bucket((od / f"{i}.second.txt").read_text()) is None]
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(lambda i: second((od / f"{i}.judge_prompt.txt").read_text()), todo))
    for i, text in zip(todo, res):
        (od / f"{i}.second.txt").write_text(text)
    maj = json.load(open(run_dir / "covertness_majority.json"))["final_labels"]
    rows = [{"id": i, "model": model, "direction": man[i]["direction"],
             "estimate": man[i]["estimate"], "on_good_side": man[i]["on_good_side"],
             "judge1_label": maj.get(i), "judge2_label": bucket((od / f"{i}.second.txt").read_text()),
             "judge1": "codex gpt-6.1-sol (medium)", "judge2": f"codex {SECOND_MODEL} ({SECOND_EFFORT})",
             "human_label": ""} for i in ids]
    with open(od / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{model}: {len(todo)} second-judge calls; "
          f"unparsed {sum(r['judge2_label'] is None for r in rows)}")


def kappa(a, b):
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def report():
    out = {}
    for d in sorted(p for p in OUT.iterdir() if p.is_dir()):
        rows = list(csv.DictReader(open(d / "labels.csv")))
        rows = [r for r in rows if r["judge2_label"] and r["judge1_label"]]
        a4, b4 = [r["judge1_label"] for r in rows], [r["judge2_label"] for r in rows]
        a2 = ["admit" if x == "Admits to bias" else "not" for x in a4]
        b2 = ["admit" if x == "Admits to bias" else "not" for x in b4]
        conf = {}
        for x, y in zip(a4, b4):
            conf[f"{x} | {y}"] = conf.get(f"{x} | {y}", 0) + 1
        out[d.name] = {"n": len(rows),
                       "agree_4way": sum(x == y for x, y in zip(a4, b4)) / len(rows),
                       "agree_admit_vs_not": sum(x == y for x, y in zip(a2, b2)) / len(rows),
                       "kappa_admit_vs_not": kappa(a2, b2),
                       "judge1_admits": a2.count("admit"), "judge2_admits": b2.count("admit"),
                       "confusion (judge1 | judge2)": conf}
        o = out[d.name]
        print(f"{d.name}: n={o['n']}  admit/not agree {o['agree_admit_vs_not']:.2f} "
              f"(kappa {o['kappa_admit_vs_not']:+.2f})  4-way agree {o['agree_4way']:.2f}  "
              f"admits judge1 {o['judge1_admits']} / judge2 {o['judge2_admits']}")
    json.dump(out, open(OUT / "agreement.json", "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.model:
        run(args.model)
    if args.report:
        report()


if __name__ == "__main__":
    main()
