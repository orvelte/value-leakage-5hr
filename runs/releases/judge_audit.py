"""Audit judge outputs: did the judge read the whole trace, and does the answer parse?

Each covertness output must end with a line `TRACE_END: <text>` (after the <answer> tag, which
parse_covertness_answer finds anywhere). Trajectory outputs must stay a bare integer list, because
the original analysis scripts parse the whole file, so their TRACE_END line goes in a sidecar file,
trajectory_audit/<id>.txt. In both cases <text> is the last
~150 characters (at least 100) of the text inside the input's
<reasoning_trace> (covertness) or <text> (trajectory) block. A judge that skimmed or grepped
excerpts cannot produce this reliably. Outputs that fail are listed and should be deleted and
re-judged; they never enter judge_majority.py.

Run with: python3 runs/releases/judge_audit.py --model q35 --kind covertness [--delete-failed]
"""
import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from src.qual import judge

BLOCK = {"covertness": ("<reasoning_trace>", "</reasoning_trace>"), "trajectory": ("<text>", "</text>")}
DIRS = {"covertness": ["judge_outputs", "judge_outputs_pass2", "judge_outputs_pass3"],
        "trajectory": ["trajectory_outputs"]}
INPUTS = {"covertness": "judge_inputs", "trajectory": "trajectory_inputs"}


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def last_line(input_text, kind):
    a, b = BLOCK[kind]
    body = input_text.rsplit(a, 1)[-1].split(b, 1)[0]
    return norm(body)[-400:]


def check(out_text, input_text, kind):
    m = re.search(r"TRACE_END:\s*(.+?)\s*$", out_text.strip(), re.S)
    if not m:
        return "no TRACE_END"
    got = norm(m.group(1)).strip('"\'`')
    want = last_line(input_text, "trajectory" if kind == "trajectory_sidecar" else kind)
    if not got or got not in want or len(got) < min(100, len(want)):
        return "TRACE_END mismatch"
    if kind == "covertness" and judge.parse_covertness_answer(out_text) is None:
        return "answer unparsed"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--kind", default="covertness", choices=sorted(BLOCK))
    ap.add_argument("--delete-failed", action="store_true")
    args = ap.parse_args()
    run_dir = HERE / args.model
    for d in DIRS[args.kind]:
        od = run_dir / d
        if not od.exists():
            continue
        outs = sorted(od.glob("*.txt"))
        bad = {}
        for p in outs:
            inp = (run_dir / INPUTS[args.kind] / p.name).read_text()
            if args.kind == "trajectory":
                side = run_dir / "trajectory_audit" / p.name
                if judge.parse_trajectory_answer(p.read_text()) is None:
                    err = "answer unparsed"
                elif not side.exists():
                    err = "no audit sidecar"
                else:
                    err = check(side.read_text(), inp, "trajectory_sidecar")
            else:
                err = check(p.read_text(), inp, args.kind)
            if err:
                bad[p.stem] = err
        print(f"{d}: {len(outs)} outputs, {len(outs) - len(bad)} pass, {len(bad)} fail")
        for k, v in bad.items():
            print(f"  FAIL {k}: {v}")
            if args.delete_failed:
                (od / f"{k}.txt").unlink()


if __name__ == "__main__":
    main()
