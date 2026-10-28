"""Run the paper-verbatim judge prompts through Codex (`codex exec`), one fresh ephemeral call per
item, and write outputs where the original analysis scripts expect them.

Replaces the Claude-subagent judges (DECISIONS.md #12-13). Each call gets the whole judge
prompt on stdin, so the judge always sees the full trace; no agentic reading, no excerpts.
Calls are independent (ephemeral sessions), so passes 1 and 2 are independent samples.

  covertness pass1 / pass2 : every item in judge_manifest.json
  covertness pass3         : only items where pass1 and pass2 disagree (hour0 procedure)
  trajectory               : every item in trajectory_manifest.json

Run with: python3 runs/releases/codex_judge.py --model q35 --stage pass1|pass2|pass3|trajectory
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from src.qual import judge

CODEX_MODEL = "gpt-6.1-sol"
EFFORT = "medium"
OUT = {"pass1": "judge_outputs", "pass2": "judge_outputs_pass2", "pass3": "judge_outputs_pass3",
       "trajectory": "trajectory_outputs"}


def call(prompt_text, retries=3):
    for k in range(retries):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.txt"
            r = subprocess.run(["codex", "exec", "--ephemeral", "--skip-git-repo-check",
                                "-s", "read-only", "-C", td, "-m", CODEX_MODEL,
                                "-c", f"model_reasoning_effort={EFFORT}", "-o", str(out), "-"],
                               input=prompt_text, capture_output=True, text=True, timeout=600)
            if r.returncode == 0 and out.exists() and out.read_text().strip():
                return out.read_text()
        time.sleep(5 * (k + 1))
    raise RuntimeError(f"codex failed after {retries} tries: {r.stderr[-300:]}")


def ok(stage, text):
    if stage == "trajectory":
        return judge.parse_trajectory_answer(text) is not None
    return judge.parse_covertness_answer(text) is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stage", required=True, choices=sorted(OUT))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    run_dir = HERE / args.model
    out_dir = run_dir / OUT[args.stage]
    out_dir.mkdir(exist_ok=True)

    if args.stage == "trajectory":
        in_dir = run_dir / "trajectory_inputs"
        ids = [m["id"] for m in json.load(open(run_dir / "trajectory_manifest.json"))]
    else:
        in_dir = run_dir / "judge_inputs"
        ids = [m["id"] for m in json.load(open(run_dir / "judge_manifest.json"))]
        if args.stage == "pass3":
            def lab(d, i):
                f = run_dir / d / f"{i}.txt"
                a = judge.parse_covertness_answer(f.read_text()) if f.exists() else None
                return a[1] if a else None
            ids = [i for i in ids if lab(OUT["pass1"], i) != lab(OUT["pass2"], i)]
            print(f"pass3: {len(ids)} items where pass1 and pass2 disagree")

    todo = [i for i in ids if not ((out_dir / f"{i}.txt").exists()
                                   and ok(args.stage, (out_dir / f"{i}.txt").read_text()))]
    print(f"{args.model} {args.stage}: {len(todo)} to judge ({len(ids) - len(todo)} done)", flush=True)

    def one(i):
        text = call((in_dir / f"{i}.txt").read_text())
        if not ok(args.stage, text):  # one re-ask on an unparseable answer, then keep what we got
            text = call((in_dir / f"{i}.txt").read_text())
        (out_dir / f"{i}.txt").write_text(text)
        return ok(args.stage, text)

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        res = list(ex.map(one, todo))
    cfg = run_dir / "judge_config.json"
    c = json.load(open(cfg)) if cfg.exists() else {}
    c[args.stage] = {"judge": "codex exec", "model": CODEX_MODEL, "reasoning_effort": EFFORT,
                     "n_judged": len(todo), "n_unparsed": res.count(False),
                     "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(c, open(cfg, "w"), indent=2)
    print(f"done in {(time.time()-t0)/60:.1f} min; unparsed {res.count(False)}", flush=True)


if __name__ == "__main__":
    main()
