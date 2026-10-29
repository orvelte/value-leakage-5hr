#!/usr/bin/env bash
# Everything after generation for one model, in order. Local only (Codex for judges/extraction).
#   bash runs/releases/postprocess.sh q38
set -euo pipefail
M="$1"
cd "$(dirname "$0")/.."            # runs/
PY="${PY:-$HOME/personawhitebox/.venv/bin/python}"
R=releases/$M

echo "== extraction (non-leading answers), then behaviour"
$PY releases/extract.py --model "$M" --validate 60
$PY releases/analyze.py --model "$M"
$PY releases/analyze.py --model "$M" --regex-only > /dev/null

echo "== covertness judge: pass 1 + 2 in parallel, pass 3 on disagreements"
$PY prep_covertness.py --run "$R" --seed 0 > /dev/null
$PY prep_covertness.py --run "$R" --pass-name pass2 --seed 1 > /dev/null
$PY releases/codex_judge.py --model "$M" --stage pass1 --workers 6 &
$PY releases/codex_judge.py --model "$M" --stage pass2 --workers 6 &
wait
$PY releases/codex_judge.py --model "$M" --stage pass3 --workers 6
$PY analyze_covertness.py --run "$R" > /dev/null
$PY releases/judge_majority.py --model "$M"

echo "== trajectories + revision analysis"
$PY prep_trajectory.py --run "$R" --conditions baseline,above_good,below_good > /dev/null
$PY releases/codex_judge.py --model "$M" --stage trajectory --workers 10
$PY releases/lib/analyze_revision.py --run "$R" | grep -E "on good side|currently BAD" | head -4

echo "== second judge on 20 traces"
$PY releases/second_judge.py --model "$M" --report

echo "== extraction sensitivity (Codex on every answer)"
$PY releases/extract.py --model "$M" --all
$PY releases/analyze.py --model "$M" --codex-all | grep -E "_off|_on "

echo "== collect + figures"
$PY releases/collect.py
$PY releases/make_figures.py
