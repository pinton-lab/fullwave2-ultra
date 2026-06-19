#!/usr/bin/env bash
# Regression for the OPT6 per-sim sbase fix: batched OPT6 sims must DIFFER. The
# unfixed bug made every sim read sim-0's field offset -> all genout identical.
# Requires bin/bench_2d_batch_opt6 + the committed nsims=4 fixture. Skips cleanly
# (exit 0) if the bin is not built or the fixture is absent.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/bin/bench_2d_batch_opt6"; FIX="$ROOT/examples/tiny_nsims4"
export CUDA_VISIBLE_DEVICES=${FW2U_GPU:-${FW2B_GPU:-0}}
[ -x "$BIN" ] && [ -d "$FIX" ] || { echo "SKIP: need bin/bench_2d_batch_opt6 + fixture"; exit 0; }
work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
cp -r "$FIX" "$work/B"; ( cd "$work/B" && rm -f genout*.dat && "$BIN" >/dev/null )
if cmp -s "$work/B/genout.dat" "$work/B/genout_s1.dat"; then
  echo "FAIL: sim0 == sim1 (OPT6 sbase missing?)"; exit 1
else echo "PASS: OPT6 batched sims differ"; fi
