#!/usr/bin/env bash
# PRIMARY 2D correctness gate: each batched sim's genout is byte-identical to a
# SOLO run of that same element. Requires bin/bench_2d_batch + bin/bench_2d_aexp
# (the 2D solo oracle) and the committed nsims=4 fixture at examples/tiny_nsims4.
# Skips cleanly (exit 0) if the bins are not built or the fixture is absent.
#
# 2D batch==solo is BIT-EXACT on GPU (the batch and solo kernels are shared), so
# this gate asserts byte identity -- unlike the 3D gate, which uses a tolerance.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BATCH="$ROOT/bin/bench_2d_batch"; SOLO="$ROOT/bin/bench_2d_aexp"
FIX="$ROOT/examples/tiny_nsims4"
export CUDA_VISIBLE_DEVICES=${FW2U_GPU:-${FW2B_GPU:-0}}
[ -x "$BATCH" ] && [ -x "$SOLO" ] || { echo "SKIP: solver binary not found -- place/fetch it in bin/ (see README)"; exit 0; }
[ -d "$FIX" ] || { echo "SKIP: missing fixture $FIX (regen via examples/tiny_nsims4/make_fixture.py)"; exit 0; }

work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
cp -r "$FIX" "$work/B"; ( cd "$work/B" && rm -f genout*.dat && "$BATCH" >/dev/null )
rdint(){ od -An -td4 -N4 "$1" | tr -d ' '; }
nsims=$(rdint "$FIX/nsims.dat")
# combined icmat is sim-major: nsims blocks of (ncoords*nTic) floats. Slice element
# e's block for the solo run (block bytes = ncoords*nTic*4, offset = (e-1)*block);
# do NOT mask a missing/short icmat -- a slice failure must fail the gate.
ncoords=$(rdint "$FIX/ncoords.dat"); nTic=$(rdint "$FIX/nTic.dat")
blk=$((ncoords*nTic*4))
fail=0
for ((e=1; e<=nsims; e++)); do
  s=$((e-1)); cp -r "$FIX" "$work/S$e"
  dd if="$FIX/icmat.dat" of="$work/S$e/icmat.dat" bs="$blk" skip="$s" count=1 status=none
  printf '\x01\x00\x00\x00' > "$work/S$e/nsims.dat"   # nsims=1
  ( cd "$work/S$e" && rm -f genout*.dat && "$SOLO" >/dev/null )
  bf=$([ $s -eq 0 ] && echo genout.dat || echo "genout_s$s.dat")
  if cmp -s "$work/B/$bf" "$work/S$e/genout.dat"; then echo "OK  elem $e"; else echo "FAIL elem $e"; fail=1; fi
done
[ $fail -eq 0 ] && echo "PASS batch==solo" || { echo "FAIL"; exit 1; }
