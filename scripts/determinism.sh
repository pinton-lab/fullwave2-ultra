#!/usr/bin/env bash
# Determinism gate, SELF-CONTAINED on the committed tiny_nsims4 fixture:
#   (1) run-to-run: run bin/bench_2d_batch on the fixture TWICE -> assert every
#       per-sim genout is bit-identical between the two runs.
#   (2) width-invariance: solo-run element 1 (icmat sliced to its block, nsims=1)
#       and confirm it equals the batch's sim-0 genout -- the per-sim output must
#       not depend on the batch width (reuses the batch==solo idea).
#
# Unlike the original fw2b determinism.sh (which marshalled an abdominal nsims=16
# medium via scripts/marshal_local.sh), this is fully self-contained on the tiny
# synthetic fixture -- marshal_local.sh is abdominal-specific and intentionally
# NOT ported. To exercise a real >4-element medium, see scripts/scale128.sh.
#
# Paths come from FW2U_* env (FW2B_* accepted as fallback). Skips cleanly (exit 0)
# if the bins are not built or the fixture is absent. Summary -> results/determinism.log.
set -eo pipefail
BASE=${FW2U_ROOT:-${FW2B_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
RUNROOT=${FW2U_RUN_DIR:-${FW2B_RUN_DIR:-$BASE/runs}}
GPU=${FW2U_GPU:-${FW2B_GPU:-0}}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU}

BATCH="$BASE/bin/bench_2d_batch"; SOLO="$BASE/bin/bench_2d_aexp"
FIX="$BASE/examples/tiny_nsims4"
mkdir -p "$BASE/results"
LOG=$BASE/results/determinism.log
: > "$LOG"
say(){ echo "$@" | tee -a "$LOG"; }

if ! { [ -x "$BATCH" ] && [ -x "$SOLO" ]; }; then say "SKIP: solver binary not found -- place/fetch it in bin/ (see README)"; exit 0; fi
if [ ! -d "$FIX" ]; then say "SKIP: missing fixture $FIX"; exit 0; fi

rdint(){ od -An -td4 -N4 "$1" | tr -d ' '; }
nsims=$(rdint "$FIX/nsims.dat")
ncoords=$(rdint "$FIX/ncoords.dat"); nTic=$(rdint "$FIX/nTic.dat")
blk=$((ncoords*nTic*4))

work="$RUNROOT/determinism"; rm -rf "$work"; mkdir -p "$work"
trap 'rm -rf "$work"' EXIT

say "[determinism] fixture nsims=$nsims ncoords=$ncoords nTic=$nTic (self-contained, tiny_nsims4)"

# (1) run-to-run bit identity ------------------------------------------------
cp -r "$FIX" "$work/run1"; cp -r "$FIX" "$work/run2"
say "[determinism] batch run #1"
( cd "$work/run1" && rm -f genout*.dat && "$BATCH" >stdout.log 2>err.log )
say "[determinism] batch run #2"
( cd "$work/run2" && rm -f genout*.dat && "$BATCH" >stdout.log 2>err.log )

say "[determinism] cmp run1 vs run2 ($nsims files)"
fail=0
for ((s=0; s<nsims; s++)); do
  [ $s -eq 0 ] && f=genout.dat || f=genout_s${s}.dat
  if cmp -s "$work/run1/$f" "$work/run2/$f"; then
    say "  OK  $f run1==run2"
  else
    say "  FAIL $f run1!=run2"; fail=1
  fi
done
[ $fail -eq 0 ] && say "PASS determinism: nsims=$nsims run-to-run bit-identical" || say "FAIL determinism"

# (2) width-invariance: solo elem 1 == batch sim-0 ---------------------------
say "[width-invariance] solo-run element 1 (nsims=1) vs batch sim-0"
cp -r "$FIX" "$work/S1"
dd if="$FIX/icmat.dat" of="$work/S1/icmat.dat" bs="$blk" skip=0 count=1 status=none
printf '\x01\x00\x00\x00' > "$work/S1/nsims.dat"   # nsims=1
( cd "$work/S1" && rm -f genout*.dat && "$SOLO" >stdout.log 2>err.log )
wfail=0
if cmp -s "$work/run1/genout.dat" "$work/S1/genout.dat"; then
  say "  OK  sim 0: batch == solo (width-independent)"
else
  say "  FAIL sim 0: batch != solo"; wfail=1
fi
[ $wfail -eq 0 ] && say "PASS width-invariance: per-sim output independent of batch width" || say "FAIL width-invariance"

say "[determinism] DONE"
[ $((fail+wfail)) -eq 0 ] || exit 1
