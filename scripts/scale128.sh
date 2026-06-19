#!/usr/bin/env bash
# Full-scale 2D correctness spot-check: run a large (nsims=128) batch in ONE fused
# launch with the correctness binary (bin/bench_2d_batch), then cmp a representative
# set of its per-sim genout against fresh CUDA-solo runs (bin/bench_2d_aexp) of the
# same elements. PASS iff every spot-checked sim is bit-identical to its solo run.
#
# GATED behind env -- this needs a REAL 128-element batch input dir, which the
# standalone package does NOT generate: the committed synthetic fixture is only
# nsims=4, and the original 128-element medium came from an abdominal marshal step
# (scripts/marshal_local.sh) that is abdominal-specific and INTENTIONALLY NOT ported.
# Provide a pre-populated batch run dir one of these ways (first match wins):
#   FW2U_RUN_DIR/B128            -- a ready batch input dir with nsims=128 + icmat
#   FW2U_DATA_DIR/B128           -- same, under the data dir
# and (for the solo spot-checks) one of:
#   FW2U_RUN_DIR/Ss   or  FW2U_DATA_DIR/Ss   -- a scratch dir holding per-element
#       solo inputs named S<e> (e=1..128); each S<e> is a complete nsims=1 input dir.
# If a B128 dir is found but no solo Ss dirs, the script will SLICE solo inputs out
# of B128's combined icmat itself (icmat is sim-major: nsims blocks of ncoords*nTic
# floats) -- so a single B128 dir is sufficient for the full spot-check.
#
# Skips cleanly (exit 0) when no 128-element input is available. Summary -> results/scale128.log.
set -eo pipefail
BASE=${FW2U_ROOT:-${FW2B_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
RUNROOT=${FW2U_RUN_DIR:-${FW2B_RUN_DIR:-$BASE/runs}}
DATADIR=${FW2U_DATA_DIR:-${FW2B_DATA_DIR:-}}
GPU=${FW2U_GPU:-${FW2B_GPU:-0}}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU}
ELEMS=${ELEMS:-"1 2 3 4 8 16 32 48 64 80 96 112 120 127 128"}

BATCH="$BASE/bin/bench_2d_batch"; SOLO="$BASE/bin/bench_2d_aexp"
mkdir -p "$BASE/results"
LOG=$BASE/results/scale128.log
: > "$LOG"; say(){ echo "$@" | tee -a "$LOG"; }

if ! { [ -x "$BATCH" ] && [ -x "$SOLO" ]; }; then say "SKIP: solver binary not found -- place/fetch it in bin/ (see README)"; exit 0; fi

# locate a 128-element batch input dir
B128=""
for cand in "$RUNROOT/B128" "${DATADIR:+$DATADIR/B128}"; do
  [ -n "$cand" ] && [ -d "$cand" ] && [ -f "$cand/icmat.dat" ] && [ -f "$cand/nsims.dat" ] && { B128="$cand"; break; }
done
if [ -z "$B128" ]; then
  say "SKIP: no 128-element batch input found."
  say "      The standalone synthetic fixture (examples/tiny_nsims4) is nsims=4; the"
  say "      abdominal 128-element marshal (scripts/marshal_local.sh) is intentionally"
  say "      not ported. Provide \$FW2U_RUN_DIR/B128 (or \$FW2U_DATA_DIR/B128) with a"
  say "      nsims=128 icmat to run this gate. See tests/test_batch_eq_solo.sh for the"
  say "      same batch==solo logic on the committed nsims=4 fixture."
  exit 0
fi

rdint(){ od -An -td4 -N4 "$1" | tr -d ' '; }
nsims=$(rdint "$B128/nsims.dat")
ncoords=$(rdint "$B128/ncoords.dat"); nTic=$(rdint "$B128/nTic.dat")
blk=$((ncoords*nTic*4))
say "[scale128] using batch input $B128 (nsims=$nsims ncoords=$ncoords nTic=$nTic)"

work="$RUNROOT/scale128"; rm -rf "$work"; mkdir -p "$work"
trap 'rm -rf "$work"' EXIT

say "[scale128] run nsims=$nsims batch (correctness binary, full nT)"
cp -r "$B128" "$work/B"; ( cd "$work/B" && rm -f genout*.dat && "$BATCH" >stdout.log 2>err.log )
nfiles=$(ls "$work/B"/genout*.dat 2>/dev/null | wc -l)
say "[scale128] batch produced $nfiles genout files (expect $nsims)"

# optional pre-populated solo input dir (S<e> subdirs); else slice from B128 icmat.
SOLODIR=""
for cand in "$RUNROOT/Ss" "${DATADIR:+$DATADIR/Ss}"; do
  [ -n "$cand" ] && [ -d "$cand" ] && { SOLODIR="$cand"; break; }
done

[ "$ELEMS" = "all" ] && ELEMS=$(seq 1 "$nsims")
say "[scale128] cmp spot-check elements: $ELEMS"
fail=0; ntot=0
for e in $ELEMS; do
  s=$((e-1))
  Sdir="$work/S$e"
  if [ -n "$SOLODIR" ] && [ -d "$SOLODIR/S$e" ]; then
    cp -r "$SOLODIR/S$e" "$Sdir"
  else
    cp -r "$B128" "$Sdir"
    dd if="$B128/icmat.dat" of="$Sdir/icmat.dat" bs="$blk" skip="$s" count=1 status=none
    printf '\x01\x00\x00\x00' > "$Sdir/nsims.dat"   # nsims=1
  fi
  ( cd "$Sdir" && rm -f genout*.dat && "$SOLO" >stdout.log 2>err.log )
  [ $s -eq 0 ] && bf=genout.dat || bf=genout_s${s}.dat
  if cmp -s "$work/B/$bf" "$Sdir/genout.dat"; then say "  OK  elem $e (sim $s) bit-identical"
  else say "  FAIL elem $e (sim $s)"; fail=1; fi
  rm -rf "$Sdir"
  ntot=$((ntot+1))
done
[ $fail -eq 0 ] && say "PASS scale128: $ntot/$ntot spot-checked sims bit-identical to solo (of $nsims in one batch)" \
                || say "FAIL scale128"
say "[scale128] DONE"
[ $fail -eq 0 ] || exit 1
