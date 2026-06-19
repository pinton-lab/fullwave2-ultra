#!/usr/bin/env bash
# 3D solver-correctness gate: the optimized 3D kernel (bin/bench_3d_opt) must agree
# with the 3D base oracle (bin/bench_3d) on a tiny synthetic run.
#
# IMPORTANT -- this gate uses a TOLERANCE, by design, NOT a byte cmp:
#   3D opt-vs-base is float32-EQUIVALENT (relative ~2e-7), NOT bit-identical,
#   because the optimized 3D kernels RESTRUCTURE the field accumulation (different
#   summation order -> float32 rounding differs). This is unlike the 2D gate, where
#   batch==solo IS bit-exact because those kernels are shared. So here we assert
#   max|base-opt| / ptp(base) < 1e-4, both finite, and identical output sizes.
#
# The 3D fixture is NOT committed (the volumes are larger than the tracked 2D one);
# it is regenerated on demand via examples/tiny_3d/make_fixture_3d.py, which needs
# a working `python -c "import fullwave2_ultra"`. We exercise BOTH the default fixture
# (constant-memory path) and a --bigndmap fixture (widened c so the optimized solver
# takes its global-memory fallback path).
#
# Skips cleanly (exit 0) if either bin is missing, python can't import the package,
# or no CUDA GPU is visible. Exits 0 on PASS or SKIP, 1 on FAIL.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE_BIN="$ROOT/bin/bench_3d"; OPT_BIN="$ROOT/bin/bench_3d_opt"
GEN="$ROOT/examples/tiny_3d/make_fixture_3d.py"
export CUDA_VISIBLE_DEVICES=${FW2U_GPU:-${FW2B_GPU:-0}}
PY=${PYTHON:-python}

[ -x "$BASE_BIN" ] && [ -x "$OPT_BIN" ] || { echo "SKIP: need bin/bench_3d + bin/bench_3d_opt -- place/fetch them in bin/ (see README)"; exit 0; }
[ -f "$GEN" ] || { echo "SKIP: missing $GEN"; exit 0; }
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" -c "import fullwave2_ultra" 2>/dev/null \
  || { echo "SKIP: python cannot import fullwave2_ultra"; exit 0; }
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 \
  || { echo "SKIP: no CUDA GPU visible"; exit 0; }

work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
fail=0

# compare(label, fixture_dir) -- run base + opt from inside copies, fp-tolerance check.
compare(){
  local label="$1" fdir="$2"
  cp -r "$fdir" "$work/${label}_base"; cp -r "$fdir" "$work/${label}_opt"
  ( cd "$work/${label}_base" && rm -f genout*.dat && "$BASE_BIN" >stdout.log 2>err.log )
  ( cd "$work/${label}_opt"  && rm -f genout*.dat && "$OPT_BIN"  >stdout.log 2>err.log )
  if PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" - "$work/${label}_base/genout.dat" "$work/${label}_opt/genout.dat" <<'PYEOF'
import sys, os
import numpy as np
bp, op = sys.argv[1], sys.argv[2]
if not (os.path.isfile(bp) and os.path.isfile(op)):
    print("  missing genout (base=%s opt=%s)" % (os.path.isfile(bp), os.path.isfile(op))); sys.exit(2)
b = np.fromfile(bp, dtype=np.float32); o = np.fromfile(op, dtype=np.float32)
if b.size == 0 or b.size != o.size:
    print("  size mismatch base=%d opt=%d" % (b.size, o.size)); sys.exit(2)
if not (np.isfinite(b).all() and np.isfinite(o).all()):
    print("  non-finite values present"); sys.exit(2)
ptp = float(np.ptp(b))
if ptp == 0.0:
    print("  base is flat (ptp==0) -- degenerate run"); sys.exit(2)
rel = float(np.max(np.abs(b - o))) / ptp
print("  size=%d ptp=%.6g maxabsdiff=%.6g rel=%.3e tol=1e-4" % (b.size, ptp, float(np.max(np.abs(b - o))), rel))
sys.exit(0 if rel < 1e-4 else 3)
PYEOF
  then echo "OK  $label: opt within float32 tolerance of base"; else echo "FAIL $label"; fail=1; fi
}

echo "[3d-gate] regen default fixture (small ndmap -> constant-memory path)"
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" "$GEN" "$work/F" >/dev/null
compare F "$work/F"

echo "[3d-gate] regen --bigndmap fixture (widened c -> opt global-memory path)"
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" "$GEN" "$work/G" --bigndmap >/dev/null
compare G "$work/G"

[ $fail -eq 0 ] && { echo "PASS 3d opt==base (float32-equivalent)"; exit 0; } || { echo "FAIL 3d opt!=base"; exit 1; }
