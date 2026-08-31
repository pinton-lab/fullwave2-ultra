#!/usr/bin/env bash
# GPU gate for the 3D additive channel's SOURCE CONSTANTS: a one-cell sheet
# radiates s/(2 CFL) to each side (the injection-shell case), and a single cell
# radiates the monopole field C s'(t - r/c)/r with C = dx^3/(4 pi c^2 dT) --
# 1/r decay, retarded time, derivative of the drive. Physics and derivation in
# tests/check_addsrc_greens.py. Requires bin/bench_3d_opt (FW2U_BIN_DIR to
# override). Skips cleanly (exit 0) without the binary or a GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BINDIR="${FW2U_BIN_DIR:-$ROOT/bin}"
OPT="$BINDIR/bench_3d_opt"
export CUDA_VISIBLE_DEVICES=${FW2U_GPU:-${FW2B_GPU:-0}}
[ -x "$OPT" ] || { echo "SKIP: bin/bench_3d_opt not built (make 3d)"; exit 0; }
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 || { echo "SKIP: no usable GPU"; exit 0; }
PYTHONPATH="$ROOT" "${PYTHON:-python3}" "$ROOT/tests/check_addsrc_greens.py" "$OPT"
