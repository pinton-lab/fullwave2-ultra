#!/usr/bin/env bash
# GPU gate for the 2D additive source channel (icc_add/icmat_add/nTic_add):
# superposition, 2D Green's-function decay/arrival, nTic_add truncation, and
# batch==solo byte identity with additive traces. Physics in
# tests/check_additive.py. Requires bin/bench_2d_batch + bin/bench_2d_aexp
# (override the directory with FW2U_BIN_DIR for freshly built binaries).
# Skips cleanly (exit 0) if the bins are not built or no GPU is present.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BINDIR="${FW2U_BIN_DIR:-$ROOT/bin}"
BATCH="$BINDIR/bench_2d_batch"; SOLO="$BINDIR/bench_2d_aexp"
export CUDA_VISIBLE_DEVICES=${FW2U_GPU:-${FW2B_GPU:-0}}
[ -x "$BATCH" ] && [ -x "$SOLO" ] || { echo "SKIP: solver binary not found -- place/fetch it in bin/ (see README)"; exit 0; }
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 || { echo "SKIP: no usable GPU"; exit 0; }
PYTHONPATH="$ROOT" "${PYTHON:-python3}" "$ROOT/tests/check_additive.py" "$BATCH" "$SOLO"
