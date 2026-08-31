#!/usr/bin/env bash
# 3D ADDITIVE SOURCE gate (icc_add / icmat_add) -- GPU.
#
# The additive channel superimposes (p += trace) instead of clamping, so a dense
# source region stays TRANSPARENT to waves crossing it. That is what makes it
# usable for distributed secondary-source injection (a hybrid RS->FDTD injection
# shell, scatterer coupling), where a Dirichlet region would act as a rigid
# screen and re-reflect everything the medium sends back at it.
#
# Cases, on both the base oracle (bench_3d) and the production solver
# (bench_3d_opt), and on their opt-vs-base agreement:
#
#   1 TRANSPARENCY   a dense additive slab driven with ZEROS leaves the field of
#                    a hard source untouched -- byte-identical genout -- while
#                    the SAME cells as a Dirichlet slab are opaque (a 3-cell
#                    p=0 screen is pressure-release, |R| -> 1). Asserting both
#                    is what makes this a physics claim and not just "adding
#                    zero changes nothing".
#   2 SUPERPOSITION  genout(A) + genout(B) == genout(A u B) to fp32 round-off,
#                    on nmap=0 (linear) decks. These decks are ncoords=0, so
#                    they also cover the additive-ONLY deck shape the hybrid
#                    split needs -- no dummy hard cell, which would be a rigid
#                    point scatterer.
#   3 ORDER          an additive coord colocated with a hard coord adds ON TOP
#                    of the clamp: p = icmat[n] + T[n]. Asserted BYTE-exactly
#                    (the traces are chosen float32-exact, so no double-rounding
#                    slack), in both directions -- coloc_add must equal the
#                    folded hard source and must NOT equal the un-added one.
#   4 WINDOWING      icmat_add is TIME-MAJOR in 3D so the solver can hold a
#                    rolling window of slices instead of the whole matrix
#                    (docs/io_contract.md, 'Additive sources'). Every fixture is far
#                    under the byte budget and would be fully resident, so
#                    addwin.dat=3 forces the refill path; its genout must be
#                    byte-identical to the resident run. Without this the
#                    refill logic is never executed by any gate.
#   5 ABSENT == 0    deleting ncoords_add.dat leaves genout byte-identical: an
#                    untouched deck cannot be perturbed by the channel existing.
#   6 ADJOINT        bench_3d_adjoint still REFUSES a nonzero ncoords_add rather
#                    than silently dropping it (silent drop is the historical failure mode).
#
# Skips cleanly (exit 0) if binaries, python or a GPU are missing.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE_BIN="$ROOT/bin/bench_3d"; OPT_BIN="$ROOT/bin/bench_3d_opt"
ADJ_BIN="$ROOT/bin/bench_3d_adjoint"
GEN="$ROOT/tests/make_addsrc_3d_decks.py"
export CUDA_VISIBLE_DEVICES=${FW2U_GPU:-${FW2B_GPU:-0}}
PY=${PYTHON:-python}

[ -x "$BASE_BIN" ] && [ -x "$OPT_BIN" ] || { echo "SKIP: need bin/bench_3d + bin/bench_3d_opt (build: make 3d)"; exit 0; }
[ -f "$GEN" ] || { echo "SKIP: missing $GEN"; exit 0; }
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" -c "import fullwave2_ultra" 2>/dev/null \
  || { echo "SKIP: python cannot import fullwave2_ultra"; exit 0; }
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 \
  || { echo "SKIP: no CUDA GPU visible"; exit 0; }

work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" "$GEN" "$work/decks" >/dev/null

DECKS="hard zeroadd padnone padhard padadd A B AB coloc_none coloc_add coloc_fold A_win A_win5"
run(){  # run <binpath> <tag> <deck>
  local bin="$1" tag="$2" deck="$3"
  cp -r "$work/decks/$deck" "$work/${tag}_${deck}"
  ( cd "$work/${tag}_${deck}" && rm -f genout*.dat && "$bin" >stdout.log 2>err.log ) \
    || { echo "FAIL: $tag/$deck exited nonzero"; tail -3 "$work/${tag}_${deck}/stdout.log"; exit 1; }
}
for d in $DECKS; do run "$BASE_BIN" base "$d"; run "$OPT_BIN" opt "$d"; done

# case 5: the same deck with ncoords_add.dat removed entirely
cp -r "$work/decks/hard" "$work/opt_noadd"
rm -f "$work/opt_noadd/ncoords_add.dat" "$work/opt_noadd/genout.dat"
( cd "$work/opt_noadd" && "$OPT_BIN" >stdout.log 2>err.log )

PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" - "$work" <<'PYEOF'
import os, sys
import numpy as np
W = sys.argv[1]
def g(tag, deck):
    p = os.path.join(W, f"{tag}_{deck}", "genout.dat")
    v = np.fromfile(p, dtype=np.float32)
    assert v.size and np.isfinite(v).all(), f"{tag}/{deck}: empty or non-finite genout"
    return v

bad = 0
def check(ok, msg):
    global bad
    print(("  PASS  " if ok else "  FAIL  ") + msg)
    if not ok: bad += 1

for tag in ("base", "opt"):
    print(f"[{tag}]")
    hard, zero = g(tag, "hard"), g(tag, "zeroadd")
    A, B, AB = g(tag, "A"), g(tag, "B"), g(tag, "AB")
    cn, ca, cf = g(tag, "coloc_none"), g(tag, "coloc_add"), g(tag, "coloc_fold")
    Aw, Aw5 = g(tag, "A_win"), g(tag, "A_win5")

    check(hard.tobytes() == zero.tobytes(),
          f"1 transparency: zero-driven additive slab is invisible (max|d|={np.abs(hard-zero).max():.3e})")
    pn, ph, pa = g(tag, "padnone"), g(tag, "padhard"), g(tag, "padadd")
    check(pa.tobytes() == pn.tobytes(),
          f"1b same slab, held all run, ADDITIVE: still invisible (max|d|={np.abs(pa-pn).max():.3e})")
    opq = float(np.abs(ph - pn).max() / max(np.ptp(pn), 1e-30))
    check(opq > 1e-1,
          f"1c ...and DIRICHLET: opaque, a pressure-release screen (|d|/ptp = {opq:.4f} > 0.1)")

    rel = float(np.abs(AB - (A + B)).max() / max(np.ptp(AB), 1e-30))
    check(rel < 1e-5, f"2 superposition: |AB-(A+B)|/ptp = {rel:.3e} < 1e-5   [ncoords=0 decks]")
    check(np.ptp(A) > 0 and np.ptp(B) > 0,
          f"2b additive-only deck radiates: ptp(A)={np.ptp(A):.4g} ptp(B)={np.ptp(B):.4g}")

    check(ca.tobytes() == cf.tobytes(),
          f"3 order: colocated add == folded hard source, BYTE-exact (max|d|={np.abs(ca-cf).max():.3e})")
    sep = float(np.abs(ca - cn).max() / max(np.ptp(cf), 1e-30))
    check(sep > 1e-2,
          f"3b order: and NOT equal to the un-added source (|d|/ptp = {sep:.4f} > 1e-2)")

    check(A.tobytes() == Aw.tobytes(),
          f"4 windowing: addwin=3 refill path == fully resident, byte-identical (max|d|={np.abs(A-Aw).max():.3e})")
    check(A.tobytes() == Aw5.tobytes(),
          f"4b windowing: addwin=5 (does NOT divide nTic_add=24 -> partial last "
          f"window) also byte-identical (max|d|={np.abs(A-Aw5).max():.3e})")

print("[opt vs base, additive present]")
for d in ("A", "AB", "coloc_add", "zeroadd"):
    b, o = g("base", d), g("opt", d)
    rel = float(np.abs(b - o).max() / max(np.ptp(b), 1e-30))
    check(rel < 1e-4, f"opt==base on {d}: rel={rel:.3e} < 1e-4")

print("[untouched decks]")
na = np.fromfile(os.path.join(W, "opt_noadd", "genout.dat"), dtype=np.float32)
check(na.tobytes() == g("opt", "hard").tobytes(),
      "5 absent ncoords_add.dat == ncoords_add=0, byte-identical")

sys.exit(1 if bad else 0)
PYEOF
rc=$?

# case 6: the adjoint must still refuse, not drop
if [ -x "$ADJ_BIN" ]; then
  cp -r "$work/decks/A" "$work/adj_refuse"
  set +e
  ( cd "$work/adj_refuse" && "$ADJ_BIN" >stdout.log 2>&1 )
  arc=$?
  set -e
  if [ $arc -ne 0 ] && grep -q "ncoords_add" "$work/adj_refuse/stdout.log"; then
    echo "  PASS  6 adjoint still refuses a nonzero ncoords_add (rc=$arc)"
  else
    echo "  FAIL  6 adjoint did NOT refuse ncoords_add (rc=$arc) -- silent drop is the failure mode the refusal exists to prevent"
    rc=1
  fi
else
  echo "  SKIP  6 adjoint refusal (bin/bench_3d_adjoint not built: make adjoint)"
fi

[ $rc -eq 0 ] && echo "PASS 3D additive source channel" || echo "FAIL 3D additive source channel"
exit $rc
