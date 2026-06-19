#!/usr/bin/env bash
# genout_mod GPU gate: the streamed, (modX,modY,modZ)-decimated full-field dump
# must (1) sample the same field as the coords-based genout, and (2) decimate
# correctly (mod=2 == the mod=1 full field strided ::2). Needs bin/bench_3d_opt +
# python (regenerates the uncommitted 3D fixture). Skips cleanly otherwise.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/bin/bench_3d_opt"
[ -x "$BIN" ] || { echo "SKIP: need bin/bench_3d_opt -- place/fetch it in bin/ (see README)"; exit 0; }
PYTHONPATH="$ROOT" python3 -c "import fullwave2_ultra" 2>/dev/null || { echo "SKIP: fullwave2_ultra not importable"; exit 0; }
gpu=${FW2U_GPU:-0}
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
gen(){ PYTHONPATH="$ROOT" python3 "$ROOT/examples/tiny_3d/make_fixture_3d.py" "$1" >/dev/null 2>&1; }
setmod(){ PYTHONPATH="$ROOT" python3 -c "from fullwave2_ultra import io_dat as d
for a,v in zip('XYZ',[$2,$3,$4]): d.write_int('$1/mod'+a+'.dat',v)"; }
gen "$W/m1"; setmod "$W/m1" 1 1 1
gen "$W/m2"; setmod "$W/m2" 2 2 2
( cd "$W/m1" && CUDA_VISIBLE_DEVICES=$gpu "$BIN" >stdout.log 2>&1 )
( cd "$W/m2" && CUDA_VISIBLE_DEVICES=$gpu "$BIN" >stdout.log 2>&1 )
PYTHONPATH="$ROOT" python3 - "$W" <<'PY'
import sys, numpy as np
from fullwave2_ultra import io_dat
W = sys.argv[1]; sc = lambda n: io_dat.read_int(f"{W}/m1/{n}.dat")
nX, nY, nZ, nco = sc("nX"), sc("nY"), sc("nZ"), sc("ncoordsout")
gen = io_dat.read_genout(f"{W}/m1/genout.dat", nco)
gm1 = io_dat.read_genout_mod(f"{W}/m1/genout_mod.dat", nX, nY, nZ)
outc = io_dat.read_coords_3d(f"{W}/m1/outc.dat", nco)
assert all(np.array_equal(gm1[:, i, j, k], gen[:, m]) for m, (i, j, k) in enumerate(outc)), \
    "genout_mod(mod=1) does not match genout at the output coords"
gm2 = io_dat.read_genout_mod(f"{W}/m2/genout_mod.dat", nX, nY, nZ, 2, 2, 2)
assert np.array_equal(gm2, gm1[:, ::2, ::2, ::2]), "genout_mod(mod=2) != full field [::2,::2,::2]"
print(f"PASS genout_mod: self-consistent with genout + correct decimation (grid {nX}x{nY}x{nZ})")
PY
