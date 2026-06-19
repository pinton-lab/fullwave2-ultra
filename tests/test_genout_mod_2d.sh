#!/usr/bin/env bash
# genout_mod 2D GPU gate: the streamed, (modX,modY)-decimated full-field dump from
# the 2D BATCH solver must (1) leave genout byte-unchanged when the mod scalars are
# absent, (2) per sim, sample the same field as the coords-based genout, and
# (3) decimate correctly (mod=2 == the mod=1 full field strided ::2). Needs
# bin/bench_2d_batch + python + the tiny_nsims4 fixture. Skips cleanly otherwise.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/bin/bench_2d_batch"
FIX="$ROOT/examples/tiny_nsims4"
[ -x "$BIN" ] || { echo "SKIP: need bin/bench_2d_batch -- place/fetch it in bin/ (see README)"; exit 0; }
[ -d "$FIX" ] || { echo "SKIP: missing fixture $FIX (regen via examples/tiny_nsims4/make_fixture.py)"; exit 0; }
PYTHONPATH="$ROOT" python3 -c "import fullwave2_ultra" 2>/dev/null || { echo "SKIP: fullwave2_ultra not importable"; exit 0; }
# regenerate the (uncommitted) fixture data if maps are missing
PYTHONPATH="$ROOT" python3 "$FIX/make_fixture.py" >/dev/null 2>&1 || true
gpu=${FW2U_GPU:-0}
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
setmod(){ PYTHONPATH="$ROOT" python3 -c "from fullwave2_ultra import io_dat as d
for a,v in zip('XY',[$2,$3]): d.write_int('$1/mod'+a+'.dat',v)"; }
for d in nomod m1 m2; do cp -r "$FIX" "$W/$d"; ( cd "$W/$d" && rm -f genout*.dat mod*.dat ); done
setmod "$W/m1" 1 1
setmod "$W/m2" 2 2
( cd "$W/nomod" && CUDA_VISIBLE_DEVICES=$gpu "$BIN" >stdout.log 2>&1 )
( cd "$W/m1"    && CUDA_VISIBLE_DEVICES=$gpu "$BIN" >stdout.log 2>&1 )
( cd "$W/m2"    && CUDA_VISIBLE_DEVICES=$gpu "$BIN" >stdout.log 2>&1 )
# (1) genout byte-unchanged with vs without the mod scalars (per sim)
nsims=$(PYTHONPATH="$ROOT" python3 -c "from fullwave2_ultra import io_dat as d;print(d.read_int('$FIX/nsims.dat'))")
for ((s=0; s<nsims; s++)); do
  f=$([ $s -eq 0 ] && echo genout.dat || echo "genout_s$s.dat")
  cmp -s "$W/nomod/$f" "$W/m1/$f" || { echo "FAIL: $f perturbed by mod scalars"; exit 1; }
done
echo "OK: genout byte-unchanged across all $nsims sims (mod scalars do not perturb it)"
PYTHONPATH="$ROOT" python3 - "$W" "$nsims" <<'PY'
import sys, numpy as np
from fullwave2_ultra import io_dat
W, nsims = sys.argv[1], int(sys.argv[2])
sc = lambda n: io_dat.read_int(f"{W}/m1/{n}.dat")
nX, nY, nco = sc("nX"), sc("nY"), sc("ncoordsout")
oc = np.fromfile(f"{W}/m1/outc.dat", dtype=np.int32).reshape(2, nco)   # [i;j]
outc = list(zip(oc[0], oc[1]))
for s in range(nsims):
    gfn = io_dat.genout_name(s); mfn = io_dat.genout_mod_name(s)
    gen = io_dat.read_genout(f"{W}/m1/{gfn}", nco)
    gm1 = io_dat.read_genout_mod_2d(f"{W}/m1/{mfn}", nX, nY)
    gm2 = io_dat.read_genout_mod_2d(f"{W}/m2/{mfn}", nX, nY, 2, 2)
    assert all(np.array_equal(gm1[:, i, j], gen[:, m]) for m, (i, j) in enumerate(outc)), \
        f"sim {s}: genout_mod(mod=1) != genout at the output coords"
    assert np.array_equal(gm2, gm1[:, ::2, ::2]), f"sim {s}: genout_mod(mod=2) != full[::2,::2]"
print(f"PASS genout_mod 2D: self-consistent with genout + correct decimation "
      f"({nsims} sims, grid {nX}x{nY})")
PY
