#!/usr/bin/env python3
"""Generate a tiny synthetic 3D batch fixture (no proprietary data).

A small volumetric grid + a source patch + a receiver plane, written through the
validated sim.write_fullwave_sim_3d / io_dat path. The 3D solver gate
(tests/test_3d_opt_eq_base.sh) runs bench_3d (base) and bench_3d_opt on this dir
and asserts the two genout.dat are byte-identical.

The medium is deterministic and re-generated on demand (NOT committed -- the 3D
volumes are larger than the tracked 2D fixture). Usage:
    PYTHONPATH=. python examples/tiny_3d/make_fixture_3d.py [outdir] [--bigndmap]

  --bigndmap : widen the sound-speed range, forcing the optimized solver onto its
               global-memory (_opt) path instead of the default constant-memory
               (_opt2) path -- exercises the wide-coefficient-range fallback.
"""
import os, sys, struct, numpy as np
from fullwave2_ultra import sim, io_dat

NXI = NYI = NZI = 24              # interior volume
# CFL=0.30: the 3D von Neumann stability limit is ~1/sqrt(3) of 1D (vs ~1/sqrt(2)
# in 2D), so the 2D production CFL=0.45 diverges in 3D. 0.30 is comfortably stable
# for the M=8 scheme on a homogeneous-ish medium.
PPW, CFL = 6.0, 0.30
C0, F0 = 1540.0, 3.7e6
OMEGA0 = 2 * np.pi * F0
NT = 64                          # short run
NTIC = 24                        # pulse length (samples)
NBDY, M = 6, 8                   # absorbing ring + FD half-width (mext=14)


def main(outdir, bigndmap=False):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(0)
    # synthetic smooth medium: c with a mild 3D gradient + small noise.
    ii, jj, kk = np.meshgrid(np.linspace(-1, 1, NXI), np.linspace(-1, 1, NYI),
                             np.linspace(-1, 1, NZI), indexing="ij")
    span = 900.0 if bigndmap else 40.0
    cmap = 1540.0 + span * 0.5 * (jj + 0.3 * ii) + rng.normal(0, 2.0, (NXI, NYI, NZI))
    lo, hi = (1500.0, 1500.0 + span + 60) if bigndmap else (1480.0, 1600.0)
    cmap = np.clip(cmap, lo, hi).astype(np.float64)
    rmap = np.full((NXI, NYI, NZI), 1000.0)
    nmap = np.zeros((NXI, NYI, NZI))
    amap = np.zeros((NXI, NYI, NZI))      # Aexp = sponge only (bounded, lossless interior)
    maps = dict(cmap=cmap, rmap=rmap, nmap=nmap, amap=amap)

    # source: a small patch on the i = 6 plane near the center; receivers: a line.
    sj, sk = np.meshgrid(np.arange(9, 15), np.arange(9, 15), indexing="ij")
    si = np.full(sj.size, 6)
    incoords = np.stack([si, sj.ravel(), sk.ravel()], 1).astype(int)
    ncoords = incoords.shape[0]
    rj = np.arange(6, 18)
    outcoords = np.stack([np.full(rj.size, 16), rj, np.full(rj.size, 12)], 1).astype(int)

    meta = sim.write_fullwave_sim_3d(outdir, C0, OMEGA0, 1.0, PPW, CFL, maps,
                                     incoords, outcoords, NTIC, 1, nbdy=NBDY, M=M)
    io_dat.write_int(os.path.join(outdir, "nT.dat"), NT)     # force the short run

    dT = meta["dT"]
    t = (np.arange(NTIC) - NTIC / 2) * dT
    icvec = np.exp(-(t * OMEGA0 / (2 * np.pi)) ** 2) * np.sin(t * OMEGA0) * 1e5
    # 3D base icmat: single sim, coord-major then time (all source coords share icvec)
    block = np.tile(icvec[None, :], (ncoords, 1))
    io_dat.write_icmat(os.path.join(outdir, "icmat.dat"), [block])

    print(f"3d fixture: interior {NXI}x{NYI}x{NZI} -> extended "
          f"{meta['nXe']}x{meta['nYe']}x{meta['nZe']}, "
          f"ncoords={meta['ncoords']} ncoordsout={meta['ncoordsout']} nT={NT} nTic={NTIC}"
          f"{'  [bigndmap -> _opt global path]' if bigndmap else ''}")
    print(f"wrote -> {outdir}")
    rd = lambda n: struct.unpack("i", open(os.path.join(outdir, n), "rb").read(4))[0]
    assert rd("nT.dat") == NT and rd("ncoords.dat") == ncoords


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    big = "--bigndmap" in sys.argv
    out = args[0] if args else os.path.dirname(os.path.abspath(__file__))
    main(out, bigndmap=big)
