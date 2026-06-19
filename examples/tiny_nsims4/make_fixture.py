#!/usr/bin/env python3
"""Generate the tiny nsims=4 batch fixture (SYNTHETIC medium -- no proprietary data).

A small grid + a 4-element source line + a receiver line, written through the same
validated sim.write_fullwave_sim / io_dat path the real pipeline uses. The solver
gates (tests/test_batch_eq_solo.sh, tests/test_opt6_batch_offset.sh) run the batch
binary on this dir and compare per-sim genout to per-element solo runs.

Deterministic: re-running reproduces byte-identical inputs (this is also the golden
reference for tests/test_fixture_parity.py). Run from the repo root:
    PYTHONPATH=. python examples/tiny_nsims4/make_fixture.py
then run bin/bench_2d_batch in the dir to (re)produce the expected genout*.dat.
"""
import os, struct, numpy as np
from fullwave2_ultra import sim, io_dat

HERE = os.path.dirname(os.path.abspath(__file__))
NXI, NYI = 56, 64                 # interior grid (axial x lateral)
NSIMS = 4
PPW, CFL = 6.0, 0.45
C0, F0 = 1540.0, 3.7e6
OMEGA0 = 2 * np.pi * F0
NT = 256                          # short run (override write_fullwave_sim's nT)
NTIC = 48                         # pulse length (samples)


def main():
    rng = np.random.default_rng(0)
    # synthetic smooth medium: c with a mild lateral ramp + small noise,
    # homogeneous rho/beta, zero attenuation (Aexp=1).
    jj = np.linspace(-1.0, 1.0, NYI)[None, :]
    cmap = 1540.0 + 30.0 * jj + rng.normal(0, 3.0, (NXI, NYI))
    cmap = np.clip(cmap, 1480, 1600).astype(np.float64)
    rmap = np.full((NXI, NYI), 1000.0)
    nmap = np.zeros((NXI, NYI))
    amap = np.zeros((NXI, NYI))           # no absorption -> Aexp == 1
    maps = dict(cmap=cmap, rmap=rmap, nmap=nmap, amap=amap)

    # 4-element source line (axial i=4) + receiver line (axial i=10), 48 cols each.
    jcols = np.arange(8, 56)
    ncoords = jcols.size
    elem = (np.arange(ncoords) // (ncoords // NSIMS) + 1).clip(1, NSIMS)
    incoords = np.stack([np.full(ncoords, 4), jcols, np.zeros(ncoords), elem], 1).astype(int)
    outcoords = np.stack([np.full(ncoords, 10), jcols, np.zeros(ncoords),
                          np.arange(1, ncoords + 1)], 1).astype(int)
    xdc = dict(incoords=incoords, outcoords=outcoords)

    dur = NT / (C0 / (C0 / F0) / CFL)     # placeholder; nT overridden below
    meta = sim.write_fullwave_sim(HERE, C0, OMEGA0, dur, PPW, CFL, maps, xdc, NTIC, 1)
    io_dat.write_int(os.path.join(HERE, "nT.dat"), NT)        # force the short run

    # 2-cycle tone-burst transmit pulse
    dT = meta["dT"]
    t = (np.arange(NTIC) - NTIC / 2) * dT
    icvec = np.exp(-(t * OMEGA0 / (2 * np.pi)) ** 2) * np.sin(t * OMEGA0) * 1e5
    blocks = [(elem == n)[:, None] * icvec[None, :] for n in range(1, NSIMS + 1)]
    io_dat.write_icmat(os.path.join(HERE, "icmat.dat"), blocks)
    io_dat.write_int(os.path.join(HERE, "nsims.dat"), NSIMS)

    nXe, nYe = meta["nXe"], meta["nYe"]
    print(f"fixture: interior {NXI}x{NYI} -> extended {nXe}x{nYe}, "
          f"ncoords={meta['ncoords']} ncoordsout={meta['ncoordsout']} nsims={NSIMS} nT={NT} nTic={NTIC}")
    print(f"wrote -> {HERE}")
    rd = lambda n: struct.unpack("i", open(os.path.join(HERE, n), "rb").read(4))[0]
    assert rd("nsims.dat") == NSIMS and rd("nT.dat") == NT


if __name__ == "__main__":
    main()
