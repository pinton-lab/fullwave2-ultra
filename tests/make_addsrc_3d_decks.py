#!/usr/bin/env python3
"""Build the 3D additive-source gate decks (tests/test_addsrc_3d.sh).

All decks share one synthetic medium, grid and time base; they differ only in
how the source is presented. The point of each is documented at its writer.

    PYTHONPATH=. python tests/make_addsrc_3d_decks.py <outdir>

Writes <outdir>/{hard,zeroadd,pad{none,hard,add},A,B,AB,coloc_*,A_win}.
"""
import os, sys
import numpy as np
from fullwave2_ultra import sim, io_dat

NXI = NYI = NZI = 24
PPW, CFL = 6.0, 0.30              # 3D limit is ~0.49 (stencil.cfl_limit dim=3)
C0, F0 = 1540.0, 3.7e6
OMEGA0 = 2 * np.pi * F0
NT, NTIC = 64, 24
NBDY, M = 6, 8


def _maps():
    rng = np.random.default_rng(0)
    ii, jj, kk = np.meshgrid(*[np.linspace(-1, 1, n) for n in (NXI, NYI, NZI)],
                             indexing="ij")
    cmap = np.clip(1540.0 + 20.0 * (jj + 0.3 * ii) + rng.normal(0, 2.0, ii.shape),
                   1480.0, 1600.0).astype(np.float64)
    z = np.zeros((NXI, NYI, NZI))
    # nmap = 0 -> the solver is LINEAR, which is what the superposition case tests.
    return dict(cmap=cmap, rmap=np.full((NXI, NYI, NZI), 1000.0),
                nmap=z, amap=z.copy())


def _burst(n, dT, amp=1.0):
    t = (np.arange(n) - n / 2) * dT
    return (amp * np.exp(-((t * F0 * 0.55) ** 2)) * np.sin(OMEGA0 * t)).astype(np.float64)


def _burst_q(n, dT, amp=1.0, bits=12):
    """A burst quantized to multiples of 2**-bits.

    The colocation case pins an ORDER, so its assertion should be exact rather
    than fp-tolerant. Quantizing lets it be: with values that are multiples of
    2**-12 and O(1), the device's float32(q) + float32(q/2) and a host-folded
    float32(1.5*q) are the SAME float32 -- no double rounding to hide behind.
    """
    q = np.round(_burst(n, dT, amp) * (1 << bits)) / (1 << bits)
    return q.astype(np.float64)


def _outcoords():
    rj = np.arange(6, 18)
    return np.stack([np.full(rj.size, 16), rj, np.full(rj.size, 12)], 1).astype(int)


def _hard_incoords():
    sj, sk = np.meshgrid(np.arange(9, 15), np.arange(9, 15), indexing="ij")
    return np.stack([np.full(sj.size, 6), sj.ravel(), sk.ravel()], 1).astype(int)


def _deck(root, name, incoords, nTic=NTIC):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    meta = sim.write_fullwave_sim_3d(d, C0, OMEGA0, 1.0, PPW, CFL, _maps(),
                                     incoords, _outcoords(), nTic, 1,
                                     nbdy=NBDY, M=M)
    io_dat.write_int(os.path.join(d, "nT.dat"), NT)
    return d, meta


def _add_region(mext, i0=14):
    """A dense slab of additive cells on an i-plane, EXTENDED-grid coords.

    Dense on purpose: a Dirichlet region of this shape is a rigid screen, so the
    transparency case only means something if the region is thick enough to
    reflect were it hard.
    """
    jj, kk = np.meshgrid(np.arange(6, 18), np.arange(6, 18), indexing="ij")
    ii = np.stack([np.full(jj.size, i0 + di) for di in (0, 1, 2)], 0).ravel()
    jj = np.tile(jj.ravel(), 3); kk = np.tile(kk.ravel(), 3)
    return np.stack([ii, jj, kk], 1).astype(int) + mext - 1


def main(root):
    os.makedirs(root, exist_ok=True)
    hard_ic = _hard_incoords()
    empty = np.zeros((0, 3), dtype=int)

    # --- hard: imposed-pressure source only. Reference for transparency. -----
    d, meta = _deck(root, "hard", hard_ic)
    dT, mext = meta["dT"], meta["mext"]
    src = np.tile(_burst(NTIC, dT), (hard_ic.shape[0], 1))
    io_dat.write_icmat(os.path.join(d, "icmat.dat"), [src])

    # --- zeroadd: same, plus a dense additive region driven with ZEROS. ------
    # Transparency: adding 0 must leave the field untouched, i.e. the region is
    # invisible to the wave crossing it (a hard region here would reflect).
    d, meta = _deck(root, "zeroadd", hard_ic)
    io_dat.write_icmat(os.path.join(d, "icmat.dat"), [src])
    reg = _add_region(mext)
    sim.write_addsrc_rundir(d, reg, np.zeros((reg.shape[0], NTIC)))

    # --- pad{none,hard,add}: the SAME slab cells, Dirichlet vs additive. -----
    # This trio is what makes case 1 a physics claim rather than "adding 0 does
    # nothing". All three run with nTic = nT so the clamp is held for the WHOLE
    # run -- with the default nTic the Dirichlet slab stops being a screen after
    # the burst and reflects almost nothing, which is not the comparison wanted.
    # padhard: the slab as p=0 Dirichlet -- a 3-cell pressure-release screen,
    #          |R| -> 1, so its field must differ from padnone by a wide margin.
    # padadd:  the same cells, same zero drive, ADDITIVE -- invisible.
    srcpad = np.zeros((hard_ic.shape[0], NT)); srcpad[:, :NTIC] = src
    slab_i = _add_region(mext) - (mext - 1)

    d, _ = _deck(root, "padnone", hard_ic, nTic=NT)
    io_dat.write_icmat(os.path.join(d, "icmat.dat"), [srcpad])

    d, _ = _deck(root, "padhard", np.concatenate([hard_ic, slab_i]), nTic=NT)
    io_dat.write_icmat(os.path.join(d, "icmat.dat"),
                       [np.concatenate([srcpad, np.zeros((slab_i.shape[0], NT))])])

    d, _ = _deck(root, "padadd", hard_ic, nTic=NT)
    io_dat.write_icmat(os.path.join(d, "icmat.dat"), [srcpad])
    sim.write_addsrc_rundir(d, reg, np.zeros((reg.shape[0], NT)))

    # --- A / B / AB: ncoords=0, additive only. Superposition + the ----------
    # additive-only deck shape the hybrid split needs (no dummy hard cell).
    setA = _add_region(mext, i0=14)[:120]
    setB = _add_region(mext, i0=14)[120:]
    trA = np.tile(_burst(NTIC, dT, 1e3), (setA.shape[0], 1))
    trB = np.tile(_burst(NTIC, dT, 1e3), (setB.shape[0], 1)) * 0.7
    for nm, cs, tr in (("A", setA, trA), ("B", setB, trB),
                       ("AB", np.concatenate([setA, setB]),
                        np.concatenate([trA, trB]))):
        d, _ = _deck(root, nm, empty)
        sim.write_addsrc_rundir(d, cs, tr)

    # --- coloc_{none,add,fold}: additive coords ON the hard coords. ---------
    # The order is set-then-add, so p = icmat[n] + T[n]. With T = q/2 and the
    # folded trace 1.5*q -- all exactly representable -- coloc_add and
    # coloc_fold must be BYTE-identical. Had the solver added before clamping,
    # coloc_add would instead equal coloc_none. Both directions are asserted.
    q = np.tile(_burst_q(NTIC, dT), (hard_ic.shape[0], 1))
    ce = hard_ic + mext - 1

    d, _ = _deck(root, "coloc_none", hard_ic)
    io_dat.write_icmat(os.path.join(d, "icmat.dat"), [q])

    d, _ = _deck(root, "coloc_add", hard_ic)
    io_dat.write_icmat(os.path.join(d, "icmat.dat"), [q])
    sim.write_addsrc_rundir(d, ce, 0.5 * q)

    d, _ = _deck(root, "coloc_fold", hard_ic)
    io_dat.write_icmat(os.path.join(d, "icmat.dat"), [1.5 * q])

    # --- A_win: deck A forced onto the windowed refill path (addwin=3). -----
    # Every fixture is far under FW2_ADDSRC_WIN_BYTES and would otherwise be
    # fully resident, so the refill logic would never be exercised.
    for nm, k in (("A_win", 3), ("A_win5", 5)):
        d, _ = _deck(root, nm, empty)
        sim.write_addsrc_rundir(d, setA, trA)
        io_dat.write_int(os.path.join(d, "addwin.dat"), k)
    # k=3 divides nTic_add=24 exactly; k=5 does NOT, so the last window is
    # partial (slices 20..23 of a 5-slice buffer) -- the refill's cnt clamp.

    print("decks written under", root)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "addsrc3d_decks")
