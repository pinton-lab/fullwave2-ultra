# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gianmarco Pinton
"""GPU gate for the 2D additive source channel (icc_add/icmat_add).

Four checks, all in homogeneous linear water (beta=0, amap=0) so superposition
holds exactly in the continuum:

  (i)   superposition: run(mono@P1) + run(mono@P2) == run(both), fp32 tolerance
  (ii)  2D Green's function: r^-1/2 peak decay exponent and arrival speed
  (iii) nTic_add truncation: zero-padded full-length traces give byte-identical
        output to truncated traces + nTic_add (no zero-clamp after the traces)
  (iv)  batch==solo with additive sources: per-sim batch genout byte-identical
        to bench_2d_aexp solo runs of the same traces

Invoked by tests/test_2d_additive_source.sh which resolves the binaries; run
directly as:  python tests/check_additive.py <batch_bin> <solo_bin>
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fullwave2_ultra import io_dat, sim  # noqa: E402

NXI, NYI = 96, 64                 # interior grid (axial x lateral)
PPW, CFL = 6.0, 0.45
C0, F0 = 1540.0, 3.7e6
OMEGA0 = 2 * np.pi * F0
NT = 384
NTIC = 4                          # hard channel: one dummy zero-clamp pixel
NTIC_ADD = 160                    # additive traces (pulse ends well before this)
P1 = (16, 32)                     # monopole positions (1-based interior)
P2 = (16, 48)
RECV = [(36, 32), (76, 32)]       # receivers axially below P1: r = 20, 60 px


def pulse(dT, scale=1.0, ncycles=2.0):
    t = (np.arange(NTIC_ADD) - NTIC_ADD / 2) * dT
    sig = ncycles / F0 / 2.355
    return (np.exp(-(t / sig) ** 2) * np.sin(t * OMEGA0) * 1e4 * scale).astype(np.float64)


def write_run(outdir, add_pts, blocks, nsims=1, nTic_add=NTIC_ADD, nT=NT):
    """One run dir: homogeneous water, dummy hard zero-clamp at (1,1), additive
    sources at add_pts with per-sim trace blocks [(ncoords_add, nTic_add), ...]."""
    maps = dict(cmap=np.full((NXI, NYI), C0), rmap=np.full((NXI, NYI), 1000.0),
                nmap=np.zeros((NXI, NYI)), amap=np.zeros((NXI, NYI)))
    incoords = np.array([[1, 1]])
    outcoords = np.array(RECV + [list(P1), list(P2)])
    xdc = dict(incoords=incoords, outcoords=outcoords)
    lam = C0 / F0
    dur = nT / (C0 / lam * PPW / CFL)
    meta = sim.write_fullwave_sim(outdir, C0, OMEGA0, dur, PPW, CFL, maps, xdc,
                                  NTIC, 1, add_coords=np.array(add_pts),
                                  nTic_add=nTic_add)
    io_dat.write_int(os.path.join(outdir, "nT.dat"), nT)
    io_dat.write_icmat(os.path.join(outdir, "icmat.dat"),
                       [np.zeros((1, NTIC))] * nsims)
    io_dat.write_icmat(os.path.join(outdir, "icmat_add.dat"), blocks)
    if nsims > 1:
        io_dat.write_int(os.path.join(outdir, "nsims.dat"), nsims)
    return meta


def run_bin(binary, rundir):
    subprocess.run([binary], cwd=rundir, check=True, capture_output=True)


def genout(rundir, ncoordsout=4, name="genout.dat"):
    return io_dat.read_genout(os.path.join(rundir, name), ncoordsout)


def main(batch_bin, solo_bin):
    lam = C0 / F0
    dT = lam / PPW / C0 * CFL
    g1, g2 = pulse(dT), pulse(dT, scale=0.7)
    work = tempfile.mkdtemp(prefix="fw2u_additive_")
    ok = True
    try:
        # --- (i) superposition ------------------------------------------------
        for tag, pts, blks in [("A", [P1], [g1[None, :]]),
                               ("B", [P2], [g2[None, :]]),
                               ("AB", [P1, P2], [np.stack([g1, g2])])]:
            write_run(os.path.join(work, tag), pts, blks)
            run_bin(batch_bin, os.path.join(work, tag))
        gA, gB, gAB = (genout(os.path.join(work, t)) for t in ("A", "B", "AB"))
        rel = np.abs(gAB - (gA + gB)).max() / np.ptp(gAB)
        print(f"superposition: max|AB-(A+B)|/ptp = {rel:.3e}")
        if rel > 1e-6:
            print("FAIL superposition"); ok = False

        # --- (ii) 2D Green's function: decay exponent + arrival speed --------
        dX = lam / PPW
        r = np.array([20.0, 60.0]) * dX
        env = np.abs(gA[:, :2])                             # receivers at r1, r2
        pk = env.max(axis=0)
        alpha = np.log(pk[0] / pk[1]) / np.log(r[1] / r[0])
        t_pk = env.argmax(axis=0) * dT                      # modT=1
        c_est = (r[1] - r[0]) / (t_pk[1] - t_pk[0])
        print(f"green: decay exponent {alpha:.3f} (want 0.5), "
              f"c_est {c_est:.1f} m/s (want {C0:.0f})")
        if abs(alpha - 0.5) > 0.05:
            print("FAIL green decay"); ok = False
        if abs(c_est - C0) / C0 > 0.01:
            print("FAIL green arrival"); ok = False

        # --- (iii) nTic_add truncation == zero-padded full length ------------
        g1_pad = np.zeros(NT); g1_pad[:NTIC_ADD] = g1
        write_run(os.path.join(work, "PAD"), [P1], [g1_pad[None, :]], nTic_add=NT)
        run_bin(batch_bin, os.path.join(work, "PAD"))
        b_trunc = open(os.path.join(work, "A", "genout.dat"), "rb").read()
        b_pad = open(os.path.join(work, "PAD", "genout.dat"), "rb").read()
        print(f"nTic_add truncation: {'byte-identical' if b_trunc == b_pad else 'MISMATCH'}")
        if b_trunc != b_pad:
            print("FAIL nTic_add"); ok = False

        # --- (iv) batch==solo with additive traces ---------------------------
        write_run(os.path.join(work, "BATCH"), [P1],
                  [g1[None, :], g2[None, :]], nsims=2)
        run_bin(batch_bin, os.path.join(work, "BATCH"))
        for s, g in ((0, g1), (1, g2)):
            d = os.path.join(work, f"SOLO{s}")
            write_run(d, [P1], [g[None, :]])
            run_bin(solo_bin, d)
            bs = open(os.path.join(work, "BATCH", io_dat.genout_name(s)), "rb").read()
            so = open(os.path.join(d, "genout.dat"), "rb").read()
            print(f"batch sim{s} vs solo: {'byte-identical' if bs == so else 'MISMATCH'}")
            if bs != so:
                print(f"FAIL batch==solo sim{s}"); ok = False
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print("PASS additive source channel" if ok else "FAIL additive source channel")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
