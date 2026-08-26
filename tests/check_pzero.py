# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gianmarco Pinton
"""GPU gate for the 2D pressure-release channel (icczero / ncoordszero).

A `pzero` sheet is a per-step Dirichlet p = 0. The point of it is that it gives
|R| = 1 with NO material contrast, so it escapes the explicit-scheme stability
ceiling that makes physical air (343 m/s, 1.2 kg/m^3) diverge outright as a
medium. This gate proves the surface is acoustically what it claims to be.

Five checks, in homogeneous lossless water (beta = 0, amap = 0):

  (i)   |R| = 1 to a few percent, measured against a no-sheet reference run
  (ii)  the reflected pulse is INVERTED -- normalised cross-correlation with the
        incident pulse near -1. This is the check that matters: a rigid surface
        also gives |R| = 1, and only the sign distinguishes the two. Amplitude
        alone would pass on a wall that is physically the opposite of the one
        being modelled.
  (iii) transmission past the sheet is at the noise floor
  (iv)  ncoordszero = 0 with icczero.dat present is byte-identical to a run with
        no icczero.dat at all
  (v)   batch == solo, byte-for-byte, with the channel active

The source is a hard line launched with source_zero_window = (nT, nT), so the
source cells stop being clamped once the burst ends and the returning echo
passes through them instead of re-reflecting off a second pressure-release
surface. That exercises the source-zero window as a side effect.

Invoked by tests/test_pzero.sh; run directly as:
    python tests/check_pzero.py <batch_bin> <solo_bin>
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fullwave2_ultra import io_dat, sim  # noqa: E402

NXI, NYI = 210, 64                # interior grid (axial x lateral)
PPW, CFL = 6.0, 0.45
C0, F0 = 1540.0, 3.7e6
OMEGA0 = 2 * np.pi * F0
NT, NTIC = 700, 96
NBDY, MSTEN = 40, 8
MEXT = NBDY + MSTEN               # replicate pad; interior i -> extended i+MEXT-1
XSRC, XR1, XSHEET, XR2 = 24, 64, 132, 168     # 1-based interior axial indices
# Free-field reference receiver at the SAME path length as the echo: the
# reflection reaches R1 having travelled (XSHEET-XSRC) + (XSHEET-XR1) = 176 px,
# while the incident wave reaches R1 after only 40. Comparing their PEAKS
# without matching path length charges the echo for 136 px of numerical
# dispersion that the incident never saw, and reads as |R| ~ 0.96.
XREF = XSRC + (XSHEET - XSRC) + (XSHEET - XR1)
YC = NYI // 2
# Sheet thickness. A one-cell Dirichlet row does NOT block an M-tap scheme --
# the fd1b/fd2b stencils reach straight across it. Measured transmission past
# the sheet, M=8: 1 cell 2.0e-1, 3 cells 2.9e-2, 5 cells 8.1e-3, 9 cells
# 8.6e-5, 17 cells 0. 2M+1 is the first thickness that decouples the two sides
# exactly, and is what a pressure-release INTERFACE should use.
THICK = 2 * 8 + 1


def pulse(dT, ncycles=2.0):
    t = (np.arange(NTIC) - NTIC / 2) * dT
    sig = ncycles / F0 / 2.355
    return (np.exp(-(t / sig) ** 2) * np.sin(t * OMEGA0) * 1e4).astype(np.float64)


def write_run(outdir, sheet=True, nzero=None):
    """Homogeneous water; hard line source across the width at XSRC; optional
    pressure-release sheet across the width at XSHEET held for the whole run."""
    maps = dict(cmap=np.full((NXI, NYI), C0), rmap=np.full((NXI, NYI), 1000.0),
                nmap=np.zeros((NXI, NYI)), amap=np.zeros((NXI, NYI)))
    # Span the FULL EXTENDED width, not just the interior. A sheet that stops at
    # the interior edge is a finite strip: the wave diffracts around it, which
    # shows up as |R| > 1 from edge arrivals at the receiver and as spurious
    # "transmission" behind the sheet. Interior index (1 - MEXT) maps to
    # extended column 0, (NYI + MEXT) to the last one.
    ys = np.arange(1 - MEXT, NYI + MEXT + 1)
    incoords = np.stack([np.full(ys.size, XSRC), ys], axis=1)
    outcoords = np.array([[XR1, YC], [XR2, YC], [XREF, YC]])
    if sheet:
        xs = np.concatenate([np.full(ys.size, XSHEET + t) for t in range(THICK)])
        zc = np.stack([xs, np.tile(ys, THICK)], axis=1)
    else:
        zc = None
    lam = C0 / F0
    dur = NT / (C0 / lam * PPW / CFL)
    kw = {}
    if zc is not None:
        kw.update(zero_coords=zc, zero_window=(0, NT))
    meta = sim.write_fullwave_sim(
        outdir, C0, OMEGA0, dur, PPW, CFL, maps,
        dict(incoords=incoords, outcoords=outcoords), NTIC, 1,
        nbdy=NBDY, M=MSTEN, source_zero_window=(NT, NT), **kw)
    io_dat.write_int(os.path.join(outdir, "nT.dat"), NT)
    if nzero is not None:                      # override for check (iv)
        io_dat.write_int(os.path.join(outdir, "ncoordszero.dat"), nzero)
    dT = meta["dT"]
    io_dat.write_icmat(os.path.join(outdir, "icmat.dat"),
                       [np.tile(pulse(dT), (incoords.shape[0], 1))])
    return meta


def run_bin(binary, rundir):
    subprocess.run([binary], cwd=rundir, check=True, capture_output=True)


def nxcorr(a, b):
    """Peak normalised cross-correlation, sign preserved."""
    a = a - a.mean(); b = b - b.mean()
    a = a / (np.linalg.norm(a) or 1.0); b = b / (np.linalg.norm(b) or 1.0)
    c = np.correlate(a, b, mode="full")
    return float(c[np.argmax(np.abs(c))])


def main(batch_bin, solo_bin):
    work = tempfile.mkdtemp(prefix="fw2u_pzero_")
    ok = True

    ref, sh = os.path.join(work, "REF"), os.path.join(work, "SHEET")
    write_run(ref, sheet=False)
    write_run(sh, sheet=True)
    for d in (ref, sh):
        run_bin(batch_bin, d)
    gr = io_dat.read_genout(os.path.join(ref, "genout.dat"), 3)
    gs = io_dat.read_genout(os.path.join(sh, "genout.dat"), 3)

    # arrival windows from geometry: cfl px per step
    t1 = int((XR1 - XSRC) / CFL)
    t2 = int((XR1 - XSRC + 2 * (XSHEET - XR1)) / CFL)
    iw = slice(max(t1 - 40, 0), t1 + 90)
    rw = slice(max(t2 - 70, 0), min(t2 + 90, NT - 1))

    reflected = (gs - gr)[rw, 0]
    incident = gr[iw, 0]                      # for the SHAPE comparison
    # amplitude reference: free field at matched path length
    tref = int((XREF - XSRC) / CFL)
    free = gr[max(tref - 70, 0):min(tref + 90, NT - 1), 2]

    # --- (i) |R| ---------------------------------------------------------
    R = np.abs(reflected).max() / np.abs(free).max()
    print(f"reflection coefficient |R| = {R:.4f}  (want 1.00)")
    if abs(R - 1.0) > 0.05:
        print("FAIL |R|"); ok = False

    # --- (ii) inversion --------------------------------------------------
    rho = nxcorr(reflected, incident)
    print(f"reflected-vs-incident correlation = {rho:+.4f}  "
          f"(want ~-1; a RIGID surface would give ~+1)")
    if rho > -0.9:
        print("FAIL inversion -- surface is not pressure-release"); ok = False

    # --- (iii) transmission ----------------------------------------------
    trans = np.abs(gs[:, 1]).max() / np.abs(gr[:, 1]).max()
    print(f"transmission past the sheet = {trans:.3e} of the free-field peak")
    if trans > 0.02:
        print("FAIL transmission"); ok = False

    # --- (iv) ncoordszero = 0 is byte-identical to no icczero.dat --------
    z0, none = os.path.join(work, "Z0"), os.path.join(work, "NONE")
    write_run(z0, sheet=True, nzero=0)
    write_run(none, sheet=False)
    for d in (z0, none):
        run_bin(batch_bin, d)
    b0 = open(os.path.join(z0, "genout.dat"), "rb").read()
    bn = open(os.path.join(none, "genout.dat"), "rb").read()
    print(f"ncoordszero=0 vs no icczero.dat: "
          f"{'byte-identical' if b0 == bn else 'MISMATCH'}")
    if b0 != bn:
        print("FAIL disabled-channel identity"); ok = False

    # --- (v) batch == solo with the channel active -----------------------
    solo = os.path.join(work, "SOLO")
    write_run(solo, sheet=True)
    run_bin(solo_bin, solo)
    bb = open(os.path.join(sh, "genout.dat"), "rb").read()
    bs = open(os.path.join(solo, "genout.dat"), "rb").read()
    print(f"batch vs solo with pzero active: "
          f"{'byte-identical' if bb == bs else 'MISMATCH'}")
    if bb != bs:
        print("FAIL batch==solo"); ok = False

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
