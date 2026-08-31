# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gianmarco Pinton
"""GPU gate: the 3D additive channel's source constants (icc_add / icmat_add).

The additive channel adds the per-step increment ``s_n`` to ``p`` at its cells.
What does that radiate? Two closed-form answers, both pinned here:

  SHEET  a one-cell sheet radiates, to EACH side, the plane wave
             p(t) = s(t - d/c) / (2 CFL)            CFL = c dT / dx, LOCAL c
         -- the drive waveform itself, no derivative, scaled by 1/(2 CFL).
         This is the injection-shell case: to inject pressure P, drive
         s = 2 CFL P (times cos(theta) at oblique incidence -- a monopole sheet
         has a jump condition on u_n, not on p).
  POINT  a single cell radiates the monopole field
             p(r, t) = C s'(t - r/c) / r,     C = dx^3 / (4 pi c^2 dT)
         -- 1/r decay, retarded time, and the TIME DERIVATIVE of the drive.

Derivation. Adding s per step is a mass source K q = s/dT over the cell. A
sheet then carries a jump [u_n] = s dx/(K dT) split half to each side, so
p = rho c [u_n]/2 = s dx/(2 c dT). A cell carries volume velocity
Q = s dx^3/(K dT) and radiates rho Q'/(4 pi r). Both are continuum limits;
this gate measures how closely the M=8 DRP scheme realizes them.

Measured (bench_3d_opt, CFL 0.30, homogeneous lossless, receivers on the three
axes at r = 4..20 cells / sheet receivers at d = 4..16 both sides):

                          12 ppw        6 ppw (production)
  point amplitude / pred  +0.5%         +1.9%      corr 1.0000
  point decay exponent    -0.9995       -0.9995    (expect -1)
  point axis anisotropy   0.00%         0.00%
  sheet amplitude / pred  +0.3%         +1.4%      corr >= 0.9998
  sheet +x vs -x          2e-7 of ptp   2e-7
  arrival, both           r/c - 0.50 dT, uniform in r and ppw

The excess scales ~ppw^-2 (second-order truncation, not a missing factor), and
the arrival is exactly HALF A STEP EARLY everywhere: the leapfrog stagger. Both
are pinned as such, at the resolution measured, rather than tolerated.

Everything is measured on ncoords = 0 decks -- the additive channel alone,
which is the shape the hybrid injection uses.

Invoked by tests/test_addsrc_greens.sh; run directly as:
    python tests/check_addsrc_greens.py <bench_3d_opt>
"""
import os
import subprocess
import sys
import tempfile
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fullwave2_ultra import io_dat, sim  # noqa: E402

C0, F0 = 1540.0, 1.0e6
OMEGA0 = 2 * np.pi * F0
CFL = 0.30                        # 3D limit is 0.4907 at M=8 (stencil.cfl_limit)
NBDY, MSTEN = 12, 8
NT, NTIC, N0 = 220, 100, 50       # burst centred at step 50, tau = half a period
POINT_R = (4, 6, 8, 12, 16, 20)   # cells from the source cell, along +x, +y, +z
SHEET_D = (4, 8, 12, 16)          # cells from the sheet, both sides

# tolerances: {ppw: (|ratio - 1|, min corr, |lag - (r/CFL + LAG0)| in steps)},
# set at ~2x the measured deviation so a real regression fails and rounding
# noise does not. LAG0 is the leapfrog half-step: the wave arrives dT/2 early.
TOL = {12.0: (0.015, 0.999, 0.15), 6.0: (0.04, 0.998, 0.15)}
LAG0 = -0.5
R_TIGHT = 6                       # point-source amplitude asserted for r >= this


MEXT = NBDY + MSTEN               # interior i -> extended i + MEXT - 1


def burst(n, ppw):
    """Gaussian-windowed sinusoid at F0 evaluated at (fractional) step n;
    envelope tau = half a period. Analytic so a prediction can be placed at a
    FRACTIONAL retarded time -- an integer-lag fit lands up to half a step
    off, which at 12 ppw costs 4.5 deg of phase and reads as a 0.5% amplitude
    loss and a corr dip on exactly the receivers whose r/CFL is half-integer."""
    P = ppw / CFL                 # steps per period at F0
    x = (np.asarray(n, float) - N0)
    return np.exp(-((x / (0.5 * P)) ** 2)) * np.sin(2 * np.pi * x / P)


def dburst(n, ppw, dT):
    """d/dt of burst(), analytic (per second)."""
    P = ppw / CFL
    x = (np.asarray(n, float) - N0)
    w = 0.5 * P
    e = np.exp(-((x / w) ** 2))
    dsdn = e * (-(2 * x / w ** 2) * np.sin(2 * np.pi * x / P)
                + (2 * np.pi / P) * np.cos(2 * np.pi * x / P))
    return dsdn / dT


def run_bin(binary, rundir):
    subprocess.run([binary], cwd=rundir, check=True, capture_output=True)


def deck(root, name, ppw, interior, cells_ext, drive, outcoords):
    """An ncoords = 0 deck with the additive channel driving `cells_ext`."""
    d = os.path.join(root, name)
    nx, ny, nz = interior
    z = np.zeros(interior)
    maps = dict(cmap=np.full(interior, C0), rmap=np.full(interior, 1000.0),
                nmap=z, amap=z.copy())
    meta = sim.write_fullwave_sim_3d(d, C0, OMEGA0, 1.0, ppw, CFL, maps,
                                     np.zeros((0, 3), int), outcoords, NTIC, 1,
                                     nbdy=NBDY, M=MSTEN)
    io_dat.write_int(os.path.join(d, "nT.dat"), NT)
    tr = np.broadcast_to(drive[:NTIC], (cells_ext.shape[0], NTIC))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)   # full-width sheet -> frozen ring
        sim.write_addsrc_rundir(d, cells_ext, tr)
    return d, meta


def genout_by_step(d, nout):
    """p^n at each receiver, indexed by the step n whose SOURCE sample s_n it
    follows. The loop at step n injects s_n into p^n, then produces p^{n+1} and
    records it as frame n-1 -- so frame f is p^{f+2}: two leading zeros."""
    g = io_dat.read_genout(os.path.join(d, "genout.dat"), nout)     # (nT-1, nout)
    return np.vstack([np.zeros((2, nout)), g])[:NT]                 # (nT, nout)


def fit(meas, predict, lag_pred, search=4.0, step=0.05):
    """Find the fractional lag in lag_pred +/- search (0.05-step grid) that
    maximizes the normalized correlation of meas with predict(lag); return
    (lag, LS amplitude ratio at that lag, corr)."""
    n = np.arange(NT, dtype=float)
    mm = float(np.dot(meas, meas))
    best = None
    for lag in np.arange(lag_pred - search, lag_pred + search + step / 2, step):
        p = predict(n - lag)
        p[n < lag] = 0.0
        den = float(np.dot(p, p))
        if den == 0:
            continue
        c = float(np.dot(meas, p)) / np.sqrt(den * mm + 1e-300)
        if best is None or c > best[2]:
            best = (float(lag), float(np.dot(meas, p)) / den, c)
    return best


def check_point(binary, root, ppw):
    n = 80
    ctr = n // 2
    axes = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}
    outc = [[ctr + r * a[0], ctr + r * a[1], ctr + r * a[2]]
            for r in POINT_R for a in axes.values()]
    outc = np.array(outc)
    d, meta = deck(root, f"point{int(ppw)}", ppw, (n, n, n),
                   np.array([[ctr, ctr, ctr]]) + MEXT - 1,
                   burst(np.arange(NT), ppw), outc)
    run_bin(binary, d)
    dT, dx = meta["dT"], meta["dX"]
    C = dx ** 3 / (4 * np.pi * C0 ** 2 * dT)
    p = genout_by_step(d, outc.shape[0])

    print(f"  [point, {ppw:g} ppw]  C = dx^3/(4 pi c^2 dT) = {C:.4e}   r/c in steps = r/CFL")
    print(f"    {'r':>3} {'axis':>4} {'ratio':>7} {'corr':>7} {'lag':>7} {'r/CFL':>6} {'-dT/2':>5}")
    tol_r, tol_c, tol_l = TOL[ppw]
    bad = []
    amps = {}
    k = 0
    for r in POINT_R:
        for ax in axes:
            rm = r * dx                                   # metres, not cells
            lag, ratio, corr = fit(p[:, k], lambda t: C * dburst(t, ppw, dT) / rm, r / CFL)
            dlag = lag - (r / CFL + LAG0)
            amps.setdefault(r, []).append(ratio)
            flag = ""
            if r >= R_TIGHT and (abs(1 - ratio) > tol_r or corr < tol_c or abs(dlag) > tol_l):
                flag = "  <-- FAIL"; bad.append((r, ax))
            print(f"    {r:>3} {ax:>4} {ratio:>7.4f} {corr:>7.4f} {lag:>7.2f} {r/CFL:>6.2f} {dlag:>+5.2f}{flag}")
            k += 1
    # 1/r: the LS ratio already divides by r, so its flatness IS the decay law;
    # report the fitted exponent of the raw amplitude too.
    rs = np.array([r for r in POINT_R if r >= R_TIGHT], float)
    raw = np.array([np.mean(amps[int(r)]) / r for r in rs])
    slope = np.polyfit(np.log(rs), np.log(raw), 1)[0]
    iso = max(np.ptp(amps[r]) / np.mean(amps[r]) for r in POINT_R if r >= R_TIGHT)
    print(f"    decay exponent (r >= {R_TIGHT}): {slope:+.4f}   (expect -1)"
          f"    axis anisotropy: {100*iso:.2f}%")
    ok = not bad and abs(slope + 1) < 0.02 and iso < 0.005
    print(f"  {'PASS' if ok else 'FAIL'} point monopole at {ppw:g} ppw")
    return ok, dict(C=C, ratios={r: float(np.mean(v)) for r, v in amps.items()},
                    slope=float(slope), iso=float(iso))


def check_sheet(binary, root, ppw):
    nx, nl = 80, 96
    i0, cl = nx // 2, nl // 2
    nle = nl + 2 * MEXT                       # extended lateral size (whole grid)
    jj, kk = np.meshgrid(np.arange(nle), np.arange(nle), indexing="ij")
    sheet = np.stack([np.full(jj.size, i0 + MEXT - 1), jj.ravel(), kk.ravel()], 1)
    outc = np.array([[i0 + sgn * dd, cl, cl] for dd in SHEET_D for sgn in (+1, -1)])
    d, meta = deck(root, f"sheet{int(ppw)}", ppw, (nx, nl, nl), sheet,
                   burst(np.arange(NT), ppw), outc)
    assert meta["nYe"] == nle, (meta["nYe"], nle)
    run_bin(binary, d)
    p = genout_by_step(d, outc.shape[0])
    print(f"  [sheet, {ppw:g} ppw]  predicted p = s(t - d/c) / (2 CFL) = s / {2*CFL:.2f}")
    print(f"    {'d':>3} {'side':>4} {'ratio':>7} {'corr':>7} {'lag':>7} {'d/CFL':>6} {'-dT/2':>5}")
    tol_r, tol_c, tol_l = TOL[ppw]
    bad = []
    ratios = []
    k = 0
    for dd in SHEET_D:
        for sgn in (+1, -1):
            lag, ratio, corr = fit(p[:, k], lambda t: burst(t, ppw) / (2 * CFL), dd / CFL)
            dlag = lag - (dd / CFL + LAG0)
            ratios.append(ratio)
            flag = ""
            if abs(1 - ratio) > tol_r or corr < tol_c or abs(dlag) > tol_l:
                flag = "  <-- FAIL"; bad.append((dd, sgn))
            print(f"    {dd:>3} {'+x' if sgn > 0 else '-x':>4} {ratio:>7.4f} {corr:>7.4f} "
                  f"{lag:>7.2f} {dd/CFL:>6.2f} {dlag:>+5.2f}{flag}")
            k += 1
    # the two sides must agree (the sheet is symmetric): report the spread
    sym = max(abs(p[:, 2 * i] - p[:, 2 * i + 1]).max() for i in range(len(SHEET_D)))
    print(f"    +x / -x max |difference| = {sym:.3e}   (ptp {np.ptp(p):.4g})")
    ok = not bad and sym < 1e-5 * np.ptp(p)
    print(f"  {'PASS' if ok else 'FAIL'} sheet at {ppw:g} ppw")
    return ok, dict(ratios=[float(x) for x in ratios])


def main(binary):
    ok = True
    with tempfile.TemporaryDirectory() as root:
        for ppw in (12.0, 6.0):
            o, _ = check_point(binary, root, ppw); ok &= o
            o, _ = check_sheet(binary, root, ppw); ok &= o
    print("PASS 3D additive source constants" if ok else "FAIL 3D additive source constants")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
