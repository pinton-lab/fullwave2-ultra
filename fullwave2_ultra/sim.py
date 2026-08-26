# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gianmarco Pinton
"""Assemble a fullwave solver run directory (.dat) from maps + transducer coords.

This is the GENERAL run-directory writer (the abdominal-FSA wrapper and its
hardcoded constants are intentionally NOT included -- only this reusable
assembler is). It composes :mod:`medium` and :mod:`io_dat` to write exactly the
solver-used inputs (the CUDA Aexp kernels ignore the PML arrays, so those files
are not written).

The solver's internal finite-difference coefficient inputs are computed by the
solver binary itself and are NOT emitted by this assembler.

:func:`write_fullwave_sim` is the 2D assembler with the default ``nbdy=40,
M=8``; :func:`write_fullwave_sim_3d` is the volumetric analogue for the 3D
solver.
"""
from __future__ import annotations
import os
import numpy as np

from . import io_dat
from .medium import extendMap, aexp_from_amap, dbmhzcm2aexp
from .stability import check_cfl


def _mround(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def write_fullwave_sim(outdir, c0, omega0, dur, ppw, cfl, maps, xdc, nTic, modT,
                       *, nbdy=40, M=8, genout_mod=None,
                       add_coords=None, nTic_add=None,
                       cfl_check="raise", zero_coords=None, zero_window=None,
                       source_zero_window=None):
    """Write the solver-used .dat for one 2D run dir (PML files omitted).

    Parameters
    ----------
    outdir : str            run directory (created if needed).
    c0, omega0 : float      reference sound speed [m/s], angular freq [rad/s].
    dur : float             simulation duration [s] (sets nT).
    ppw, cfl : float        points-per-wavelength, Courant number.
    maps : dict             interior maps (nX,nY): cmap, rmap, nmap, amap.
    xdc : dict              must hold incoords, outcoords ([i,j,...], 0-based).
    nTic, modT : int        source length, genout decimation.
    nbdy, M : int           absorbing-ring width and FD half-width;
                            mext = nbdy + M is the replicate-pad applied to maps.
    add_coords : array|None ADDITIVE source coords (n,2+), same interior
                            convention as incoords. Superimposed on the field
                            (p += trace) instead of hard-set, so dense source
                            regions stay acoustically transparent. Coords must
                            be unique (the kernel adds once per coord per step).
                            The caller writes the traces separately:
                            io_dat.write_icmat("icmat_add.dat", blocks) with
                            per-sim (n, nTic_add) blocks, like icmat.
    nTic_add : int|None     additive trace length (defaults to nTic in-binary).
    zero_coords : array|None
                            PRESSURE-RELEASE coords (n, >=2), same 1-based
                            interior convention as incoords. The solver holds
                            p = 0 there every step of its window, AFTER the
                            source injection. That models a tissue/air
                            interface: |R| = 1 with a pi phase reversal, and no
                            material contrast, so none of the explicit-scheme
                            stability cost of a real gas. Coords need NOT be
                            unique -- assignment is idempotent, unlike the
                            additive channel's sum. 2D only.
                            NOTE a one-cell sheet does NOT block an M-tap
                            scheme; use 2M+1 cells. See docs/io_contract.md.
    zero_window : (n0, n1)|None
                            Steps over which zero_coords is held, half-open.
                            DEFAULT (0, nTic) -- the historical behaviour, in
                            which the sheet existed only while the source
                            played. A reflection measurement needs (0, nT):
                            the echo arrives long after a short burst.
    source_zero_window : (n0, n1)|None
                            Steps over which the SOURCE cells are clamped to 0.
                            DEFAULT (nTic, nT) -- the historical behaviour.
                            Pass (nT, nT) to leave them evolving freely after
                            the burst, as the 3D solver does; that also stops a
                            returning echo re-reflecting off the source line.

    Returns a dict of the realized scalars (nXe,nYe,nT,dX,dY,dT,ncoords,...).
    Does NOT write icmat / nsims -- the caller writes the source(s) separately.
    """
    os.makedirs(outdir, exist_ok=True)
    lam = c0 / omega0 * 2 * np.pi
    dX = lam / ppw
    dY = dX
    dT = dY / c0 * cfl
    nT = int(_mround(dur * c0 / lam * ppw / cfl))
    mext = nbdy + M

    c = extendMap(maps["cmap"], mext)
    rho = extendMap(maps["rmap"], mext)
    beta = extendMap(maps["nmap"], mext)
    A_ext = extendMap(maps["amap"], mext)
    K = c ** 2 * rho
    nXe, nYe = c.shape
    # `cfl` is referenced to c0; stability is set by the FASTEST material, so the
    # realized ratio is cfl*max(c)/c0. See stability.check_cfl.
    check_cfl(float(np.max(c)), dT, dX, c0=c0, M=M, dim=2, on_violation=cfl_check)

    Aexp = aexp_from_amap(A_ext, c0, omega0, dT, nbdy=nbdy)

    inc = np.asarray(xdc["incoords"]); outc = np.asarray(xdc["outcoords"])
    ncoords, ncoordsout = inc.shape[0], outc.shape[0]
    # coords + (nbdy+M) then writeCoords(coords-1)  -> +mext-1
    ic_i = inc[:, 0].astype(np.int64) + mext - 1
    ic_j = inc[:, 1].astype(np.int64) + mext - 1
    oc_i = outc[:, 0].astype(np.int64) + mext - 1
    oc_j = outc[:, 1].astype(np.int64) + mext - 1

    p = lambda f: os.path.join(outdir, f)
    io_dat.write_map(p("c.dat"), c);   io_dat.write_map(p("rho.dat"), rho)
    io_dat.write_map(p("K.dat"), K);   io_dat.write_map(p("beta.dat"), beta)
    io_dat.write_map(p("Aexp.dat"), Aexp)
    io_dat.write_coords(p("icc.dat"), ic_i, ic_j)
    io_dat.write_coords(p("outc.dat"), oc_i, oc_j)
    for name, v in [("nX", nXe), ("nY", nYe), ("nT", nT), ("ncoords", ncoords),
                    ("ncoordsout", ncoordsout), ("nTic", nTic), ("modT", modT),
                    ("ncoordszero", 0)]:
        io_dat.write_int(p(f"{name}.dat"), v)
    if zero_coords is not None and len(zero_coords):   # opt-in pzero sheet
        zc = np.asarray(zero_coords)
        if zc.ndim != 2 or zc.shape[1] < 2:
            raise ValueError(f"zero_coords must be (n, >=2) [i, j]; got {zc.shape}")
        zi = zc[:, 0].astype(np.int64) + mext - 1
        zj = zc[:, 1].astype(np.int64) + mext - 1
        oob = (zi < 0) | (zi >= nXe) | (zj < 0) | (zj >= nYe)
        if oob.any():
            raise ValueError(f"{int(oob.sum())} of {zc.shape[0]} zero_coords fall "
                             f"outside the extended grid ({nXe}, {nYe})")
        io_dat.write_coords(p("icczero.dat"), zi, zj)
        io_dat.write_int(p("ncoordszero.dat"), int(zc.shape[0]))
    # Window scalars are written ONLY when asked for. Absent means the solver's
    # own defaults -- (0, nTic) and (nTic, nT) -- so a deck that does not use
    # them is byte-identical to one built before the windows existed.
    for win, names in ((zero_window, ("nzero0", "nzero1")),
                       (source_zero_window, ("nszero0", "nszero1"))):
        if win is None:
            continue
        n0, n1 = (int(x) for x in win)
        if n0 < 0 or n1 < n0:
            raise ValueError(f"window {names} must satisfy 0 <= n0 <= n1; got {win}")
        io_dat.write_int(p(f"{names[0]}.dat"), n0)
        io_dat.write_int(p(f"{names[1]}.dat"), n1)
    for name, v in [("dX", dX), ("dY", dY), ("dT", dT), ("c0", c0)]:
        io_dat.write_float(p(f"{name}.dat"), v)
    if genout_mod is not None:               # opt-in genout_mod field-dump decimation (2D)
        mods = (int(genout_mod),) * 2 if np.isscalar(genout_mod) else tuple(int(x) for x in genout_mod)
        for name, v in zip(("modX", "modY"), mods):
            io_dat.write_int(p(f"{name}.dat"), v)
    ncoords_add = 0
    if add_coords is not None:               # opt-in additive source channel
        addc = np.asarray(add_coords)
        ncoords_add = addc.shape[0]
        ac_i = addc[:, 0].astype(np.int64) + mext - 1
        ac_j = addc[:, 1].astype(np.int64) + mext - 1
        io_dat.write_coords(p("icc_add.dat"), ac_i, ac_j)
        io_dat.write_int(p("ncoords_add.dat"), ncoords_add)
        if nTic_add is not None:
            io_dat.write_int(p("nTic_add.dat"), int(nTic_add))
    return dict(nXe=nXe, nYe=nYe, nT=nT, dX=dX, dY=dY, dT=dT,
                ncoords=ncoords, ncoordsout=ncoordsout, ncoords_add=ncoords_add)


def _sponge_nd(shape, nbdy):
    """sin^2 absorbing-ring multiplier, ones in the interior, on every face of an
    N-D grid. Generalizes the 2D aexp_from_amap sponge to 3D (and beyond)."""
    out = np.ones(shape, dtype=np.float64)
    for axis, n in enumerate(shape):
        s = np.ones(n, dtype=np.float64)
        for b in range(1, nbdy + 1):
            v = np.sin(np.pi / 2.0 * (b - 1) / nbdy) ** 2
            s[b - 1] = min(s[b - 1], v)
            s[n - b] = min(s[n - b], v)
        out *= s.reshape([n if a == axis else 1 for a in range(len(shape))])
    return out


def write_fullwave_sim_3d(outdir, c0, omega0, dur, ppw, cfl, maps,
                          incoords, outcoords, nTic, modT,
                          *, dZ=None, nbdy=8, M=8, genout_mod=None,
                          cfl_check="raise", zero_coords=None):
    """Write the solver-used .dat for one 3D run dir (volumetric analogue).

    maps : dict of interior volumes (nX,nY,nZ): cmap, rmap, nmap, amap.
    incoords/outcoords : (N, >=3) arrays of 0-based [i,j,k,...] interior coords.
    Aexp = dbmhzcm2aexp(amap_ext) * 6-face sin^2 sponge. Returns realized scalars.
    Does NOT write icmat / nsims -- the caller writes the source separately.
    """
    if zero_coords is not None and len(zero_coords):
        raise NotImplementedError(
            "the pressure-release (icczero) channel is 2D-only: the 3D solvers "
            "have no update_p_zero kernel and never read icczero.dat, so the "
            "coords would be silently ignored. Model the interface as a "
            "material contrast in 3D, or run 2D.")
    os.makedirs(outdir, exist_ok=True)
    lam = c0 / omega0 * 2 * np.pi
    dX = lam / ppw
    dY = dX
    dZ = dX if dZ is None else dZ
    dT = dX / c0 * cfl
    nT = int(_mround(dur * c0 / lam * ppw / cfl))
    mext = nbdy + M

    pad = ((mext, mext), (mext, mext), (mext, mext))
    c = np.pad(np.asarray(maps["cmap"], float), pad, mode="edge")
    rho = np.pad(np.asarray(maps["rmap"], float), pad, mode="edge")
    beta = np.pad(np.asarray(maps["nmap"], float), pad, mode="edge")
    A_ext = np.pad(np.asarray(maps["amap"], float), pad, mode="edge")
    K = c ** 2 * rho
    nXe, nYe, nZe = c.shape
    # No 3D stability limit ships in the public package, so this reports the
    # realized Courant ratio and warns when it exceeds the requested one -- the
    # reference-speed trap that NaNs a skull run -- without a threshold compare.
    check_cfl(float(np.max(c)), dT, dX, c0=c0, M=M, dim=3, on_violation=cfl_check)

    Aexp = dbmhzcm2aexp(A_ext, c0, omega0, dT) * _sponge_nd((nXe, nYe, nZe), nbdy)

    inc = np.asarray(incoords); outc = np.asarray(outcoords)
    ncoords, ncoordsout = inc.shape[0], outc.shape[0]
    ic = inc[:, :3].astype(np.int64) + mext - 1     # +mext then writeCoords(-1)
    oc = outc[:, :3].astype(np.int64) + mext - 1

    p = lambda f: os.path.join(outdir, f)
    io_dat.write_map_3d(p("c.dat"), c);   io_dat.write_map_3d(p("rho.dat"), rho)
    io_dat.write_map_3d(p("K.dat"), K);   io_dat.write_map_3d(p("beta.dat"), beta)
    io_dat.write_map_3d(p("Aexp.dat"), Aexp)
    io_dat.write_coords_3d(p("icc.dat"), ic[:, 0], ic[:, 1], ic[:, 2])
    io_dat.write_coords_3d(p("outc.dat"), oc[:, 0], oc[:, 1], oc[:, 2])
    for name, v in [("nX", nXe), ("nY", nYe), ("nZ", nZe), ("nT", nT),
                    ("ncoords", ncoords), ("ncoordsout", ncoordsout),
                    ("nTic", nTic), ("modT", modT),
                    ("ncoords_add", 0)]:
        io_dat.write_int(p(f"{name}.dat"), v)
    for name, v in [("dX", dX), ("dY", dY), ("dZ", dZ), ("dT", dT), ("c0", c0)]:
        io_dat.write_float(p(f"{name}.dat"), v)
    if genout_mod is not None:               # opt-in genout_mod field-dump decimation
        mods = (int(genout_mod),) * 3 if np.isscalar(genout_mod) else tuple(int(x) for x in genout_mod)
        for name, v in zip(("modX", "modY", "modZ"), mods):
            io_dat.write_int(p(f"{name}.dat"), v)
    return dict(nXe=nXe, nYe=nYe, nZe=nZe, nT=nT, dX=dX, dY=dY, dZ=dZ, dT=dT,
                ncoords=ncoords, ncoordsout=ncoordsout, mext=mext)
