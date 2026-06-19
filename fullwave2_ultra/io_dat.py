# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gianmarco Pinton
"""Raw .dat I/O for the fullwave solvers (single source of truth, 2D and 3D).

The CUDA solvers read/write raw little-endian binary files from a run directory
(CWD). This module is the Python implementation of that contract; see
``docs/io_contract.md``.

The 2D section is BYTE-IDENTICAL to the validated ``fw2b``
implementation (do not change it -- the parity gate depends on it). The 3D
helpers extend the same contract to volumetric maps and 3-column coords.

Conventions (2D)
----------------
* scalars  : one value per file, int32 (nX,nY,nT,nTic,modT,ncoords,ncoordszero,
             ncoordsout,ndmap,nsims) or float32 (dX,dY,dT,c0).
* maps     : nX*nY float32, ROW-MAJOR over (i,j) i.e. element [i,j] at i*nY+j
             (numpy C-order tofile() of a (nX,nY) array does exactly this).
* coords   : int32, COLUMN-MAJOR [all i ; all j], 0-based (the solver reads the
             first ncoords*2 ints as the i-column then the j-column).
* icmat    : nsims*ncoords*nTic float32, SIM-major then COORD-major then time:
             value(sim s, coord m, time n) at ((s*ncoords+m)*nTic + n).
* genout   : per-sim file, [frame][pt] frame-major float32; sim 0 -> genout.dat,
             sim s>0 -> genout_s<s>.dat; nframes = (nT-1)//modT.

Conventions (3D)
----------------
* scalars  : additionally nZ (int32), dZ (float32), ncoords_add (int32).
* maps     : nX*nY*nZ float32, C-ORDER over (i,j,k) i.e. [i,j,k] at (i*nY+j)*nZ+k.
* coords   : int32, [all i ; all j ; all k], 0-based (3*ncoords ints).
* icmat    : single-sim, COORD-major then time: value(coord m, time n) at m*nTic+n.
"""
from __future__ import annotations
import os
import numpy as np

# ---- scalars -------------------------------------------------------------
def write_int(path: str, v: int) -> None:
    np.asarray([v], dtype=np.int32).tofile(path)

def write_float(path: str, v: float) -> None:
    np.asarray([v], dtype=np.float32).tofile(path)

def read_int(path: str) -> int:
    return int(np.fromfile(path, dtype=np.int32, count=1)[0])

def read_float(path: str) -> float:
    return float(np.fromfile(path, dtype=np.float32, count=1)[0])

# ---- maps (nX,nY) --------------------------------------------------------
def write_map(path: str, M: np.ndarray) -> None:
    """M shape (nX, nY); written row-major so [i,j] -> i*nY+j."""
    np.ascontiguousarray(M, dtype=np.float32).tofile(path)

def read_map(path: str, nX: int, nY: int) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32, count=nX * nY).reshape(nX, nY)

def write_dcmap(path: str, dc: np.ndarray, ndmap: int) -> None:
    """Per-cell zone index map (int32). Clamped to [0, ndmap-1] -- guards against
    an out-of-range zone index (one reaching ndmap). Shape-agnostic: works for
    2D (nX,nY) and 3D (nX,nY,nZ) zone maps."""
    dc = np.clip(np.asarray(dc, dtype=np.int64), 0, ndmap - 1).astype(np.int32)
    np.ascontiguousarray(dc).tofile(path)

# ---- coords --------------------------------------------------------------
def write_coords(path: str, i_idx, j_idx) -> None:
    """0-based column-major [all i ; all j] int32 (2D)."""
    i_idx = np.asarray(i_idx, dtype=np.int32).ravel()
    j_idx = np.asarray(j_idx, dtype=np.int32).ravel()
    np.concatenate([i_idx, j_idx]).astype(np.int32).tofile(path)

# ---- sources (icmat) -----------------------------------------------------
def write_icmat(path: str, blocks) -> None:
    """blocks: iterable of (ncoords, nTic) float arrays, one per sim. Written
    sim-major, coord-major within a sim, time contiguous. For a single block
    this is the ordinary single-sim icmat (also the 3D layout)."""
    with open(path, "wb") as f:
        for blk in blocks:
            np.ascontiguousarray(blk, dtype=np.float32).tofile(f)  # (ncoords,nTic) row-major

# ---- outputs (genout) ----------------------------------------------------
def genout_name(sim: int) -> str:
    return "genout.dat" if sim == 0 else f"genout_s{sim}.dat"

def read_genout(path: str, ncoordsout: int) -> np.ndarray:
    """Return (nframes, ncoordsout) for one sim's genout file."""
    v = np.fromfile(path, dtype=np.float32)
    return v.reshape(-1, ncoordsout)

def read_genout_batch(rundir: str, nsims: int, ncoordsout: int) -> np.ndarray:
    """Return (nsims, nframes, ncoordsout)."""
    return np.stack([read_genout(os.path.join(rundir, genout_name(s)), ncoordsout)
                     for s in range(nsims)], axis=0)

# ==========================================================================
# 3D extensions (volumetric maps + 3-column coords). The 3D solver reads the
# same scalar/icmat contract plus nZ/dZ/ncoords_add and C-order volumes.
# ==========================================================================
def write_map_3d(path: str, V: np.ndarray) -> None:
    """V shape (nX, nY, nZ); C-order so [i,j,k] -> (i*nY+j)*nZ+k."""
    np.ascontiguousarray(V, dtype=np.float32).tofile(path)

def read_map_3d(path: str, nX: int, nY: int, nZ: int) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32, count=nX * nY * nZ).reshape(nX, nY, nZ)

def write_coords_3d(path: str, i_idx, j_idx, k_idx) -> None:
    """0-based [all i ; all j ; all k] int32 (3D); the solver reads
    coords[m]=i, coords[ncoords+m]=j, coords[2*ncoords+m]=k."""
    i_idx = np.asarray(i_idx, dtype=np.int32).ravel()
    j_idx = np.asarray(j_idx, dtype=np.int32).ravel()
    k_idx = np.asarray(k_idx, dtype=np.int32).ravel()
    np.concatenate([i_idx, j_idx, k_idx]).astype(np.int32).tofile(path)

def read_coords_3d(path: str, ncoords: int) -> np.ndarray:
    """Return (ncoords, 3) int array of [i, j, k] from a 3D coords file."""
    v = np.fromfile(path, dtype=np.int32, count=ncoords * 3)
    return np.stack([v[:ncoords], v[ncoords:2 * ncoords], v[2 * ncoords:3 * ncoords]], axis=1)

# ---- genout_mod (decimated full-field dump) ------------------------------
def genout_mod_dims(nX: int, nY: int, nZ: int, modX: int = 1, modY: int = 1, modZ: int = 1):
    """Decimated grid shape (nX2, nY2, nZ2) of a genout_mod dump (== len(0:mod:N))."""
    return ((nX + modX - 1) // modX, (nY + modY - 1) // modY, (nZ + modZ - 1) // modZ)

def read_genout_mod(path: str, nX: int, nY: int, nZ: int,
                    modX: int = 1, modY: int = 1, modZ: int = 1) -> np.ndarray:
    """Read a genout_mod.dat (the streamed, (modX,modY,modZ)-decimated FULL-field
    dump written by the 3D solver) -> ``(nframes, nX2, nY2, nZ2)`` float32, where
    nX2 = ceil(nX/modX) etc. ``nX,nY,nZ`` are the FULL (extended) grid the solver ran
    (the nX/nY/nZ scalars in the run dir); nframes is inferred from the file size.
    Layout is frame-major, C-order over (i,j,k) -- i.e. ``field[::modX,::modY,::modZ]``."""
    nX2, nY2, nZ2 = genout_mod_dims(nX, nY, nZ, modX, modY, modZ)
    v = np.fromfile(path, dtype=np.float32)
    return v.reshape(-1, nX2, nY2, nZ2)

def genout_mod_dims_2d(nX: int, nY: int, modX: int = 1, modY: int = 1):
    """Decimated grid shape (nX2, nY2) of a 2D genout_mod dump (== len(0:mod:N))."""
    return ((nX + modX - 1) // modX, (nY + modY - 1) // modY)

def genout_mod_name(sim: int) -> str:
    """Per-sim genout_mod filename (2D batch): sim 0 -> genout_mod.dat,
    sim s -> genout_mod_s<s>.dat (mirrors genout's per-sim naming)."""
    return "genout_mod.dat" if sim == 0 else f"genout_mod_s{sim}.dat"

def read_genout_mod_2d(path: str, nX: int, nY: int,
                       modX: int = 1, modY: int = 1) -> np.ndarray:
    """Read a 2D genout_mod.dat (the streamed, (modX,modY)-decimated FULL-field dump
    written by the 2D solvers) -> ``(nframes, nX2, nY2)`` float32, nX2 = ceil(nX/modX).
    ``nX,nY`` are the FULL (extended) grid the solver ran; nframes is inferred from the
    file size. Frame-major, C-order over (i,j) -- i.e. ``field[::modX,::modY]``. For the
    batch solver this is one sim's file; use ``read_genout_mod_batch_2d`` for all sims."""
    nX2, nY2 = genout_mod_dims_2d(nX, nY, modX, modY)
    v = np.fromfile(path, dtype=np.float32)
    return v.reshape(-1, nX2, nY2)

def read_genout_mod_batch_2d(rundir: str, nsims: int, nX: int, nY: int,
                             modX: int = 1, modY: int = 1) -> np.ndarray:
    """Stack the per-sim 2D genout_mod dumps -> ``(nsims, nframes, nX2, nY2)``."""
    return np.stack([read_genout_mod_2d(os.path.join(rundir, genout_mod_name(s)),
                                        nX, nY, modX, modY) for s in range(nsims)], axis=0)
