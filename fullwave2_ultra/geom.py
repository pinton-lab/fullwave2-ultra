# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gianmarco Pinton
"""Transducer geometry for the C5-2v curvilinear array (numpy) + coord
rasterization helpers (2D and 3D).

``make_xdc_c5_2v`` is a faithful, deterministic builder. It uses, throughout:

  * 1-based column-major linear indexing in the circle rasterization
    (``circle_idx``);
  * the ``map_to_coords`` convention of returning 0-based ``[i, j]`` grid
    coordinates;
  * stable sort of the coords by the first (row / lateral) column, which
    preserves the column-major (increasing j) order within each row;
  * round half away from zero in the rasterization.

The returned ``incoords``/``outcoords`` are integer grid coordinates with 4
columns ``[i, j, val, label]``. ``incoords2`` / ``outcoords2`` are the
per-element (sub-element-averaged) center coordinates.

:func:`map_to_coords_3d` is the general 3D analogue of :func:`map_to_coords` --
it turns a nonzero volume into ``[i, j, k, val]`` coords for the 3D .dat contract.

LOCAL / deterministic only -- no randomness, no I/O.
"""
from __future__ import annotations

import numpy as np

__all__ = ["make_xdc_c5_2v", "circle_idx", "map_to_coords", "map_to_coords_3d",
           "coords_matrix", "coords_matrix_3d", "volume_recorders"]


def _mround(x):
    """Round half away from zero (numpy rounds half to even)."""
    x = np.asarray(x, dtype=np.float64)
    return np.floor(np.abs(x) + 0.5) * np.sign(x)


def circle_idx(dims, cen, rad):
    """Rasterize a filled circle.

    Returns 1-based column-major linear indices of the pixels of a
    ``dims = (nX, nY)`` grid that fall inside the circle of center ``cen``
    (``[cen_x, cen_y]``, 1-based grid units) and radius ``rad``.

    Note: this may contain duplicate indices; the caller only uses them to set a
    binary map, so duplicates are harmless.
    """
    nX, nY = int(dims[0]), int(dims[1])
    cx, cy = float(cen[0]), float(cen[1])
    crad = int(np.ceil(rad))

    # Vectorized form of the i-outer/j-inner double loop. meshgrid(indexing 'ij')
    # ravels i-outer/j-inner (C-order), matching the loop order exactly; the same
    # np.sqrt(i^2+j^2)<=rad test and _mround keep the result identical to the
    # scalar loop -- just ~80x faster (the loop is O((2*ceil(rad))^2), ~2e6 iters
    # at ppw=6). Duplicate indices are preserved (harmless: the only consumer sets
    # a binary inmap).
    rng = np.arange(-crad, crad + 1)
    ii, jj = np.meshgrid(rng, rng, indexing="ij")          # i outer, j inner
    inside = np.sqrt(ii * ii + jj * jj) <= rad
    ii, jj = ii[inside], jj[inside]
    # round(i+cen(1)) + round(j+cen(2))*dims(1)
    idx = _mround(ii + cx).astype(np.int64) + _mround(jj + cy).astype(np.int64) * nX
    idx = idx[(idx > 0) & (idx <= nX * nY)]
    return idx


def map_to_coords(map2d):
    """Nonzero coordinates of a 2D map.

    Returns an ``(N, 3)`` float array ``[i, j, val]`` where ``i``/``j`` are
    0-based grid coordinates of the nonzero entries of ``map2d``, in
    column-major scan order.
    """
    map2d = np.asarray(map2d)
    # column-major nonzero scan: numpy column-major nonzero == flatten order 'F'.
    cols, rows = np.nonzero(map2d.T)  # rows of map2d.T == cols of map2d
    # cols here = column index of map2d (the "outer" / slow index in 'F')
    # rows here = row index of map2d (the "inner" / fast index in 'F')
    i = rows  # row index
    j = cols  # col index
    vals = map2d[i, j]
    # coords = [i, j, vals], 0-based.
    coords = np.column_stack([i.astype(np.float64),
                              j.astype(np.float64),
                              vals.astype(np.float64)])
    return coords


def map_to_coords_3d(vol):
    """General 3D analogue of map_to_coords: nonzero voxels of a (nX,nY,nZ) volume
    to an ``(N, 4)`` array ``[i, j, k, val]``, 0-based, in C-order (i slowest,
    k fastest) -- the order the 3D solver expects when coords are written
    ``[all i ; all j ; all k]`` (see :func:`io_dat.write_coords_3d`)."""
    vol = np.asarray(vol)
    i, j, k = np.nonzero(vol)          # C-order: i slowest, k fastest
    vals = vol[i, j, k]
    return np.column_stack([i.astype(np.float64), j.astype(np.float64),
                            k.astype(np.float64), vals.astype(np.float64)])


def _NxNyNz(N):
    """Normalize ``N`` to ``(Nx, Ny, Nz)`` -- a scalar means a cubic grid."""
    N = np.atleast_1d(np.asarray(N, dtype=np.int64))
    if N.size == 1:
        return int(N[0]), int(N[0]), int(N[0])
    return int(N[0]), int(N[1]), int(N[2])


def coords_matrix(nX, nY, modX=1, modY=1):
    """Strided 2D output-coordinate grid -- every ``modX``-th / ``modY``-th point.

    0-based grid: the ``for i=0:modX:nX, for j=0:modY:nY`` (i-outer, j-inner)
    loop == C-order over (i, j). Returns ``(n, 2)`` int ``[i, j]``, 0-based to
    match the rest of the pipeline. The spatial analogue of ``modT``.
    """
    I, J = np.meshgrid(np.arange(0, nX, modX), np.arange(0, nY, modY), indexing="ij")
    return np.column_stack([I.ravel(order="C"), J.ravel(order="C")]).astype(np.int64)


def coords_matrix_3d(nX, nY, nZ, modX=1, modY=1, modZ=1):
    """Strided 3D output-coordinate grid -- every ``(modX, modY, modZ)``-th point.

    0-based grid (i-outer / j-mid / k-inner loop == C-order, matching
    :func:`map_to_coords_3d` and ``io_dat.write_coords_3d``).
    Returns ``(n, 3)`` int ``[i, j, k]``.
    """
    I, J, K = np.meshgrid(np.arange(0, nX, modX), np.arange(0, nY, modY),
                          np.arange(0, nZ, modZ), indexing="ij")
    return np.column_stack([I.ravel(order="C"), J.ravel(order="C"),
                            K.ravel(order="C")]).astype(np.int64)


def volume_recorders(N, modX, modY, modZ):
    """``ndgrid(0:modX:N-1, ...)`` decimated volume recorders (column-major /
    ndgrid order). Returns ``(n, 5)`` ``[x, y, z, -1, 2]`` (col4 = element id / -1,
    col5 = role: 2 = recorder), usable directly as solver output coords. ``N`` is
    a scalar (cubic) or per-axis ``(Nx, Ny, Nz)``.

    NOTE the row order differs from :func:`coords_matrix_3d`: this mirrors the
    ndgrid ``(:)`` (column-major, i-fastest) launchers; ``coords_matrix_3d``
    uses C-order (i-slowest). Same index set.
    """
    Nx, Ny, Nz = _NxNyNz(N)
    JX, JY, JZ = np.meshgrid(np.arange(0, Nx, modX), np.arange(0, Ny, modY),
                             np.arange(0, Nz, modZ), indexing="ij")
    vol = np.empty((JX.size, 5), dtype=np.float64)
    vol[:, 0] = JX.ravel(order="F"); vol[:, 1] = JY.ravel(order="F"); vol[:, 2] = JZ.ravel(order="F")
    vol[:, 3] = -1; vol[:, 4] = 2
    return vol


def make_xdc_c5_2v(lam, freq_div, wX, wY):
    """Build the C5-2v transducer geometry on the simulation grid.

    Parameters
    ----------
    lam : float
        Acoustic wavelength [m].
    freq_div : float
        Sub-element frequency divisor (controls ppw / sub-element pitch).
    wX, wY : float
        Lateral and depth extents of the grid [m].

    Returns
    -------
    dict with keys: ``ppw, nX, nY, dX, dY, xdc``. ``xdc`` is itself a dict with
    fields: ``ptch, rad, npx, dTheta, thetas, cen, inmap, incoords, outcoords,
    thetas_in, thetas_out, incoords2, outcoords2, nOutPx, nInPx, surf``.
    """
    # --- grid parameters based on ppw -------------------------------------
    ptch_m = 0.508e-3                      # physical pitch [m]
    subelem_ptch = 15.0 / freq_div         # sub-elements per element
    ppw = lam / (ptch_m / subelem_ptch)    # points per wavelength
    nX = int(_mround(wX / lam * ppw))
    nY = int(_mround(wY / lam * ppw))
    dX = lam / ppw
    dY = lam / ppw

    # --- transducer variables --------------------------------------------
    zero_offset = 12.4e-3                   # how far the face comes into grid
    xdc = {}
    xdc["rad"] = 4.957e-2 / dY              # radius in pixel-idx units
    xdc["npx"] = 128
    xdc["ptch"] = 15.0 / freq_div           # pitch in sub-element units
    xdc["dTheta"] = np.arctan2(xdc["ptch"], xdc["rad"])
    npx = xdc["npx"]
    dTheta = xdc["dTheta"]
    # thetas initialized then immediately overwritten below; we keep only the
    # final value.
    xdc["cen"] = np.array([nX / 2.0, zero_offset / dY - xdc["rad"]],
                          dtype=np.float64)

    # --- thetas at the center of each element -----------------------------
    thetas = np.array([n * dTheta for n in range(1, npx + 1)], dtype=np.float64)
    thetas = thetas - np.mean(thetas)
    xdc["thetas"] = thetas

    # --- transducer-surface circle ----------------------------------------
    inmap = np.zeros((nX, nY), dtype=np.float64)
    outmap = np.zeros((nX, nY), dtype=np.float64)
    lin = circle_idx((nX, nY), xdc["cen"], xdc["rad"])
    # linear index (column-major, 1-based) -> set inmap = 1
    li0 = lin - 1
    rows = li0 % nX
    cols = li0 // nX
    inmap[rows, cols] = 1.0

    # --- edge coords (inmap inner edge, outmap outer edge) ----------------
    for i in range(nX):
        # find inmap coords: first column where inmap(i,:)==0  (1-based j)
        zero_cols = np.nonzero(inmap[i, :] == 0)[0]  # 0-based
        if zero_cols.size > 0:
            j = zero_cols[0] + 1                     # 1-based
            # inmap(i, 1:max([j-8 0])) = 0
            top = max(j - 8, 0)                      # 1-based inclusive count
            if top > 0:
                inmap[i, 0:top] = 0.0                # cols 1..top (0-based 0:top)

        # find outmap coords: last column where inmap(i,:)==1, +2
        one_cols = np.nonzero(inmap[i, :] == 1)[0]   # 0-based
        if one_cols.size > 0:
            j = (one_cols[-1] + 1) + 2               # 1-based last +2
            if 1 <= j <= nY:
                outmap[i, j - 1] = 1.0

    xdc["inmap"] = inmap

    # --- incoords / outcoords (sorted by lateral column) ------------------
    incoords = map_to_coords(inmap)
    order = np.argsort(incoords[:, 0], kind="stable")  # sort by col 1 (row)
    incoords = incoords[order]

    outcoords = map_to_coords(outmap)
    order = np.argsort(outcoords[:, 0], kind="stable")
    outcoords = outcoords[order]

    # --- per-coord element angles -----------------------------------------
    cen = xdc["cen"]
    thetas_in = np.arctan2(incoords[:, 0] - cen[0], incoords[:, 1] - cen[1])
    thetas_out = np.arctan2(outcoords[:, 0] - cen[0], outcoords[:, 1] - cen[1])
    xdc["thetas_in"] = thetas_in
    xdc["thetas_out"] = thetas_out

    # --- 4-column form with val / label columns ---------------------------
    incoords4 = np.zeros((incoords.shape[0], 4), dtype=np.float64)
    incoords4[:, 0:2] = incoords[:, 0:2]
    # incoords keeps its val column (col3) from map_to_coords (==1); col4 (label)
    # stays 0 here and is filled per-element below.
    incoords4[:, 2] = incoords[:, 2]

    outcoords4 = np.zeros((outcoords.shape[0], 4), dtype=np.float64)
    outcoords4[:, 0:2] = outcoords[:, 0:2]
    outcoords4[:, 2] = 1.0   # outcoords(:,3) = 1
    # outcoords(:,4) = 0

    incoords2 = np.zeros((npx, 2), dtype=np.float64)
    outcoords2 = np.zeros((npx, 2), dtype=np.float64)

    for tt in range(npx):  # 0-based; label value = tt+1
        th = thetas[tt]
        lo = th - dTheta / 2.0
        hi = th + dTheta / 2.0

        sel_in = (thetas_in < hi) & (thetas_in > lo)
        incoords4[sel_in, 3] = tt + 1
        incoords2[tt, 0] = np.mean(incoords4[sel_in, 0]) if sel_in.any() else np.nan
        incoords2[tt, 1] = np.mean(incoords4[sel_in, 1]) if sel_in.any() else np.nan

        sel_out = (thetas_out < hi) & (thetas_out > lo)
        outcoords4[sel_out, 3] = tt + 1
        outcoords2[tt, 0] = np.mean(outcoords4[sel_out, 0]) if sel_out.any() else np.nan
        outcoords2[tt, 1] = np.mean(outcoords4[sel_out, 1]) if sel_out.any() else np.nan

    xdc["incoords"] = incoords4
    xdc["outcoords"] = outcoords4
    xdc["incoords2"] = incoords2
    xdc["outcoords2"] = outcoords2
    xdc["nOutPx"] = thetas_out.shape[0]
    xdc["nInPx"] = thetas_in.shape[0]

    # --- surface label vector ---------------------------------------------
    surf = np.zeros(nX, dtype=np.float64)
    half = int(_mround(ppw / 2.0))
    for i in range(nX):
        one_cols = np.nonzero(inmap[i, :] == 1)[0]  # 0-based
        if one_cols.size > 0:
            j = one_cols[-1] + 1                     # 1-based last
        else:
            j = 1
        surf[i] = j + half
    xdc["surf"] = surf

    return {
        "ppw": ppw,
        "nX": nX,
        "nY": nY,
        "dX": dX,
        "dY": dY,
        "xdc": xdc,
    }
