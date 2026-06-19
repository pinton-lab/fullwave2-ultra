# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gianmarco Pinton
"""Medium pipeline in numpy (resample tissue, regenerate scatterers, Aexp).

These are the genuinely general medium helpers (no abdominal-FSA constants, no
hardcoded tissue tables -- tissue variances are exposed as parameters, not baked
in). All public functions operate on (nX, nY) arrays with rows = nX =
lateral/element axis, cols = nY = depth axis.

Behaviour:
  * extendMap          -> replicate pad
  * resample_map       -> linear interpolation after a separable Gaussian
  * aexp_from_amap     -> dbmhzcm2aexp + sin^2 sponge
  * regen_scatterers   -> a sparse zero-mean scatterer field at this grid's
    density (nnz fraction ~ scat_density, per-tissue variance scales as the
    squared tissue multiplier from modify_scat_var, zero above the transducer
    surface). Uses numpy's PRNG, so the realization depends on the seed.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Tissue scatterer-variance multipliers. Several are random
# (var = a + b*rand), drawn with numpy's RNG, so the realization depends on the
# seed. The deterministic ones (liver=1, water=0, tissue=1) are fixed. We expose
# the formulas so callers can validate statistics: per-tissue nonzero variance
# scales as var[k]**2. This is a DEFAULT phantom spec; callers may pass their own
# var_table to tissue_var_table / regen_scatterers.
# ---------------------------------------------------------------------------
# name -> ('const', value)  or  ('rand', lo, span)  meaning value = lo + span*rand
_TISSUE_VAR_SPEC = {
    "fat":               ("rand", 0.5, 0.5),
    "liver":             ("const", 1.0),
    "muscle":            ("rand", 1.0, 0.5),
    "water":             ("const", 0.0),
    "skin":              ("rand", 0.75, 0.5),
    "tissue":            ("const", 1.0),
    "connective":        ("rand", 1.0, 0.5),
    "blood":             ("rand", 0.0, 0.1),   # 0.1*rand
    "epidermis":         ("rand", 0.75, 0.5),
    "papillary_dermis":  ("rand", 1.0, 0.5),
}


def tissue_var_table(rng: np.random.Generator | None = None,
                     seed: int | None = None,
                     spec: dict | None = None,
                     order: list | None = None) -> dict[str, float]:
    """Build the {tissue_name: var} table.

    The random entries are drawn with `rand` in the field-assignment order (fat,
    liver, muscle, water, skin, tissue, connective, blood, epidermis,
    papillary_dermis), so the table is reproducible within numpy for a given
    seed.

    Pass ``spec`` (name -> ('const',v) | ('rand',lo,span)) and/or ``order`` to
    override the default phantom table without editing this module.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    if spec is None:
        spec = _TISSUE_VAR_SPEC
    if order is None:
        # rand-draw order = field-assignment order
        order = ["fat", "liver", "muscle", "water", "skin", "tissue",
                 "connective", "blood", "epidermis", "papillary_dermis"]
    table: dict[str, float] = {}
    for name in order:
        s = spec[name]
        if s[0] == "const":
            table[name] = float(s[1])
        else:
            _, lo, span = s
            table[name] = float(lo + span * rng.random())
    return table


# ---------------------------------------------------------------------------
# extendMap -- replicate-pad by n on all sides.
# ---------------------------------------------------------------------------
def extendMap(M: np.ndarray, n: int) -> np.ndarray:
    """Replicate-pad (M) by (n) cells on every side -> (nX+2n, nY+2n).

    Center holds M; edges replicate the nearest border row/column; corners
    replicate the corner value. np.pad('edge') does exactly this.
    """
    M = np.asarray(M, dtype=np.float64)
    if n == 0:
        return M.copy()
    return np.pad(M, ((n, n), (n, n)), mode="edge")


# ---------------------------------------------------------------------------
# gblur -- separable Gaussian with replicate pad.
# ---------------------------------------------------------------------------
def _gblur(A: np.ndarray, sg: float) -> np.ndarray:
    """Separable Gaussian blur with replicate padding.

        r = ceil(3*sg); x=-r:r; h=exp(-x.^2/(2 sg^2)); h=h/sum(h)
        replicate-pad by r, then a 'valid' convolution.
    The 1-D h convolves rows then cols (separable); cross-correlation ==
    convolution since h is symmetric.
    """
    A = np.asarray(A, dtype=np.float64)
    if sg <= 0:
        return A.copy()
    r = int(np.ceil(3 * sg))
    x = np.arange(-r, r + 1, dtype=np.float64)
    h = np.exp(-x ** 2 / (2 * sg ** 2))
    h = h / h.sum()
    Ap = np.pad(A, ((r, r), (r, r)), mode="edge")
    # convolve along rows (axis=1) then cols (axis=0); symmetric h -> conv==corr
    # 'valid' over the r-padded array returns the original (m, n) shape.
    B = _conv1d_valid(Ap, h, axis=1)
    B = _conv1d_valid(B, h, axis=0)
    return B


def _conv1d_valid(A: np.ndarray, h: np.ndarray, axis: int) -> np.ndarray:
    """1-D 'valid' convolution of A with kernel h along (axis)."""
    if axis == 0:
        A = A.T
        out = _conv1d_valid(A, h, axis=1)
        return out.T
    k = h.size
    n = A.shape[1]
    out_n = n - k + 1
    # convolution: conv2(h, ., 'valid') == sum_j A[..,j:j+out_n]*h_reversed[j]
    hr = h[::-1]
    out = np.zeros((A.shape[0], out_n), dtype=np.float64)
    for j in range(k):
        out += A[:, j:j + out_n] * hr[j]
    return out


# ---------------------------------------------------------------------------
# resample_map: linear interpolation of the Gaussian-blurred map.
# ---------------------------------------------------------------------------
def _interp2_linear(A: np.ndarray, nX: int, nY: int) -> np.ndarray:
    """Bilinear resample of A (n0x, n0y) onto a regular (nX, nY) grid where row
    coords are linspace(0,n0x-1,nX) and col coords are linspace(0,n0y-1,nY).

    No extrapolation needed (queries are interior).
    """
    A = np.asarray(A, dtype=np.float64)
    n0x, n0y = A.shape
    xq = np.linspace(0.0, n0x - 1, nX)      # row sample positions (0-based)
    yq = np.linspace(0.0, n0y - 1, nY)      # col sample positions (0-based)

    # row interpolation
    x0 = np.floor(xq).astype(np.int64)
    x0 = np.clip(x0, 0, n0x - 2)
    wx = (xq - x0)[:, None]                 # (nX,1)
    A_row = A[x0, :] * (1 - wx) + A[x0 + 1, :] * wx   # (nX, n0y)

    # column interpolation
    y0 = np.floor(yq).astype(np.int64)
    y0 = np.clip(y0, 0, n0y - 2)
    wy = (yq - y0)[None, :]                 # (1,nY)
    out = A_row[:, y0] * (1 - wy) + A_row[:, y0 + 1] * wy   # (nX, nY)
    return out


def resample_map(M: np.ndarray, nX: int, nY: int, sigma: float) -> np.ndarray:
    """Anti-alias (separable Gaussian, sigma in native cells) then linearly
    resample (M) to (nX, nY)."""
    Mb = _gblur(M, sigma)
    return _interp2_linear(Mb, nX, nY)


def resample_labels(M: np.ndarray, nX: int, nY: int) -> np.ndarray:
    """Nearest-neighbour resample of a tissue-LABEL map, then round."""
    A = np.asarray(M, dtype=np.float64)
    n0x, n0y = A.shape
    xq = np.linspace(0.0, n0x - 1, nX)
    yq = np.linspace(0.0, n0y - 1, nY)
    # nearest uses round-half-away-from-zero; coords are >=0 so round-half-up.
    # np.rint is banker's rounding -> use explicit half-up.
    xi = np.floor(xq + 0.5).astype(np.int64)
    yi = np.floor(yq + 0.5).astype(np.int64)
    xi = np.clip(xi, 0, n0x - 1)
    yi = np.clip(yi, 0, n0y - 1)
    out = A[np.ix_(xi, yi)]
    # final round() (labels are already integral after nearest)
    return np.floor(out + 0.5)


# ---------------------------------------------------------------------------
# rescell2d
# ---------------------------------------------------------------------------
def rescell2d(lam: float, dy: float, ay: float, ncycles: float,
              dY: float, dZ: float) -> float:
    """resy=lam*dy/ay; resz=lam*ncycles/2; rescell=resy/dY * resz/dZ."""
    resy = lam * dy / ay
    resz = lam * ncycles / 2.0
    return (resy / dY) * (resz / dZ)


# ---------------------------------------------------------------------------
# modify_scat_var: scale each tissue's scatterers by var[k].
# ---------------------------------------------------------------------------
def modify_scat_var(scat: np.ndarray, label_map: np.ndarray,
                    labels: dict[int, str],
                    var_table: dict[str, float]) -> np.ndarray:
    """For every tissue index present in (label_map), multiply the corresponding
    cells of (scat) by var_table[labels[idx]]. Cells whose label is absent stay
    zero (the output is initialised to zeros)."""
    label_map = np.asarray(label_map)
    out = np.zeros_like(scat, dtype=np.float64)
    for idx in np.unique(label_map):
        idx_i = int(round(float(idx)))
        name = labels.get(idx_i)
        if name is None:
            continue
        var = var_table.get(name)
        if var is None:
            continue
        m = (label_map == idx)
        out[m] = scat[m] * var
    return out


# ---------------------------------------------------------------------------
# regen_scatterers
# ---------------------------------------------------------------------------
def regen_scatterers(map_labels: np.ndarray,
                     labels: dict[int, str],
                     ppw: float, nX: int, nY: int,
                     dX: float, dY: float, wY: float,
                     lam: float, ncycles: float,
                     surf: np.ndarray | None = None,
                     num_scat: int = 18, seed: int = 7,
                     var_table: dict[str, float] | None = None,
                     return_info: bool = False):
    """Regenerate a sparse zero-mean scatterer field at this grid's density.

    Steps:
        res_cell = rescell2d(lambda, nY/2*dY, wY, ncycles, dX, dY)
        scat_density = num_scat/res_cell
        rng(seed); scat = rand(nX,nY)/scat_density;
        scat[scat>1] = 0.5; scat = scat - 0.5         # sparse, zero-mean
        scat = modify_scat_var(scat, map_labels, labels)  # per-tissue variance
        for i: scat[i, :surf[i]-3] = 0                # none above tx surface

    Uses numpy's PRNG, so the realization depends on the seed; the statistics
    (nnz fraction, per-tissue variance scaling, zeroing above surf) are fixed.

    Returns scat (nX, nY). If return_info, also returns a dict with res_cell,
    scat_density, var_table, nnz, etc. for validation.
    """
    rng = np.random.default_rng(seed)
    if var_table is None:
        var_table = tissue_var_table(rng=rng)

    res_cell = rescell2d(lam, nY / 2.0 * dY, wY, ncycles, dX, dY)
    scat_density = num_scat / res_cell

    scat = rng.random((nX, nY)) / scat_density
    scat[scat > 1] = 0.5
    scat = scat - 0.5                                   # sparse zero-mean

    scat = modify_scat_var(scat, np.asarray(map_labels), labels, var_table)

    if surf is not None:
        surf = np.asarray(surf).ravel()
        for i in range(nX):
            s0 = min(nY, max(1, int(surf[i]) - 3))
            scat[i, :s0] = 0.0

    if return_info:
        info = dict(res_cell=res_cell, scat_density=scat_density,
                    var_table=var_table, nnz=int(np.count_nonzero(scat)))
        return scat, info
    return scat


# ---------------------------------------------------------------------------
# dbmhzcm2aexp
# ---------------------------------------------------------------------------
def dbmhzcm2aexp(alpha: np.ndarray, c0: float, omega0: float,
                 dT: float) -> np.ndarray:
    """alpha (dB/MHz/cm) -> per-cell per-step attenuation multiplier Aexp.
        np    = -db(exp(-1))            # dB per Neper, db(x)=20*log10|x|
        F0    = omega0/2/pi/1e6         # MHz
        texp  = alpha*F0*c0/1e-2/np     # Nepers/s
        aexp  = exp(-dT*texp)
    """
    npr = -20.0 * np.log10(np.exp(-1.0))      # -db(exp(-1)) = 20*log10(e) = +8.686
    F0 = omega0 / 2.0 / np.pi / 1e6
    texp = np.asarray(alpha, dtype=np.float64) * F0 * c0 / 1e-2 / npr
    return np.exp(-dT * texp)


# ---------------------------------------------------------------------------
# aexp_from_amap: dbmhzcm2aexp + sin^2 sponge.
# ---------------------------------------------------------------------------
def aexp_from_amap(amap_ext: np.ndarray, c0: float, omega0: float,
                   dT: float, nbdy: int = 40) -> np.ndarray:
    """Build the per-cell Aexp on the EXTENDED grid from the extended attenuation
    map (dB/MHz/cm), then overlay a sin^2 absorbing sponge on the outer (nbdy)
    ring (the Aexp kernel path ignores PML).

    amap_ext is expected already extended (e.g. extendMap(amap, nbdy+M)); the
    returned array has the same shape.
    """
    amap_ext = np.asarray(amap_ext, dtype=np.float64)
    nXe, nYe = amap_ext.shape
    Aexp = dbmhzcm2aexp(amap_ext, c0, omega0, dT)

    spx = np.ones(nXe, dtype=np.float64)
    spy = np.ones(nYe, dtype=np.float64)
    for b in range(1, nbdy + 1):                         # b = 1..nbdy
        v = np.sin(np.pi / 2.0 * (b - 1) / nbdy) ** 2
        spx[b - 1] = min(spx[b - 1], v)
        spx[nXe - b] = min(spx[nXe - b], v)             # nXe-b+1 (1-based)
        spy[b - 1] = min(spy[b - 1], v)
        spy[nYe - b] = min(spy[nYe - b], v)
    Aexp = Aexp * np.outer(spx, spy)
    return Aexp
