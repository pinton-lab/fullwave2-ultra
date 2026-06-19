"""Structural invariants for fullwave2_ultra.geom (CPU, clean checkout, no fw2b).

These do NOT require the original fw2b package -- they assert the self-consistent
geometry contract of make_xdc_c5_2v plus the map_to_coords / map_to_coords_3d
rasterization order. The byte-for-byte parity with fw2b is in
test_parity_vs_fw2b.py (env-gated).
"""
import numpy as np
from fullwave2_ultra import geom

LAM = 1540.0 / 3.7e6
PTCH_M = 0.508e-3
# ppw ~ 6: sub-element divisor uses freq_div = lam*15/(ppw*ptch_m) with ppw=6.
FREQ_DIV = LAM * 15.0 / (6.0 * PTCH_M)
WX, WY = 7e-2, 8.4e-2


def _build():
    return geom.make_xdc_c5_2v(LAM, FREQ_DIV, WX, WY)


def test_grid_and_ppw():
    g = _build()
    assert g["nX"] > 0 and g["nY"] > 0
    assert abs(g["ppw"] - 6.0) < 1e-6
    assert g["dX"] > 0 and g["dY"] > 0


def test_coords_shape_and_order():
    g = _build()
    xdc = g["xdc"]
    incoords = xdc["incoords"]
    outcoords = xdc["outcoords"]
    assert incoords.ndim == 2 and incoords.shape[1] == 4
    assert outcoords.ndim == 2 and outcoords.shape[1] == 4
    assert incoords.shape[0] > 0 and outcoords.shape[0] > 0
    # lateral (row / first) column is stable-sorted nondecreasing
    assert np.all(np.diff(incoords[:, 0]) >= 0)
    assert np.all(np.diff(outcoords[:, 0]) >= 0)


def test_thetas_symmetric():
    g = _build()
    thetas = g["xdc"]["thetas"]
    assert thetas.size == g["xdc"]["npx"]
    # thetas are de-meaned at the element centers -> symmetric about 0
    assert abs(float(np.mean(thetas))) < 1e-12


def test_determinism():
    a = _build()
    b = _build()
    for k in ("incoords", "outcoords", "incoords2", "outcoords2", "surf",
              "thetas", "thetas_in", "thetas_out", "cen"):
        assert np.array_equal(a["xdc"][k], b["xdc"][k], equal_nan=True), k
    assert a["nX"] == b["nX"] and a["nY"] == b["nY"] and a["ppw"] == b["ppw"]


def test_map_to_coords_column_major():
    # 2D binary map; map_to_coords returns nonzero [i,j,val] in column-major
    # order: j (col) slowest, i (row) fastest.
    M = np.zeros((3, 2), dtype=np.float64)
    M[0, 0] = 1.0
    M[2, 0] = 1.0
    M[1, 1] = 1.0
    coords = geom.map_to_coords(M)
    assert coords.shape == (3, 3)
    # column-major scan: (i=0,j=0), (i=2,j=0), (i=1,j=1)
    expect_ij = np.array([[0, 0], [2, 0], [1, 1]], dtype=np.float64)
    assert np.array_equal(coords[:, 0:2], expect_ij)
    assert np.array_equal(coords[:, 2], np.ones(3))


def test_map_to_coords_3d_c_order():
    # 3D volume; map_to_coords_3d returns nonzero [i,j,k,val] in C-order
    # (i slowest, k fastest).
    V = np.zeros((2, 2, 2), dtype=np.float64)
    V[0, 0, 1] = 3.0
    V[0, 1, 0] = 4.0
    V[1, 0, 0] = 5.0
    coords = geom.map_to_coords_3d(V)
    assert coords.shape == (3, 4)
    expect = np.array([[0, 0, 1, 3.0],
                       [0, 1, 0, 4.0],
                       [1, 0, 0, 5.0]], dtype=np.float64)
    assert np.array_equal(coords, expect)
