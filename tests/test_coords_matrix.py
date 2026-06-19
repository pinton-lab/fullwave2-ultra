"""Strided output-coordinate grids (the modX/modY/modZ spatial decimation):
coords_matrix / coords_matrix_3d are 0-based strided grids; volume_recorders is
the ndgrid-order variant.
"""
import numpy as np
from fullwave2_ultra import geom


def _ref_2d(nX, nY, modX, modY):
    """0-based strided grid loop (i-outer, j-inner == C-order)."""
    return np.array([[i, j] for i in range(0, nX, modX) for j in range(0, nY, modY)],
                    dtype=np.int64)


def _ref_3d(nX, nY, nZ, modX, modY, modZ):
    return np.array([[i, j, k] for i in range(0, nX, modX)
                     for j in range(0, nY, modY) for k in range(0, nZ, modZ)], dtype=np.int64)


def test_coords_matrix_2d_matches_loop():
    for nX, nY, mX, mY in [(7, 5, 1, 1), (10, 8, 2, 3), (9, 9, 4, 2), (6, 4, 3, 3)]:
        got = geom.coords_matrix(nX, nY, mX, mY)
        assert np.array_equal(got, _ref_2d(nX, nY, mX, mY)), (nX, nY, mX, mY)
        assert got.shape == (len(range(0, nX, mX)) * len(range(0, nY, mY)), 2)


def test_coords_matrix_3d_matches_loop():
    for nX, nY, nZ, mX, mY, mZ in [(5, 5, 5, 1, 1, 1), (8, 6, 4, 2, 3, 2), (9, 9, 9, 4, 4, 4)]:
        got = geom.coords_matrix_3d(nX, nY, nZ, mX, mY, mZ)
        assert np.array_equal(got, _ref_3d(nX, nY, nZ, mX, mY, mZ))


def test_full_grid_equals_map_to_coords_3d():
    # mod=1 -> every point, in the SAME C-order as map_to_coords_3d on a full volume
    nX, nY, nZ = 4, 3, 2
    c = geom.coords_matrix_3d(nX, nY, nZ)
    assert c.shape == (nX * nY * nZ, 3)
    full = geom.map_to_coords_3d(np.ones((nX, nY, nZ)))[:, :3].astype(np.int64)
    assert np.array_equal(c, full)


def test_coords_matrix_2d_is_3d_with_nz1():
    assert np.array_equal(geom.coords_matrix(7, 5, 2, 1),
                          geom.coords_matrix_3d(7, 5, 1, 2, 1, 1)[:, :2])


def test_volume_recorders_same_set_ndgrid_order():
    Nx, Ny, Nz, m = 8, 6, 4, 2
    vr = geom.volume_recorders((Nx, Ny, Nz), m, m, m)
    assert vr.shape[1] == 5 and np.all(vr[:, 3] == -1) and np.all(vr[:, 4] == 2)
    cm = geom.coords_matrix_3d(Nx, Ny, Nz, m, m, m)
    assert set(map(tuple, vr[:, :3].astype(int))) == set(map(tuple, cm))   # same index set
    assert vr[1, 0] - vr[0, 0] == m and vr[1, 1] == vr[0, 1]               # i-fastest (ndgrid)
    # scalar N == cubic grid
    assert np.array_equal(geom.volume_recorders(5, 2, 2, 2), geom.volume_recorders((5, 5, 5), 2, 2, 2))
