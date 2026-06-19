"""Round-trip tests for the 3D io_dat helpers (CPU, clean checkout, no GPU).

Verifies the on-disk 3D .dat contract:
  * write_map_3d / read_map_3d preserve a (nX,nY,nZ) float32 C-order array, and
    the on-disk layout is [i,j,k] at flat offset (i*nY+j)*nZ+k.
  * write_coords_3d writes [all i ; all j ; all k] and read_coords_3d returns
    (ncoords, 3) [i,j,k].
  * write_dcmap clamps a 3D zone map to [0, ndmap-1] (shape-agnostic).
"""
import numpy as np
from fullwave2_ultra import io_dat


def test_map_3d_round_trip_and_layout(tmp_path):
    nX, nY, nZ = 3, 4, 5
    rng = np.random.default_rng(0)
    V = rng.standard_normal((nX, nY, nZ)).astype(np.float32)
    p = str(tmp_path / "vol.dat")
    io_dat.write_map_3d(p, V)

    # round-trip is exact (float32 in, float32 out)
    back = io_dat.read_map_3d(p, nX, nY, nZ)
    assert back.shape == (nX, nY, nZ)
    assert np.array_equal(back, V)

    # on-disk layout: [i,j,k] lives at flat offset (i*nY+j)*nZ+k
    raw = np.fromfile(p, np.float32)
    assert raw.size == nX * nY * nZ
    for i in range(nX):
        for j in range(nY):
            for k in range(nZ):
                off = (i * nY + j) * nZ + k
                assert raw[off] == V[i, j, k]


def test_coords_3d_round_trip(tmp_path):
    i_idx = np.array([0, 5, 2, 9], dtype=np.int64)
    j_idx = np.array([1, 3, 8, 4], dtype=np.int64)
    k_idx = np.array([7, 0, 6, 2], dtype=np.int64)
    ncoords = i_idx.size
    p = str(tmp_path / "coords.dat")
    io_dat.write_coords_3d(p, i_idx, j_idx, k_idx)

    # raw on-disk order is [all i ; all j ; all k]
    raw = np.fromfile(p, np.int32)
    assert raw.size == 3 * ncoords
    assert np.array_equal(raw[:ncoords], i_idx.astype(np.int32))
    assert np.array_equal(raw[ncoords:2 * ncoords], j_idx.astype(np.int32))
    assert np.array_equal(raw[2 * ncoords:3 * ncoords], k_idx.astype(np.int32))

    # read_coords_3d returns (ncoords, 3) [i, j, k]
    coords = io_dat.read_coords_3d(p, ncoords)
    assert coords.shape == (ncoords, 3)
    assert np.array_equal(coords[:, 0], i_idx)
    assert np.array_equal(coords[:, 1], j_idx)
    assert np.array_equal(coords[:, 2], k_idx)


def test_dcmap_clamps_3d_zone_map(tmp_path):
    ndmap = 12
    nX, nY, nZ = 2, 3, 4
    rng = np.random.default_rng(1)
    # zone indices spanning well outside [0, ndmap-1] (negatives + overflow)
    dc = rng.integers(-8, ndmap + 8, size=(nX, nY, nZ)).astype(np.int64)
    dc.flat[0] = -5             # force an under-range value
    dc.flat[-1] = ndmap + 4     # force an over-range value
    p = str(tmp_path / "dcmap3d.dat")
    io_dat.write_dcmap(p, dc, ndmap)

    back = np.fromfile(p, np.int32)
    assert back.size == nX * nY * nZ
    assert back.min() >= 0
    assert back.max() <= ndmap - 1
    assert back[0] == 0                      # under-range clamps up to 0
    assert back[-1] == ndmap - 1             # over-range clamps down to ndmap-1
    # interior in-range values are preserved (compare against the C-order clamp)
    expect = np.clip(dc, 0, ndmap - 1).astype(np.int32).ravel()
    assert np.array_equal(back, expect)
