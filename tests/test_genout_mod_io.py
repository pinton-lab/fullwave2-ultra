"""CPU test for the genout_mod reader (reshape / decimated-dims), no GPU."""
import numpy as np
from fullwave2_ultra import io_dat


def test_genout_mod_dims():
    assert io_dat.genout_mod_dims(10, 10, 10) == (10, 10, 10)
    assert io_dat.genout_mod_dims(9, 8, 7, 2, 3, 4) == (5, 3, 2)   # ceil(N/mod)


def test_read_genout_mod_reshape(tmp_path):
    nframes, nX, nY, nZ, mod = 3, 7, 5, 4, 2
    nX2, nY2, nZ2 = io_dat.genout_mod_dims(nX, nY, nZ, mod, mod, mod)
    assert (nX2, nY2, nZ2) == (4, 3, 2)
    a = np.arange(nframes * nX2 * nY2 * nZ2, dtype=np.float32)
    p = tmp_path / "genout_mod.dat"
    a.tofile(p)
    g = io_dat.read_genout_mod(str(p), nX, nY, nZ, mod, mod, mod)
    assert g.shape == (nframes, nX2, nY2, nZ2)
    assert np.array_equal(g.ravel(), a)            # frame-major, C-order


def test_genout_mod_dims_2d():
    assert io_dat.genout_mod_dims_2d(10, 10) == (10, 10)
    assert io_dat.genout_mod_dims_2d(9, 8, 2, 3) == (5, 3)   # ceil(N/mod)


def test_genout_mod_name():
    assert io_dat.genout_mod_name(0) == "genout_mod.dat"
    assert io_dat.genout_mod_name(3) == "genout_mod_s3.dat"


def test_read_genout_mod_2d_reshape(tmp_path):
    nframes, nX, nY, mod = 4, 7, 5, 2
    nX2, nY2 = io_dat.genout_mod_dims_2d(nX, nY, mod, mod)
    assert (nX2, nY2) == (4, 3)
    a = np.arange(nframes * nX2 * nY2, dtype=np.float32)
    p = tmp_path / "genout_mod.dat"
    a.tofile(p)
    g = io_dat.read_genout_mod_2d(str(p), nX, nY, mod, mod)
    assert g.shape == (nframes, nX2, nY2)
    assert np.array_equal(g.ravel(), a)            # frame-major, C-order


def test_read_genout_mod_batch_2d(tmp_path):
    nsims, nframes, nX, nY = 3, 2, 6, 4
    for s in range(nsims):
        a = np.full(nframes * nX * nY, s, dtype=np.float32)
        a.tofile(tmp_path / io_dat.genout_mod_name(s))
    g = io_dat.read_genout_mod_batch_2d(str(tmp_path), nsims, nX, nY)
    assert g.shape == (nsims, nframes, nX, nY)
    assert all((g[s] == s).all() for s in range(nsims))   # each sim's file kept distinct
