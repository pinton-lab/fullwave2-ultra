"""Gates for sim.write_addsrc_rundir -- the supported writer for the ADDITIVE
source channel (icc_add/icmat_add, docs/io_contract.md).

The one channel with layout subtleties (sim-major blocks in 2D, TIME-major in
3D, its own nTic_add, and a uniqueness requirement the solver's atomicAdd does
not enforce) gets a writer and validation. These tests pin the round trip on
disk and every refusal.
"""
import os

import numpy as np
import pytest

from fullwave2_ultra import io_dat, sim

C0, OMEGA0, PPW, CFL = 1540.0, 2 * np.pi * 1e6, 6.0, 0.40


def _rundir(tmp_path, nsims=1, nX=24, nY=24):
    out = str(tmp_path / "run")
    maps = dict(cmap=np.full((nX, nY), C0), rmap=np.full((nX, nY), 1000.0),
                nmap=np.zeros((nX, nY)), amap=np.zeros((nX, nY)))
    xdc = dict(incoords=np.array([[5, 5], [5, 6]]),
               outcoords=np.array([[10, 10]]))
    meta = sim.write_fullwave_sim(out, C0, OMEGA0, 2e-6, PPW, CFL, maps, xdc,
                                  nTic=20, modT=1, nbdy=4, M=8)
    io_dat.write_icmat(os.path.join(out, "icmat.dat"),
                       [np.zeros((meta["ncoords"], 20)) for _ in range(nsims)])
    if nsims > 1:
        io_dat.write_int(os.path.join(out, "nsims.dat"), nsims)
    return out, meta


def _interior(meta, n, mext=12):
    """n distinct coords comfortably inside the active region."""
    return np.stack([np.arange(n) + mext + 2, np.full(n, mext + 3)], axis=1)


def test_round_trip_single_sim(tmp_path):
    run, meta = _rundir(tmp_path)
    ia = _interior(meta, 3)
    tr = np.arange(3 * 12, dtype=np.float64).reshape(3, 12) * 0.01
    info = sim.write_addsrc_rundir(run, ia, tr)
    assert info == dict(ncoords_add=3, nTic_add=12, nsims=1)

    assert io_dat.read_int(os.path.join(run, "ncoords_add.dat")) == 3
    assert io_dat.read_int(os.path.join(run, "nTic_add.dat")) == 12
    v = np.fromfile(os.path.join(run, "icc_add.dat"), np.int32)
    np.testing.assert_array_equal(np.stack([v[:3], v[3:]], 1), ia)  # [all i; all j]
    raw = np.fromfile(os.path.join(run, "icmat_add.dat"), np.float32)
    np.testing.assert_allclose(raw.reshape(3, 12), tr.astype(np.float32), rtol=0, atol=0)


def test_round_trip_batched_is_sim_major(tmp_path):
    run, meta = _rundir(tmp_path, nsims=3)
    ia = _interior(meta, 2)
    blocks = [np.full((2, 8), float(s + 1)) for s in range(3)]
    sim.write_addsrc_rundir(run, ia, blocks)
    raw = np.fromfile(os.path.join(run, "icmat_add.dat"), np.float32)
    raw = raw.reshape(3, 2, 8)
    for s in range(3):                       # sim-major, not coord-major
        assert np.all(raw[s] == s + 1)


def test_nTic_add_truncates_not_pads(tmp_path):
    run, meta = _rundir(tmp_path)
    ia = _interior(meta, 2)
    tr = np.tile(np.arange(10.0), (2, 1))
    sim.write_addsrc_rundir(run, ia, tr, nTic_add=6)
    assert io_dat.read_int(os.path.join(run, "nTic_add.dat")) == 6
    raw = np.fromfile(os.path.join(run, "icmat_add.dat"), np.float32)
    assert raw.size == 2 * 6                 # 2 coords x 6 samples, nothing padded


@pytest.mark.parametrize("bad,msg", [
    ("dup", "duplicate"),
    ("oob", "outside the extended grid"),
    ("rows", "rows but"),
    ("nsims", "trace block"),
    ("nTic", "exceeds the trace length"),
    ("nonfinite", "non-finite"),
])
def test_refusals(tmp_path, bad, msg):
    run, meta = _rundir(tmp_path, nsims=2 if bad == "nsims" else 1)
    ia = _interior(meta, 3)
    tr = np.zeros((3, 8))
    kw = {}
    if bad == "dup":
        ia = np.concatenate([ia[:2], ia[:1]])
    elif bad == "oob":
        ia = ia.copy(); ia[0, 0] = 100000
    elif bad == "rows":
        tr = np.zeros((2, 8))
    elif bad == "nTic":
        kw["nTic_add"] = 99
    elif bad == "nonfinite":
        tr = tr.copy(); tr[0, 0] = np.nan
    with pytest.raises(ValueError, match=msg):
        sim.write_addsrc_rundir(run, ia, tr, **kw)


def test_frozen_shell_coords_warn(tmp_path):
    run, meta = _rundir(tmp_path)
    ia = np.array([[1, 1], [2, 2]])          # inside the never-updated M+1 shell
    with pytest.warns(UserWarning, match="frozen"):
        sim.write_addsrc_rundir(run, ia, np.zeros((2, 8)))


# ---------------------------------------------------------------- 3D ------
# The 3D channel shipped 2026-08-28. Its one real hazard is that icmat_add is
# TIME-MAJOR in 3D and coord-major in 2D, and a transposed file has exactly the
# right SIZE -- so a hand-rolled tofile produces a deck the solver reads happily
# and injects with the axes swapped (io_contract.md, 'Additive sources'). These pin the orientation.

def _rundir3d(tmp_path, n=8, nTic=20, incoords=None):
    out = str(tmp_path / "run3")
    maps = dict(cmap=np.full((n, n, n), C0), rmap=np.full((n, n, n), 1000.0),
                nmap=np.zeros((n, n, n)), amap=np.zeros((n, n, n)))
    inc = np.array([[3, 3, 3], [3, 4, 3]]) if incoords is None else incoords
    meta = sim.write_fullwave_sim_3d(out, C0, OMEGA0, 2e-6, PPW, CFL, maps,
                                     inc, np.array([[5, 5, 5]]),
                                     nTic=nTic, modT=1, nbdy=4, M=8,
                                     cfl_check="ignore")
    return out, meta


def _interior3d(meta, n):
    m = meta["mext"]
    return np.stack([np.arange(n) + m + 2, np.full(n, m + 3), np.full(n, m + 3)], axis=1)


def test_3d_round_trip_is_time_major(tmp_path):
    """(N, nTic) in, time-major on disk, and the same array back out."""
    run, meta = _rundir3d(tmp_path)
    ia = _interior3d(meta, 4)
    tr = np.arange(4 * 6, dtype=np.float64).reshape(4, 6)
    info = sim.write_addsrc_rundir(run, ia, tr)
    assert info == dict(ncoords_add=4, nTic_add=6, nsims=1)

    p = os.path.join(run, "icmat_add.dat")
    raw = np.fromfile(p, dtype=np.float32)
    assert raw.size == 4 * 6
    # slice n is contiguous: the n-th run of ncoords_add floats is column n
    np.testing.assert_array_equal(raw.reshape(6, 4), tr.T.astype(np.float32))
    np.testing.assert_array_equal(io_dat.read_icmat_time_major(p, 4, 6),
                                  tr.astype(np.float32))
    v3 = np.fromfile(os.path.join(run, "icc_add.dat"), np.int32)
    np.testing.assert_array_equal(np.stack([v3[:4], v3[4:8], v3[8:]], 1), ia)


def test_3d_layout_differs_from_2d(tmp_path):
    """The SAME traces produce different bytes in 2D and 3D. This is the trap:
    both files are nTic*ncoords floats, so a size check cannot catch a swap."""
    tr = np.arange(3 * 5, dtype=np.float64).reshape(3, 5)
    r2, m2 = _rundir(tmp_path)
    sim.write_addsrc_rundir(r2, _interior(m2, 3), tr)
    r3, m3 = _rundir3d(tmp_path)
    sim.write_addsrc_rundir(r3, _interior3d(m3, 3), tr)
    b2 = open(os.path.join(r2, "icmat_add.dat"), "rb").read()
    b3 = open(os.path.join(r3, "icmat_add.dat"), "rb").read()
    assert len(b2) == len(b3) and b2 != b3


def test_3d_needs_three_columns(tmp_path):
    run, meta = _rundir3d(tmp_path)
    with pytest.raises(ValueError, match=r"\(N, >=3\)"):
        sim.write_addsrc_rundir(run, np.array([[12, 12], [13, 12]]),
                                np.zeros((2, 4)))


def test_3d_refuses_a_batch(tmp_path):
    """The 3D solvers have no sim axis; two blocks must not be written."""
    run, meta = _rundir3d(tmp_path)
    ia = _interior3d(meta, 2)
    with pytest.raises(ValueError, match="no batch axis"):
        sim.write_addsrc_rundir(run, ia, np.zeros((2, 2, 4)))


def test_3d_bounds_and_frozen(tmp_path):
    run, meta = _rundir3d(tmp_path)
    nZ = io_dat.read_int(os.path.join(run, "nZ.dat"))
    with pytest.raises(ValueError, match="outside the extended grid"):
        sim.write_addsrc_rundir(run, np.array([[5, 5, nZ + 3]]), np.zeros((1, 4)))
    with pytest.warns(UserWarning, match="frozen"):
        sim.write_addsrc_rundir(run, np.array([[1, 1, 1]]), np.zeros((1, 4)))


def test_3d_ncoords_zero_deck(tmp_path):
    """ncoords=0 is a valid 3D deck: the additive channel carries the source and
    icc/icmat are omitted (a dummy hard cell would be a rigid point scatterer)."""
    run, meta = _rundir3d(tmp_path, incoords=np.zeros((0, 3), dtype=int))
    assert meta["ncoords"] == 0
    assert io_dat.read_int(os.path.join(run, "ncoords.dat")) == 0
    assert not os.path.exists(os.path.join(run, "icc.dat"))
    info = sim.write_addsrc_rundir(run, _interior3d(meta, 3), np.zeros((3, 4)))
    assert info["ncoords_add"] == 3
