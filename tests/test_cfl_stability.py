"""Gates for the published CFL limits and the reference-speed check.

Two things are pinned. That the shipped constants stay put -- they are measured
numbers, not conventions, and a silent edit would mis-size every deck. And that
the deck writers check the REALIZED Courant ratio in the fastest material rather
than the nominal `cfl` they were handed, which is the trap that NaNs a run in
bone with no warning from the requested number.
"""
import math
import warnings

import numpy as np
import pytest

from fullwave2_ultra import sim
from fullwave2_ultra.stability import (CFL_LIMIT, cfl_limit, check_cfl,
                                       realized_cfl)

C0, OMEGA0, PPW = 1540.0, 2 * np.pi * 1e6, 6.0


def test_published_limits_are_below_the_folklore_value():
    """The optimized taps buy phase accuracy with zone-corner amplitude, which
    is what the stability bound integrates -- so they are LESS stable than a
    plain second-order leapfrog, not more."""
    for (M, dim), lim in CFL_LIMIT.items():
        assert 0.0 < lim < 1.0 / math.sqrt(dim), (M, dim, lim)


def test_constants_are_pinned():
    assert cfl_limit(M=8, dim=2) == pytest.approx(0.5326, abs=1e-4)
    assert cfl_limit(M=6, dim=2) == pytest.approx(0.6108, abs=1e-4)


def test_no_3d_limit_is_published_and_saying_so_is_explicit():
    with pytest.raises(KeyError, match="not part of the public package"):
        cfl_limit(M=8, dim=3)


def test_realized_ratio_uses_the_fastest_material():
    dX, dT = 1e-4, 0.30 * 1e-4 / 1540.0        # a caller asking for cfl = 0.30
    assert realized_cfl(1540.0, dT, dX) == pytest.approx(0.30, rel=1e-9)
    assert realized_cfl(2900.0, dT, dX) == pytest.approx(0.5649, abs=1e-4)


def test_2d_over_the_limit_raises():
    dX, dT = 1e-4, 0.45 * 1e-4 / 1540.0
    assert check_cfl(1540.0, dT, dX, dim=2)["ok"]
    with pytest.raises(ValueError, match="unstable time step"):
        check_cfl(2900.0, dT, dX, dim=2)


def test_reference_speed_warning_fires_without_a_published_limit():
    """The 3D path has no threshold, but the c0-vs-max(c) trap is dimension
    independent and must still be reported."""
    dX, dT = 1e-4, 0.30 * 1e-4 / 1540.0
    with pytest.warns(UserWarning, match="no CFL limit is published"):
        rep = check_cfl(2900.0, dT, dX, c0=1540.0, dim=3)
    assert rep["limit"] is None and rep["r"] == pytest.approx(0.5649, abs=1e-4)


def _maps3d(cmax, n=24):
    c = np.full((n, n, n), 1540.0)
    c[n // 2:] = cmax
    return dict(cmap=c, rmap=np.full((n, n, n), 1000.0),
                nmap=np.zeros((n, n, n)), amap=np.zeros((n, n, n)))


def test_3d_writer_warns_on_the_reference_speed_trap(tmp_path):
    with pytest.warns(UserWarning, match="You asked for cfl=0.3000"):
        sim.write_fullwave_sim_3d(str(tmp_path / "a"), C0, OMEGA0, 1e-6, PPW,
                                  0.30, _maps3d(2900.0), np.array([[4, 4, 4]]),
                                  np.array([[8, 8, 8]]), 10, 1, nbdy=4)


def test_2d_writer_refuses_a_deck_that_would_diverge(tmp_path):
    n = 24
    c = np.full((n, n), 1540.0); c[n // 2:] = 2900.0
    maps = dict(cmap=c, rmap=np.full((n, n), 1000.0),
                nmap=np.zeros((n, n)), amap=np.zeros((n, n)))
    xdc = dict(incoords=np.array([[5, 5]]), outcoords=np.array([[10, 10]]))
    with pytest.raises(ValueError, match="unstable time step"):
        sim.write_fullwave_sim(str(tmp_path / "b"), C0, OMEGA0, 1e-6, PPW,
                               0.45, maps, xdc, 10, 1, nbdy=4)


def test_homogeneous_deck_is_unaffected(tmp_path):
    """The guard must not fire on the ordinary case."""
    n = 24
    maps = dict(cmap=np.full((n, n), C0), rmap=np.full((n, n), 1000.0),
                nmap=np.zeros((n, n)), amap=np.zeros((n, n)))
    xdc = dict(incoords=np.array([[5, 5]]), outcoords=np.array([[10, 10]]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # any warning fails the test
        sim.write_fullwave_sim(str(tmp_path / "c"), C0, OMEGA0, 1e-6, PPW,
                               0.40, maps, xdc, 10, 1, nbdy=4)


def test_ordinary_tissue_heterogeneity_stays_silent(tmp_path):
    """A few percent of scatterer contrast above c0 must NOT warn -- a guard that
    fires on every realistic deck trains people to ignore it."""
    n = 24
    c = np.full((n, n), C0); c[10:14, 10:14] = 1576.0     # ~2.4% over c0
    maps = dict(cmap=c, rmap=np.full((n, n), 1000.0),
                nmap=np.zeros((n, n)), amap=np.zeros((n, n)))
    xdc = dict(incoords=np.array([[5, 5]]), outcoords=np.array([[10, 10]]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sim.write_fullwave_sim(str(tmp_path / "t"), C0, OMEGA0, 1e-6, PPW,
                               0.40, maps, xdc, 10, 1, nbdy=4)
