"""Byte-for-byte parity gate vs the ORIGINAL fw2b package (ENV-GATED).

Asserts that fullwave2_ultra's geom / medium reproduce the validated fw2b private
modules (fw2b._geom, fw2b._medium) EXACTLY on synthetic input. This is the
regression that guards the port from drifting.

Gating: runs only if FW2U_FW2B_DIR (or FW2B_DIR) points at an existing fw2b repo;
otherwise the whole module is skipped (a clean checkout, with no fw2b present,
skips green). The fw2b repo dir is inserted on sys.path so 'import fw2b' resolves
there. fw2b is READ-ONLY (import only). No path is baked in -- set the env var
(e.g. FW2U_FW2B_DIR=/path/to/fw2b pytest tests/test_parity_vs_fw2b.py).
"""
import os
import sys
import numpy as np
import pytest

FW2B_DIR = os.environ.get("FW2U_FW2B_DIR") or os.environ.get("FW2B_DIR") or ""

pytestmark = pytest.mark.skipif(
    not (FW2B_DIR and os.path.isdir(FW2B_DIR) and
         os.path.isfile(os.path.join(FW2B_DIR, "fw2b", "__init__.py"))),
    reason="set FW2U_FW2B_DIR to an existing fw2b repo for the parity gate")

if FW2B_DIR and os.path.isdir(FW2B_DIR) and FW2B_DIR not in sys.path:
    sys.path.insert(0, FW2B_DIR)

# Import fw2b lazily-guarded: at collection time the module path may not be set
# on a clean checkout, but the skipif above makes the body unreachable then.
try:
    from fw2b import _geom as fw2b_geom
    from fw2b import _medium as fw2b_medium
    _FW2B_OK = True
except Exception:  # pragma: no cover - only on a clean checkout w/o fw2b
    _FW2B_OK = False

from fullwave2_ultra import geom, medium

skip_no_fw2b = pytest.mark.skipif(not _FW2B_OK, reason="fw2b not importable")

# synthetic constants
LAM = 1540.0 / 3.7e6
PTCH_M = 0.508e-3
C0, F0 = 1540.0, 3.7e6
OMEGA0 = 2 * np.pi * F0


def _synthetic_c(nX=40, nY=48):
    """Lateral ramp + clip -> a smooth heterogeneous sound-speed map."""
    jj = np.linspace(-1.0, 1.0, nY)[None, :]
    c = 1540.0 + 30.0 * jj + np.zeros((nX, nY))
    return np.clip(c, 1480.0, 1600.0).astype(np.float64)


@skip_no_fw2b
@pytest.mark.parametrize("ppw", [4, 6, 8])
def test_make_xdc_c5_2v_full_struct(ppw):
    # IMPORTANT: freq_div uses lam = 1540/3.7e6 (NOT 1540).
    freq_div = LAM * 15.0 / (ppw * PTCH_M)
    wX, wY = 7e-2, 8.4e-2
    a = geom.make_xdc_c5_2v(LAM, freq_div, wX, wY)
    b = fw2b_geom.make_xdc_c5_2v(LAM, freq_div, wX, wY)
    assert a["nX"] == b["nX"] and a["nY"] == b["nY"]
    assert a["ppw"] == b["ppw"]
    ax, bx = a["xdc"], b["xdc"]
    for k in ("incoords", "outcoords", "incoords2", "outcoords2", "surf",
              "thetas", "thetas_in", "thetas_out", "cen"):
        assert np.array_equal(ax[k], bx[k], equal_nan=True), f"ppw={ppw} key={k}"


@skip_no_fw2b
def test_medium_extendmap_exact():
    M = _synthetic_c(12, 14)
    assert np.array_equal(medium.extendMap(M, 5), fw2b_medium.extendMap(M, 5))


@skip_no_fw2b
def test_medium_resample_map_exact():
    M = _synthetic_c(30, 36)
    out_a = medium.resample_map(M, 18, 22, 1.5)
    out_b = fw2b_medium.resample_map(M, 18, 22, 1.5)
    assert np.array_equal(out_a, out_b)


@skip_no_fw2b
def test_medium_resample_labels_exact():
    rng = np.random.default_rng(3)
    M = rng.integers(0, 5, size=(30, 36)).astype(np.float64)
    out_a = medium.resample_labels(M, 18, 22)
    out_b = fw2b_medium.resample_labels(M, 18, 22)
    assert np.array_equal(out_a, out_b)


@skip_no_fw2b
def test_medium_dbmhzcm2aexp_exact():
    dX = LAM / 6.0
    dT = dX / C0 * 0.45
    alpha = np.array([0.0, 0.3, 0.5, 1.0, 1.62, 2.0])
    assert np.array_equal(medium.dbmhzcm2aexp(alpha, C0, OMEGA0, dT),
                          fw2b_medium.dbmhzcm2aexp(alpha, C0, OMEGA0, dT))


@skip_no_fw2b
def test_medium_aexp_from_amap_exact():
    dX = LAM / 6.0
    dT = dX / C0 * 0.45
    rng = np.random.default_rng(4)
    amap_ext = (0.5 + 0.5 * rng.random((60, 70))).astype(np.float64)
    out_a = medium.aexp_from_amap(amap_ext, C0, OMEGA0, dT, nbdy=40)
    out_b = fw2b_medium.aexp_from_amap(amap_ext, C0, OMEGA0, dT, nbdy=40)
    assert np.array_equal(out_a, out_b)


@skip_no_fw2b
def test_medium_rescell2d_exact():
    args = (LAM, 0.02, 8.4e-2, 2.0, LAM / 6.0, LAM / 6.0)
    assert medium.rescell2d(*args) == fw2b_medium.rescell2d(*args)
