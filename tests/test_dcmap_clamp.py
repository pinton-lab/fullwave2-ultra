"""Guard for the io_dat.write_dcmap index clamp.

A zone index can reach ndmap (one past the valid range [0, ndmap-1]) -> an
out-of-range read in the solver. io_dat.write_dcmap must clamp to [0, ndmap-1].
"""
import numpy as np
from fullwave2_ultra import io_dat


def test_dcmap_clamped_to_range(tmp_path):
    ndmap = 30
    dc = np.array([-5, 0, 7, ndmap - 1, ndmap, ndmap + 4, 100], dtype=np.int64)
    p = str(tmp_path / "dcmap.dat")
    io_dat.write_dcmap(p, dc, ndmap)
    back = np.fromfile(p, np.int32)
    assert back.min() >= 0
    assert back.max() <= ndmap - 1
    assert back.max() == ndmap - 1            # the out-of-range values clamp down to ndmap-1
    assert back.min() == 0                     # the negative clamps up to 0
