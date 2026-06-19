"""Guard for adaptive modT / RF Nyquist (gotcha #2).

A fixed modT under-samples the 3.7 MHz RF at low ppw. The adaptive rule
modT = max(1, round((1/dT)/14.4e6)) must keep the genout sample rate
fs = (1/dT)/modT above Nyquist (fs/2 > f0) at every ppw.
"""
import numpy as np

C0, F0, CFL = 1540.0, 3.7e6, 0.45


def _mround(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def test_fs_above_nyquist_across_ppw():
    for ppw in [3, 4, 6, 8, 12, 16]:
        dX = (C0 / F0) / ppw
        dT = dX / C0 * CFL
        modT = max(1, int(_mround((1.0 / dT) / 14.4e6)))
        fs = (1.0 / dT) / modT
        assert fs / 2.0 > F0, f"ppw={ppw}: fs/2={fs/2:.3e} <= f0={F0:.3e} (modT={modT})"
