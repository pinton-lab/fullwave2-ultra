"""Regression for the Aexp attenuation-sign bug (no dataset needed).

dbmhzcm2aexp must ATTENUATE: aexp <= 1 for alpha >= 0 (exactly 1 at alpha=0),
strictly < 1 and monotonically decreasing for alpha > 0. A double-negation once
flipped the sign so the kernel amplified the field with depth. Golden values are
the np=-db(exp(-1)) result at the ppw=6 fine dT.
"""
import numpy as np
from fullwave2_ultra.medium import dbmhzcm2aexp

C0, F0 = 1540.0, 3.7e6
OMEGA0 = 2 * np.pi * F0
DT = (C0 / F0 / 6.0) / C0 * 0.45            # ppw=6 fine solver dT = 2.027e-8 s


def test_aexp_attenuates():
    alpha = np.array([0.0, 0.5, 1.0, 1.62])
    a = dbmhzcm2aexp(alpha, C0, OMEGA0, DT)
    assert a[0] == 1.0                                   # no attenuation at alpha=0
    assert np.all(a <= 1.0), f"Aexp amplifies (>1): {a}"  # the sign invariant
    assert np.all(np.diff(a) < 0)                        # decreasing with alpha


def test_aexp_matches_golden():
    # dbmhzcm2aexp (np=-db(exp(-1))=+8.686) at DT above
    golden = {0.5: 0.99934, 1.0: 0.99867, 1.62: 0.99785}
    for al, exp in golden.items():
        got = float(dbmhzcm2aexp(np.array([al]), C0, OMEGA0, DT)[0])
        assert abs(got - exp) < 5e-4, f"alpha={al}: {got} vs golden {exp}"
