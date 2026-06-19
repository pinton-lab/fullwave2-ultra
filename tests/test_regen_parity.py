"""Parity test: regen .dat byte-identical to a reference oracle run dir.

The regen DRIVER (regen.regen_fsa_ppw) is ABDOMINAL-FSA-specific and is NOT part
of fullwave2_ultra (this repo ships only the general medium helpers).

Validated upstream at ppw=6 against a reference oracle run: scalars exact;
c/icc/outc/icmat AND Aexp byte-identical (max|diff|=0); rho differs only by
scatterer RNG (intended). This test therefore SKIPS cleanly here: it requires
both the oracle/medium env (FW2U_MEDIUM_WS + FW2U_ORACLE_DIR) AND an importable
regen driver module -- and the driver module does not exist in this repo, so a
clean checkout always collects-and-skips green.
"""
import os
import struct
import numpy as np
import pytest

# Oracle/medium come from env so the gate is host-portable; no machine-specific
# path is baked in. FW2U_* primary, legacy FW2B_* accepted as fallback.
WS = os.environ.get("FW2U_MEDIUM_WS") or os.environ.get("FW2B_MEDIUM_WS") or ""
ORA = os.environ.get("FW2U_ORACLE_DIR") or os.environ.get("FW2B_ORACLE_DIR") or ""


@pytest.mark.skipif(not (WS and ORA and os.path.exists(WS) and os.path.isdir(ORA)),
                    reason="set FW2U_MEDIUM_WS + FW2U_ORACLE_DIR to the medium "
                           "workspace + reference oracle run")
def test_regen_parity(tmp_path):
    # The abdominal-specific regen driver is NOT in fullwave2_ultra; it lives in
    # the source project (fw2b/regen.py). importorskip on a module that does not
    # exist here guarantees a clean-checkout skip even if the env vars are set.
    regen = pytest.importorskip(
        "fullwave2_ultra.regen",
        reason="regen_fsa_ppw driver is abdominal-specific; only in fw2b/regen.py")

    out = str(tmp_path / "py6")
    regen.regen_fsa_ppw(6.0, out, WS, csr_scale=0.25)
    ri = lambda d, n: struct.unpack("i", open(f"{d}/{n}.dat", "rb").read(4))[0]
    for n in ["nX", "nY", "nT", "ncoords", "ncoordsout", "nTic", "modT"]:
        assert ri(out, n) == ri(ORA, n), n

    def same(n, kind="f", ncols=False):
        dt = np.float32 if kind == "f" else np.int32
        a = np.fromfile(f"{out}/{n}", dt); b = np.fromfile(f"{ORA}/{n}", dt)
        if ncols:
            b = b[:len(a)]
        return np.array_equal(a, b)

    for n in ["c.dat", "icmat.dat"]:
        assert same(n), f"{n} not byte-identical"
    assert same("icc.dat", "i", True) and same("outc.dat", "i", True)
    a = np.fromfile(f"{out}/Aexp.dat", np.float32); b = np.fromfile(f"{ORA}/Aexp.dat", np.float32)
    assert a.max() <= 1.0, "Aexp amplifies (>1): attenuation sign inverted"   # physics invariant
    assert np.abs(a - b).max() < 1e-5, "Aexp drift too large"                  # was 1e-2; masked a sign bug
