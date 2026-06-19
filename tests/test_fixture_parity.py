"""Core .dat-level parity gate (CPU, clean checkout, NO GPU / NO dataset).

Regenerates the committed examples/tiny_nsims4 fixture into tmp_path and asserts
that EVERY *.dat it produces is byte-identical to the committed golden fixture.
This proves sim.write_fullwave_sim + medium + io_dat reproduce the golden inputs
bit-for-bit on a clean checkout.

Approach: the fixture script writes everything to its module-level ``HERE``
(the script's own directory). We read the script text, replace the single
``HERE = ...`` assignment line with ``HERE = <tmp_path>``, exec the patched
source in a fresh namespace, and call ``main()``. This is robust to the script
deriving other constants from HERE because the substitution happens before any
of main() runs, and it avoids importing/polluting the real fixture directory.
"""
import os
import re
import filecmp
import glob
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE_DIR = os.path.join(REPO, "examples", "tiny_nsims4")
MAKE_FIXTURE = os.path.join(FIXTURE_DIR, "make_fixture.py")


@pytest.mark.skipif(not os.path.isfile(MAKE_FIXTURE),
                    reason="examples/tiny_nsims4/make_fixture.py not present")
def test_tiny_nsims4_fixture_byte_identical(tmp_path):
    with open(MAKE_FIXTURE, "r") as f:
        src = f.read()

    # Redirect the module-level HERE to tmp_path (matches the top-level
    # 'HERE = os.path.dirname(os.path.abspath(__file__))' assignment line).
    patched, n = re.subn(r"(?m)^HERE\s*=.*$",
                         "HERE = " + repr(str(tmp_path)),
                         src, count=1)
    assert n == 1, "could not locate the HERE assignment in make_fixture.py"

    ns = {"__name__": "make_fixture_under_test", "__file__": MAKE_FIXTURE}
    code = compile(patched, MAKE_FIXTURE, "exec")
    exec(code, ns)
    assert "main" in ns, "make_fixture.py has no main()"
    ns["main"]()

    golden = sorted(os.path.basename(p)
                    for p in glob.glob(os.path.join(FIXTURE_DIR, "*.dat")))
    produced = sorted(os.path.basename(p)
                      for p in glob.glob(os.path.join(str(tmp_path), "*.dat")))

    assert golden, "no committed *.dat golden files found"
    # the regenerated set must match the committed set exactly
    assert set(produced) == set(golden), (
        f"dat file sets differ: only-produced={set(produced) - set(golden)} "
        f"only-golden={set(golden) - set(produced)}")

    mismatched = []
    for name in golden:
        a = os.path.join(FIXTURE_DIR, name)
        b = os.path.join(str(tmp_path), name)
        if not filecmp.cmp(a, b, shallow=False):
            mismatched.append(name)
    assert not mismatched, f"not byte-identical to golden: {mismatched}"
