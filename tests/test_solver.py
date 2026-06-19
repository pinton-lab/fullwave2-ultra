"""CPU-safe tests for solver preflight + binary resolution (no GPU required).

The preflight DECISION logic (native / PTX-JIT / arch-uncovered / driver-too-old /
no-driver / cuobjdump-absent) is exercised with fakes; the resolver's search and
optional-download branches with tmp dirs + a monkeypatched urlretrieve.
"""
import os
import stat

import pytest

import fullwave2_ultra.cuda_utils as cu
import fullwave2_ultra.solver as sv


def test_sm_to_cc():
    assert sv._sm_to_cc("sm_86") == (8, 6)
    assert sv._sm_to_cc("sm_90") == (9, 0)
    assert sv._sm_to_cc("sm_100") == (10, 0)
    assert sv._sm_to_cc("sm_103") == (10, 3)
    assert sv._sm_to_cc("sm_120") == (12, 0)
    for bad in ("sm_", "sm_1", "sm_8x"):
        with pytest.raises(ValueError):
            sv._sm_to_cc(bad)


def test_embedded_sass_parsing(monkeypatch):
    fake = "Fatbin elf code:\narch = sm_70\narch = sm_86\narch = sm_90\n"
    monkeypatch.setattr(sv, "_cuobjdump", lambda *a, **k: fake)
    assert sv.embedded_sass_arches("x") == ["sm_70", "sm_86", "sm_90"]
    monkeypatch.setattr(sv, "_cuobjdump", lambda *a, **k: None)   # cuobjdump absent
    assert sv.embedded_sass_arches("x") is None


def _fake_gpu(monkeypatch, *, sm, driver=12.4, sass=("sm_60", "sm_86", "sm_90"),
             ptx=("sm_90",), arch="ampere", have=True):
    monkeypatch.setattr(cu, "have_cuda", lambda: have)
    monkeypatch.setattr(cu, "driver_cuda_version", lambda: driver)
    monkeypatch.setattr(cu, "device_architecture",
                        lambda i=0: {"index": i, "sm_arch": sm, "architecture": arch})
    monkeypatch.setattr(sv, "embedded_sass_arches", lambda b: (list(sass) if sass is not None else None))
    monkeypatch.setattr(sv, "embedded_ptx_arches", lambda b: (list(ptx) if ptx is not None else None))


@pytest.fixture
def fake_binary(tmp_path):
    p = tmp_path / "bench_2d_batch"
    p.write_bytes(b"\x7fELF fake")
    return p


def test_preflight_native(monkeypatch, fake_binary):
    _fake_gpu(monkeypatch, sm="sm_86")
    r = sv.preflight(fake_binary)
    assert r["ok"] and r["status"] == "native"
    # the report carries the actually-parsed SASS/PTX (not a hardcoded guess)
    assert r["sass"] == ["sm_60", "sm_86", "sm_90"] and r["ptx"] == ["sm_90"]


def test_preflight_ptx_jit_for_future_arch(monkeypatch, fake_binary):
    _fake_gpu(monkeypatch, sm="sm_120", sass=("sm_60", "sm_86", "sm_90"), arch="blackwell")
    r = sv.preflight(fake_binary)
    assert r["ok"] and r["status"] == "ptx_jit"
    assert any("JIT" in m for m in r["messages"])


def test_preflight_ptx_floor_keyed_off_binary(monkeypatch, fake_binary):
    # Binary embeds only compute_80 PTX -> JIT floor is sm_80 (NOT the hardcoded 90).
    # sm_86 (not in SASS) must JIT (8.6 >= 8.0); sm_75 must NOT (7.5 < 8.0). This
    # would be wrong if the decision used the constant PTX_FORWARD_CC=(9,0).
    _fake_gpu(monkeypatch, sm="sm_86", sass=("sm_60", "sm_90"), ptx=("sm_80",))
    r = sv.preflight(fake_binary)
    assert r["ok"] and r["status"] == "ptx_jit" and any("compute_80" in m for m in r["messages"])

    _fake_gpu(monkeypatch, sm="sm_75", sass=("sm_60", "sm_90"), ptx=("sm_80",), arch="turing")
    with pytest.raises(sv.PreflightError, match="compute_80"):
        sv.preflight(fake_binary)


def test_preflight_driver_minor_compat_warns(monkeypatch, fake_binary):
    # driver supports CUDA 12.1 (< build 12.3) -> ok via minor-version compat + a note.
    _fake_gpu(monkeypatch, sm="sm_86", driver=12.1)
    r = sv.preflight(fake_binary)
    assert r["ok"] and r["status"] == "native"
    assert any("minor-version compatibility" in m for m in r["messages"])


def test_preflight_driver_floor_does_not_raise(monkeypatch, fake_binary):
    # driver at the exact floor (CUDA 12.0) must NOT raise for a 12.3-built binary.
    _fake_gpu(monkeypatch, sm="sm_86", driver=12.0)
    assert sv.preflight(fake_binary)["ok"]


def test_preflight_arch_uncovered_raises(monkeypatch, fake_binary):
    _fake_gpu(monkeypatch, sm="sm_50", sass=("sm_60", "sm_86", "sm_90"), arch="maxwell")
    with pytest.raises(sv.PreflightError, match="not in the binary's SASS"):
        sv.preflight(fake_binary)


def test_preflight_driver_too_old_raises(monkeypatch, fake_binary):
    _fake_gpu(monkeypatch, sm="sm_86", driver=11.8)
    with pytest.raises(sv.PreflightError, match="needs a driver"):
        sv.preflight(fake_binary)


def test_preflight_no_driver(monkeypatch, fake_binary):
    _fake_gpu(monkeypatch, sm="sm_86", have=False)
    with pytest.raises(sv.PreflightError, match="no CUDA driver"):
        sv.preflight(fake_binary, require_gpu=True)
    # soft mode: no raise, status recorded
    r = sv.preflight(fake_binary, require_gpu=False)
    assert r["ok"] is False and r["status"] == "no_driver"


def test_preflight_cuobjdump_absent_assumes_default(monkeypatch, fake_binary):
    # sass=None simulates cuobjdump missing -> assume EXPECTED_SASS (incl. sm_86).
    _fake_gpu(monkeypatch, sm="sm_86", sass=None)
    r = sv.preflight(fake_binary)
    assert r["ok"] and r["status"] == "native"
    assert any("cuobjdump unavailable" in m for m in r["messages"])


def test_preflight_missing_binary_raises(tmp_path):
    with pytest.raises(sv.PreflightError, match="not found"):
        sv.preflight(tmp_path / "does_not_exist")


def test_resolve_binary_found_in_dir(tmp_path):
    (tmp_path / "bench_2d_batch").write_bytes(b"x")
    got = sv.resolve_binary("bench_2d_batch", extra_dirs=(str(tmp_path),))
    assert got == tmp_path / "bench_2d_batch"


def test_resolve_binary_missing_no_download(tmp_path, monkeypatch):
    monkeypatch.setattr(sv.config, "BIN_DIR", str(tmp_path / "empty"))
    monkeypatch.setattr(sv, "CACHE_BINS", tmp_path / "cache")
    with pytest.raises(FileNotFoundError, match="not found"):
        sv.resolve_binary("nope_bin", allow_download=False)


def test_resolve_binary_rejects_nonhttp_scheme(tmp_path, monkeypatch):
    monkeypatch.setattr(sv.config, "BIN_DIR", str(tmp_path / "empty"))
    monkeypatch.setattr(sv, "CACHE_BINS", tmp_path / "cache")
    with pytest.raises(ValueError, match="http"):
        sv.resolve_binary("x", tag="t", base_url="file:///etc", allow_download=True)


def test_resolve_binary_download_branch(tmp_path, monkeypatch):
    monkeypatch.setattr(sv.config, "BIN_DIR", str(tmp_path / "empty"))
    monkeypatch.setattr(sv, "CACHE_BINS", tmp_path / "cache")

    def fake_urlretrieve(url, dest):
        with open(dest, "wb") as f:
            f.write(b"downloaded-binary")

    monkeypatch.setattr(sv.urllib.request, "urlretrieve", fake_urlretrieve)
    got = sv.resolve_binary("bench_3d", tag="bin_v1", base_url="https://example/rel",
                            allow_download=True)
    assert got.exists() and got.read_bytes() == b"downloaded-binary"
    assert os.stat(got).st_mode & stat.S_IXUSR        # made executable
