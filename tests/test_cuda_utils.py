"""CPU-safe tests for the ctypes CUDA introspection (no GPU required).

Verifies import-safety and graceful degradation when no driver is present, the
semver->arch table, and (only when a real driver IS present, e.g. this dev box)
that device specs look sane.
"""
import fullwave2_ultra.cuda_utils as cu


def test_imports_without_gpu():
    # Module import must not require a GPU/driver, and have_cuda() returns a bool.
    assert isinstance(cu.have_cuda(), bool)


def test_no_driver_degrades_gracefully(monkeypatch):
    # Simulate "no driver": _lib() -> None. All functions must return empty/None,
    # never raise.
    cu._lib.cache_clear()
    monkeypatch.setattr(cu, "_lib", lambda: None)
    assert cu.have_cuda() is False
    assert cu.get_cuda_device_specs() == []
    assert cu.get_cuda_architecture() == []
    assert cu.driver_cuda_version() is None
    assert cu.device_sm(0) is None
    # monkeypatch restores the real lru_cached _lib at teardown.


def test_device_lookup_by_physical_index(monkeypatch):
    # Simulate device 0 having failed to query (skipped) so only physical index 1
    # survives -- lookups must key off the 'index' field, not list position.
    monkeypatch.setattr(cu, "get_cuda_architecture",
                        lambda: [{"index": 1, "sm_arch": "sm_86", "architecture": "ampere"}])
    assert cu.device_sm(1) == "sm_86"
    assert cu.device_architecture(1)["architecture"] == "ampere"
    assert cu.device_sm(0) is None          # NOT the positional [0] (which would be index 1)
    assert cu.device_architecture(0) is None


def test_semver_to_arch_table():
    assert cu.SEMVER_TO_ARCH[(8, 6)] == "ampere"      # RTX A6000 / 30-series
    assert cu.SEMVER_TO_ARCH[(9, 0)] == "hopper"       # H100
    assert cu.SEMVER_TO_ARCH[(7, 5)] == "turing"
    assert cu.SEMVER_TO_ARCH.get((99, 9), "unknown") == "unknown"


def test_live_specs_when_driver_present():
    # Only asserts shape when a real driver+GPU is here; skips on CPU-only hosts/CI.
    if not cu.have_cuda():
        import pytest
        pytest.skip("no CUDA driver on this host")
    specs = cu.get_cuda_device_specs()
    if not specs:
        import pytest
        pytest.skip("driver present but no devices")
    s0 = specs[0]
    assert s0["sm_arch"].startswith("sm_")
    assert isinstance(s0["compute_capability"], tuple) and len(s0["compute_capability"]) == 2
    assert cu.device_sm(0) == s0["sm_arch"]
    v = cu.driver_cuda_version()
    assert v is None or v > 0
