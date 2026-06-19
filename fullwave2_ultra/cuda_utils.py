# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gianmarco Pinton
"""Dependency-free CUDA device / driver introspection via ``ctypes`` on libcuda.

No cupy, no CUDA toolkit, no nvidia-smi -- this dlopen's the user-mode driver
(``libcuda.so.1``) and calls the CUDA Driver API directly, so it works wherever a
driver is installed and is safe to import on a CPU-only / clean checkout (the
library is loaded lazily; every function degrades gracefully to ``None``/``[]``
and NEVER raises when no driver / a broken device is present).

Used by :mod:`fullwave2_ultra.solver` to preflight a run: detect the GPU's
compute capability (``sm_XX``) and the driver's max CUDA version, so a launch can
fail with a clear "this GPU/driver isn't covered by the binary" message instead
of a cryptic CUDA error.

Adapted from fullwave25 (github.com/pinton-lab/fullwave25, ``fullwave/solver/cuda_utils.py``),
itself derived from a public gist by f0k
(gist.github.com/f0k/63a664160d016a491b2cbea15913d549 ; specifically the
permalink_comment_id=5043495 update). See NOTICE.
"""
from __future__ import annotations

import ctypes
import logging
from ctypes import POINTER, c_char, c_char_p, c_int, c_size_t, c_void_p
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16

# compute capability (major, minor) -> marketing architecture name.
# Updated from the "GPUs supported" section of en.wikipedia.org/wiki/CUDA.
SEMVER_TO_ARCH: dict[tuple[int, int], str] = {
    (3, 0): "kepler", (3, 2): "kepler", (3, 5): "kepler", (3, 7): "kepler",
    (5, 0): "maxwell", (5, 2): "maxwell", (5, 3): "maxwell",
    (6, 0): "pascal", (6, 1): "pascal", (6, 2): "pascal",
    (7, 0): "volta", (7, 2): "volta",
    (7, 5): "turing",
    (8, 0): "ampere", (8, 6): "ampere", (8, 7): "ampere",
    (8, 9): "ada lovelace",
    (9, 0): "hopper",
    (10, 0): "blackwell", (10, 1): "blackwell", (12, 0): "blackwell",
}

_LIBNAMES = ("libcuda.so", "libcuda.so.1", "libcuda.dylib", "nvcuda.dll")

# argtypes/restype for the driver calls we use, so pointer/size args marshal at
# the right width on LP64/LLP64 instead of relying on ctypes' int default. Keyed
# by symbol; applied to whichever of the symbol / its _v2 form the driver exports.
_SIGNATURES = {
    "cuInit": ([c_int], c_int),
    "cuDriverGetVersion": ([POINTER(c_int)], c_int),
    "cuDeviceGetCount": ([POINTER(c_int)], c_int),
    "cuDeviceGet": ([POINTER(c_int), c_int], c_int),
    "cuDeviceGetName": ([c_char_p, c_int, c_int], c_int),
    "cuDeviceComputeCapability": ([POINTER(c_int), POINTER(c_int), c_int], c_int),
    "cuDeviceGetAttribute": ([POINTER(c_int), c_int, c_int], c_int),
    "cuGetErrorString": ([c_int, POINTER(c_char_p)], c_int),
    "cuCtxCreate": ([POINTER(c_void_p), c_int, c_int], c_int),
    "cuCtxCreate_v2": ([POINTER(c_void_p), c_int, c_int], c_int),
    "cuMemGetInfo": ([POINTER(c_size_t), POINTER(c_size_t)], c_int),
    "cuMemGetInfo_v2": ([POINTER(c_size_t), POINTER(c_size_t)], c_int),
    "cuCtxDestroy": ([c_void_p], c_int),
    "cuCtxDestroy_v2": ([c_void_p], c_int),
}


@lru_cache(maxsize=1)
def _lib() -> ctypes.CDLL | None:
    """Lazily dlopen the CUDA driver library; None if no driver is installed."""
    for name in _LIBNAMES:
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        for sym, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(lib, sym, None)
            if fn is not None:
                fn.argtypes, fn.restype = argtypes, restype
        return lib
    logger.debug("no CUDA driver library found (tried %s)", ", ".join(_LIBNAMES))
    return None


def have_cuda() -> bool:
    """True if the CUDA driver library can be loaded (a GPU driver is present)."""
    return _lib() is not None


def _resolve(lib, *names):
    """First available symbol among *names* (handles the _v2 vs base API split)."""
    for n in names:
        fn = getattr(lib, n, None)
        if fn is not None:
            return fn
    return None


def _check(lib, code: int, fn: str) -> None:
    if code != CUDA_SUCCESS:
        msg = ctypes.c_char_p()
        try:
            lib.cuGetErrorString(code, ctypes.byref(msg))
            detail = msg.value.decode() if msg.value else "unknown error"
        except Exception:
            detail = "unknown error"
        raise RuntimeError(f"{fn} failed (code {code}): {detail}")


def driver_cuda_version() -> float | None:
    """Max CUDA version supported by the installed driver (e.g. 12.3), or None.

    This is the driver's capability via ``cuDriverGetVersion`` -- the ceiling on
    the CUDA toolkit version a binary may have been built with to run here.
    """
    lib = _lib()
    if lib is None:
        return None
    try:
        ver = ctypes.c_int()
        if lib.cuDriverGetVersion(ctypes.byref(ver)) != CUDA_SUCCESS:
            return None
        return float(f"{ver.value // 1000}.{(ver.value % 1000) // 10}")
    except (AttributeError, OSError, ctypes.ArgumentError) as e:  # pragma: no cover
        logger.debug("cuDriverGetVersion failed: %s", e)
        return None


# Any driver/ctypes failure for a single device is swallowed so the module keeps
# its "never raises on a missing/broken driver" contract.
_SOFT = (RuntimeError, AttributeError, OSError, ctypes.ArgumentError)


def get_cuda_device_specs() -> list[dict[str, Any]]:
    """Per-device specs: index, name, compute_capability (major,minor), sm_arch,
    architecture, multiprocessor_count, total/free memory (MB). Empty list if no
    driver/devices. Never raises -- returns ``[]`` so callers can branch on it.
    """
    lib = _lib()
    if lib is None:
        return []
    try:
        _check(lib, lib.cuInit(0), "cuInit")
        n = ctypes.c_int()
        _check(lib, lib.cuDeviceGetCount(ctypes.byref(n)), "cuDeviceGetCount")
    except _SOFT as e:
        logger.warning("CUDA init failed: %s", e)
        return []

    specs: list[dict[str, Any]] = []
    for i in range(n.value):
        try:
            dev = ctypes.c_int()
            _check(lib, lib.cuDeviceGet(ctypes.byref(dev), i), "cuDeviceGet")

            name = (c_char * 256)()
            _check(lib, lib.cuDeviceGetName(name, 256, dev), "cuDeviceGetName")

            major, minor = ctypes.c_int(), ctypes.c_int()
            _check(lib, lib.cuDeviceComputeCapability(ctypes.byref(major), ctypes.byref(minor), dev),
                   "cuDeviceComputeCapability")
            cc = (major.value, minor.value)

            sm_count = ctypes.c_int()
            _check(lib, lib.cuDeviceGetAttribute(ctypes.byref(sm_count),
                   CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev), "cuDeviceGetAttribute")

            spec: dict[str, Any] = {
                "index": i,
                "name": bytes(name).split(b"\0", 1)[0].decode(errors="replace"),
                "compute_capability": cc,
                "sm_arch": f"sm_{cc[0]}{cc[1]}",
                "architecture": SEMVER_TO_ARCH.get(cc, "unknown"),
                "multiprocessor_count": sm_count.value,
            }

            # memory needs a context; ensure it is always destroyed (no leak on error)
            create = _resolve(lib, "cuCtxCreate_v2", "cuCtxCreate")
            meminfo = _resolve(lib, "cuMemGetInfo_v2", "cuMemGetInfo")
            destroy = _resolve(lib, "cuCtxDestroy_v2", "cuCtxDestroy")
            if create is not None:
                ctx = ctypes.c_void_p()
                if create(ctypes.byref(ctx), 0, dev) == CUDA_SUCCESS:
                    try:
                        if meminfo is not None:
                            free_b, total_b = ctypes.c_size_t(), ctypes.c_size_t()
                            if meminfo(ctypes.byref(free_b), ctypes.byref(total_b)) == CUDA_SUCCESS:
                                spec["total_mem_mb"] = total_b.value / 1024 ** 2
                                spec["free_mem_mb"] = free_b.value / 1024 ** 2
                    finally:
                        if destroy is not None:
                            destroy(ctx)

            specs.append(spec)
        except _SOFT as e:  # one bad device shouldn't kill the rest or escape
            logger.warning("could not query device %d: %s", i, e)
    return specs


def get_cuda_architecture() -> list[dict[str, Any]]:
    """Lightweight per-device {index, name, compute_capability, sm_arch, architecture}
    (no context / memory query). Empty list if no driver/devices."""
    return [{k: s[k] for k in ("index", "name", "compute_capability", "sm_arch", "architecture")}
            for s in get_cuda_device_specs()]


def _by_index(index: int) -> dict[str, Any] | None:
    """The arch dict for PHYSICAL device ``index`` (matched on the 'index' field,
    not list position -- robust if an earlier device failed to query and was skipped)."""
    for a in get_cuda_architecture():
        if a["index"] == index:
            return a
    return None


def device_architecture(index: int = 0) -> dict[str, Any] | None:
    """{index, name, compute_capability, sm_arch, architecture} for device ``index``, or None."""
    return _by_index(index)


def device_sm(index: int = 0) -> str | None:
    """``sm_XX`` of device ``index`` (e.g. ``"sm_86"``), or None if unavailable."""
    a = _by_index(index)
    return a["sm_arch"] if a else None


def _cli() -> int:
    """``fw2u-cuda-info`` entry point: print a GPU/driver report."""
    import json
    if not have_cuda():
        print("No CUDA driver found (libcuda not loadable). CPU-only host.")
        return 0
    ver = driver_cuda_version()
    specs = get_cuda_device_specs()
    print(f"driver max CUDA version: {ver if ver is not None else 'unknown'}")
    print(f"devices: {len(specs)}")
    print(json.dumps(specs, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
