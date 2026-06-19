# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gianmarco Pinton
"""Solver preflight + binary resolution + a thin launcher.

Ties the Python pipeline to the compiled CUDA solvers (``bin/bench_*``) with the
patterns learned from fullwave25 (github.com/pinton-lab/fullwave25):

* :func:`preflight` -- before launching, detect the GPU (via :mod:`.cuda_utils`,
  no cupy/toolkit needed) and check it against what the FAT BINARY actually
  contains (its embedded SASS arches + PTX fallback, read with ``cuobjdump``) and
  whether the driver is new enough. Fails with an actionable message
  ("GPU sm_XX not in this binary; rebuild or fetch a matching build") instead of a
  cryptic CUDA launch error.
* :func:`resolve_binary` -- locate a solver binary: built (``bin/``) -> per-tag
  cache (``~/.cache/fullwave2_ultra/bins/<tag>/``) -> optional download from a
  configured release URL (off by default; this repo builds locally). Mirrors
  fullwave25's ``ensure_binary`` so the package *could* be published thin.
* :func:`run` -- preflight then ``subprocess`` the solver in a run dir.

Imports without a GPU; introspection/preflight degrade gracefully.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import stat
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import config, cuda_utils

logger = logging.getLogger(__name__)

# The CUDA toolkit the solver binaries are built with (CUDA 12.3).
# A 12.x binary needs a driver supporting CUDA >= 12.0 (minor-version compat);
# >= BUILD_CUDA_VERSION is recommended. Overridable for rebuilds on another toolkit.
BUILD_CUDA_VERSION = float(os.environ.get("FW2U_BUILD_CUDA", "12.3"))
MIN_DRIVER_CUDA = float(int(BUILD_CUDA_VERSION))          # 12.x -> 12.0 floor
# PTX forward-compat virtual arch embedded in the fat binary (Makefile PTX_SM=90).
PTX_FORWARD_CC = (9, 0)
# Fallback SASS set (Makefile SMS default) used only when cuobjdump is unavailable.
EXPECTED_SASS = ("sm_60", "sm_61", "sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90")

BINARY_TAG = os.environ.get("FW2U_BINARY_TAG", "")          # e.g. "bin_v1"
BINARY_BASEURL = os.environ.get("FW2U_BINARY_BASEURL", "")  # e.g. a GitHub release dir URL
CACHE_BINS = Path(os.environ.get("FW2U_CACHE_DIR", str(Path(config.CACHE_DIR)))) / "bins"


class PreflightError(RuntimeError):
    """Raised when the host GPU/driver cannot run the given solver binary."""


def _sm_to_cc(sm: str) -> tuple[int, int]:
    """'sm_86' -> (8, 6); 'sm_120' -> (12, 0). NVIDIA sm codes are ``<major><minor>``
    with a SINGLE-digit minor (true for every shipping arch: sm_86, sm_90, sm_100,
    sm_103, sm_120, sm_121), so the minor is the last digit and the major is the rest."""
    n = sm.split("_", 1)[1]
    if not n[:-1] or not n.isdigit():
        raise ValueError(f"malformed CUDA arch code: {sm!r}")
    return (int(n[:-1]), int(n[-1]))


def _cuobjdump(binary: str | Path, *args: str) -> str | None:
    """Run cuobjdump on a binary; None if cuobjdump isn't installed or errors."""
    exe = shutil.which("cuobjdump")
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, *args, str(binary)], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:  # pragma: no cover
        logger.debug("cuobjdump failed: %s", e)
        return None
    return out.stdout if out.returncode == 0 else None


def embedded_sass_arches(binary: str | Path) -> list[str] | None:
    """Sorted list of SASS arches embedded in the fat binary (e.g. ['sm_60',...]),
    or None if cuobjdump is unavailable (introspection not possible)."""
    out = _cuobjdump(binary)
    if out is None:
        return None
    return sorted(set(re.findall(r"sm_\d+", out)), key=_sm_to_cc)


def embedded_ptx_arches(binary: str | Path) -> list[str] | None:
    """Sorted list of PTX (virtual) arches embedded for JIT forward-compat, or None."""
    out = _cuobjdump(binary, "--list-ptx")
    if out is None:
        return None
    return sorted(set(re.findall(r"sm_\d+|compute_\d+", out)), key=lambda s: _sm_to_cc(s.replace("compute_", "sm_")))


def preflight(binary: str | Path, *, device: int = 0, require_gpu: bool = True) -> dict:
    """Check the host can run ``binary`` on GPU ``device``. Returns a report dict
    (``ok``, ``status``, ``messages``, ``sm``, ``driver_cuda``, ``sass``, ...).

    Raises :class:`PreflightError` on a hard incompatibility (no GPU when
    required, driver too old, or the GPU's arch is neither in the binary's SASS
    nor JIT-able from its PTX). Warnings (e.g. minor-version-compat driver, or
    cuobjdump-unavailable assumptions) are collected in ``messages`` but do not
    raise. ``require_gpu=False`` turns "no GPU" into a soft (non-raising) status.
    """
    binary = Path(binary)
    report: dict = {"binary": str(binary), "device": device, "ok": False,
                    "status": "", "messages": [], "sm": None, "architecture": None,
                    "driver_cuda": None, "sass": None, "ptx": None}

    if not binary.exists():
        report["status"] = "missing_binary"
        raise PreflightError(f"solver binary not found: {binary} (build it from source or "
                             "fetch a prebuilt binary -- see resolve_binary() / README)")

    if not cuda_utils.have_cuda():
        report["status"] = "no_driver"
        msg = ("no CUDA driver found (libcuda not loadable) -- need an NVIDIA GPU + driver "
               ">= R525 (R545 recommended) to run the solver. See README.")
        if require_gpu:
            raise PreflightError(msg)
        report["messages"].append(msg)
        return report

    driver_cuda = cuda_utils.driver_cuda_version()
    report["driver_cuda"] = driver_cuda
    if driver_cuda is not None:
        if driver_cuda < MIN_DRIVER_CUDA:
            report["status"] = "driver_too_old"
            raise PreflightError(
                f"driver supports CUDA {driver_cuda}, but the binary is built with CUDA "
                f"{BUILD_CUDA_VERSION} and needs a driver supporting CUDA >= {MIN_DRIVER_CUDA} "
                "(>= R525). Update the NVIDIA driver. See README.")
        if driver_cuda < BUILD_CUDA_VERSION:
            report["messages"].append(
                f"driver supports CUDA {driver_cuda} < build {BUILD_CUDA_VERSION}; relying on "
                "12.x minor-version compatibility (recommended driver supports >= "
                f"{BUILD_CUDA_VERSION}).")

    arch_info = cuda_utils.device_architecture(device)   # matched on physical index
    if arch_info is None:
        report["status"] = "no_device"
        raise PreflightError(f"no CUDA device at index {device}")
    sm = arch_info["sm_arch"]
    report["sm"] = sm
    report["architecture"] = arch_info["architecture"]

    sass = embedded_sass_arches(binary)
    ptx = embedded_ptx_arches(binary)
    report["sass"], report["ptx"] = sass, ptx
    if sass is None:
        sass = list(EXPECTED_SASS)
        report["messages"].append(
            "cuobjdump unavailable -- assuming the Makefile-default SASS set "
            f"{EXPECTED_SASS}; install cuda-cuobjdump to verify the actual binary.")
    # PTX JIT floor = the LOWEST compute capability among the binary's ACTUAL
    # embedded PTX (PTX for compute_X JITs forward to any device cc >= X). Fall back
    # to the Makefile default only when cuobjdump couldn't introspect the binary.
    ptx_floor = (min(_sm_to_cc(p.replace("compute_", "sm_")) for p in ptx)
                 if ptx else PTX_FORWARD_CC)

    if sm in sass:
        report["ok"], report["status"] = True, "native"
    elif _sm_to_cc(sm) >= ptx_floor:
        report["ok"], report["status"] = True, "ptx_jit"
        report["messages"].append(
            f"{sm} not in the binary's SASS; will JIT from the embedded "
            f"compute_{ptx_floor[0]}{ptx_floor[1]} PTX on first launch (one-time cost; "
            "set CUDA_CACHE_PATH on read-only filesystems).")
    else:
        report["status"] = "arch_uncovered"
        raise PreflightError(
            f"GPU {sm} ({report['architecture']}) is not in the binary's SASS {sass} and is "
            f"below its PTX floor (compute_{ptx_floor[0]}{ptx_floor[1]}). Obtain a binary built "
            f"for {sm} -- rebuild from source with that arch, or fetch a matching build.")
    return report


def resolve_binary(name: str = "bench_2d_batch", *, extra_dirs: tuple[str, ...] = (),
                   tag: str | None = None, base_url: str | None = None,
                   allow_download: bool | None = None) -> Path:
    """Locate a solver binary: built dir(s) -> per-tag cache -> optional download.

    Search order: ``extra_dirs`` + ``config.BIN_DIR`` (where ``make`` puts them),
    then ``CACHE_BINS/<tag>/``. If still missing and downloading is enabled
    (``allow_download`` or ``FW2U_BINARY_DOWNLOAD=1``) with a ``base_url``
    (``FW2U_BINARY_BASEURL``) + ``tag`` (``FW2U_BINARY_TAG``), fetch ``<base_url>/<tag>/<name>``
    into the cache. Raises a clear error (with manual-placement hint) otherwise.
    """
    tag = tag if tag is not None else BINARY_TAG
    base_url = base_url if base_url is not None else BINARY_BASEURL
    if allow_download is None:
        allow_download = os.environ.get("FW2U_BINARY_DOWNLOAD", "") == "1"

    searched = []
    for d in (*extra_dirs, config.BIN_DIR):
        p = Path(d) / name
        searched.append(str(p))
        if p.exists():
            logger.debug("resolved binary (built): %s", p)
            return p
    cached = CACHE_BINS / (tag or "untagged") / name
    searched.append(str(cached))
    if cached.exists():
        logger.debug("resolved binary (cache): %s", cached)
        return cached

    if allow_download and base_url and tag:
        scheme = urllib.parse.urlparse(base_url).scheme
        if scheme not in ("http", "https"):
            raise ValueError(f"FW2U_BINARY_BASEURL must be http(s) (got scheme {scheme!r}) "
                             "-- refusing to fetch a binary over a non-network scheme.")
        url = f"{base_url.rstrip('/')}/{tag}/{name}"
        cached.parent.mkdir(parents=True, exist_ok=True)
        tmp = cached.with_suffix(".download_tmp")
        logger.info("downloading solver binary %s from %s", name, url)
        manual = (f"Verify release '{tag}' contains '{name}', set FW2U_BINARY_TAG, "
                  f"or place it manually at {cached}.")
        try:
            urllib.request.urlretrieve(url, tmp)  # noqa: S310
            tmp.rename(cached)
            cached.chmod(cached.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            return cached
        except urllib.error.HTTPError as e:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"could not download '{name}' from {url}: HTTP {e.code} {e.reason}. "
                               f"{manual}") from e
        except Exception as e:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"failed to download '{name}' from {url}: {e}. {manual}") from e

    hint = ("build it from source, or set FW2U_BINARY_BASEURL + FW2U_BINARY_TAG + "
            f"FW2U_BINARY_DOWNLOAD=1 to fetch it, or place it at {cached}")
    raise FileNotFoundError(f"solver binary '{name}' not found. Searched: {searched}. {hint}.")


def run(rundir: str | Path, *, name: str = "bench_2d_batch", binary: str | Path | None = None,
        device: int | None = None, preflight_gpu: bool = True, check: bool = True,
        capture_output: bool = False, timeout: float | None = None) -> subprocess.CompletedProcess:
    """Run a solver binary in ``rundir`` (which must ALREADY hold the ``.dat`` inputs;
    see :mod:`.sim` / ``docs/io_contract.md``).

    Resolves the binary (if not given), and -- unless ``preflight_gpu=False`` --
    :func:`preflight`s the GPU first (raises :class:`PreflightError` on an
    incompatible/absent GPU). Sets ``CUDA_VISIBLE_DEVICES`` and ``subprocess.run``s
    with cwd=rundir. ``check`` is forwarded to ``subprocess.run`` (raise on a
    nonzero solver exit) -- independent of the preflight toggle.
    """
    binary = Path(binary) if binary is not None else resolve_binary(name)
    dev = device if device is not None else int(config.GPU or 0)
    if preflight_gpu:
        preflight(binary, device=dev)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(dev)}
    return subprocess.run([str(Path(binary).resolve())], cwd=str(rundir), env=env,
                          capture_output=capture_output, text=True, timeout=timeout, check=check)


def _cli() -> int:
    """``fw2u-preflight`` entry point: report whether a binary can run here."""
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Preflight a fullwave2-ultra solver binary against this host's GPU/driver.")
    ap.add_argument("binary", nargs="?", help="path to a solver binary (default: resolve bench_2d_batch)")
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()
    try:
        binary = args.binary or resolve_binary()
        report = preflight(binary, device=args.device, require_gpu=True)
    except (PreflightError, FileNotFoundError) as e:
        print(f"PREFLIGHT FAIL: {e}")
        return 1
    print(json.dumps(report, indent=2, default=str))
    print(f"OK ({report['status']}): {report['sm']} can run {Path(report['binary']).name}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
