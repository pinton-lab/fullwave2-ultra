# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gianmarco Pinton
"""Single source of truth for run/cache/data locations and GPU selection.

Everything is overridable by environment variable so the repo carries no
machine-specific absolute paths. Defaults are repo-relative or under
``$XDG_CACHE_HOME`` -- nothing points at a particular host.

Env vars (all optional). The ``FW2U_*`` names are canonical; the legacy
``FW2B_*`` names are accepted as a fallback so existing shells keep working:
  FW2U_ROOT       repo root              (default: parent of this package)
  FW2U_BIN_DIR    compiled solver dir    (default: $FW2U_ROOT/bin)
  FW2U_CACHE_DIR  marshalled input cache (default: $XDG_CACHE_HOME/fw2u or ~/.cache/fw2u)
  FW2U_RUN_DIR    scratch run dirs       (default: $FW2U_ROOT/runs)
  FW2U_DATA_DIR   medium .mat / oracle runs for parity tests (default: unset -> tests skip)
  FW2U_GPU        CUDA device index      (default: 0)

The same names are honoured by the shell scripts in ``scripts/``.
"""
from __future__ import annotations
import os


def _env(name: str, default=None):
    """Read FW2U_<name>, falling back to the legacy FW2B_<name>."""
    return os.environ.get(f"FW2U_{name}") or os.environ.get(f"FW2B_{name}") or default


ROOT = _env("ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cache_default() -> str:
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = xdg if xdg else os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "fw2u")


BIN_DIR = _env("BIN_DIR") or os.path.join(ROOT, "bin")
CACHE_DIR = _env("CACHE_DIR") or _cache_default()
RUN_DIR = _env("RUN_DIR") or os.path.join(ROOT, "runs")
DATA_DIR = _env("DATA_DIR")          # may be None -> data-dependent steps skip
GPU = _env("GPU", "0")


def data_path(*parts: str) -> str | None:
    """Join under FW2U_DATA_DIR (or FW2B_DATA_DIR), or None if unset."""
    if not DATA_DIR:
        return None
    return os.path.join(DATA_DIR, *parts)
