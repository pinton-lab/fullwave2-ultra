# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gianmarco Pinton
"""fullwave2-ultra: reusable fullwave/ultrasound utilities.

A clean, byte-validated Python pipeline for the fullwave solvers. The optimized
2D batched + 3D Aexp CUDA solver executables are distributed separately (see
``solver.resolve_binary``); this package is the pipeline around them:

  io_dat   -- the raw .dat solver contract (2D + 3D), byte-identical to fw2b.
  geom     -- C5-2v transducer geometry + 2D/3D coord rasterization.
  medium   -- general medium helpers (extend/resample, scatterers, Aexp).
  sim       -- assemble a solver run directory (.dat) from maps + coords.
  beamform  -- scan_convert + MACH FSA beamformer (optional [beamform] extra).
  cuda_utils-- dependency-free (ctypes/libcuda) GPU + driver introspection.
  solver    -- preflight (GPU/driver vs the binary's arches) + binary resolution + run.
  config    -- FW2U_* / FW2B_* env configuration.

Everything imports without a GPU; only ``beamform.beamform_fsa`` needs MACH+cupy
at call time. CLIs: ``fw2u-cuda-info`` (host GPU report), ``fw2u-preflight`` (can a
solver binary run here?).
"""
__version__ = "0.1.0.dev0"
