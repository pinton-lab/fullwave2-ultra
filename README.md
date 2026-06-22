# fullwave2-ultra

Reusable **fullwave / ultrasound simulation utilities**: a validated
Python pipeline for assembling fullwave solver inputs.

> **Status:** v0.1, pre-release. The Python package is here under the
> **Apache-2.0** license (see `LICENSE`) — use it freely, including commercially.
> The optimized CUDA solver **executables ship prebuilt in `bin/`** (see *Running the
> solver* below) under a **noncommercial** license (`LICENSE-binaries.txt`) — commercial
> use of the executables needs a separate license. Their CUDA source is **not** published.

## What's here
- **`fullwave2_ultra/`** — the importable package (pure `numpy`/`scipy`; imports
  without a GPU):
  - `io_dat` — the raw `.dat` solver I/O contract (2D + 3D).
  - `geom` — C5-2v transducer geometry + 2D/3D coord rasterization.
  - `medium` — extend/resample maps, scatterer regeneration, attenuation (Aexp).
  - `sim` — assemble a solver run directory (`.dat`) from maps + transducer coords.
  - `beamform` — `scan_convert` + MACH FSA beamformer (optional `[beamform]` extra).
  - `cuda_utils` — dependency-free (ctypes/libcuda) GPU + driver introspection.
  - `solver` — preflight (GPU/driver vs the binary's arches) + binary resolution + run.
  - `config` — `FW2U_*` environment configuration.
- **`tests/`** — CPU correctness/parity gates (run anywhere) + GPU gate scripts
  (skip cleanly without a built/fetched solver binary).
- **`examples/`** — synthetic `tiny_nsims4` (2D, committed golden fixture) and
  `tiny_3d` (regenerated) — no proprietary data.
- **`docs/io_contract.md`** — the `.dat` binary contract (2D + 3D).

## Install
```bash
pip install -e .                 # base: numpy + scipy
pip install -e .[beamform]       # + MACH GPU FSA beamformer (cupy-cuda12x)
pip install -e .[viz]            # + matplotlib (figure output)
pip install -e .[dev]            # + ruff, pytest
```
Every module imports without a GPU; `beamform.beamform_fsa` needs MACH+cupy only
at call time.

## Running the solver
The compiled CUDA solver executables (`bench_2d_batch`, `bench_3d_opt`, …) ship
**prebuilt in `bin/`** in this repo, under the **PolyForm Noncommercial 1.0.0** license
(`LICENSE-binaries.txt`): free for academic/noncommercial use, but **commercial use
requires a separate license** (contact the Pinton Lab). Their CUDA source is not
published. On a clone they're found automatically — no download, no env vars:

```bash
fw2u-cuda-info                       # report host GPU(s) + driver CUDA version
fw2u-preflight bin/bench_2d_batch    # is this GPU/driver covered by the binary?
python -c "from fullwave2_ultra import solver; solver.run('<rundir>', name='bench_2d_batch')"
```
`solver.resolve_binary()` searches `bin/` first. Verify the binaries against
`bin/SHA256SUMS` (`cd bin && sha256sum -c SHA256SUMS`); build provenance is in
`bin/MANIFEST.txt`.

If you instead install the package as a wheel (without this repo's `bin/`), point the
resolver at a hosted copy — it fetches `<FW2U_BINARY_BASEURL>/<tag>/<name>`:
```bash
export FW2U_BINARY_BASEURL=https://<host>/<path>
export FW2U_BINARY_TAG=<tag>
export FW2U_BINARY_DOWNLOAD=1
```
Build a run dir of `.dat` with `fullwave2_ultra.sim` (see `docs/io_contract.md`).

## Tests
```bash
make test     # pytest: CPU parity/gotcha gates run; GPU + data-gated tests skip
make lint     # ruff (F, E9)
```
CPU tests need no GPU and no data. Tests that need the original `fw2b` package, a
GPU, a fetched binary, or a dataset skip cleanly (set the relevant `FW2U_*` env to
enable them).

## Provenance & licensing
Dual-licensed by component:
- **Python package** (this repo) — **Apache-2.0** (`LICENSE`): use, modify, and
  redistribute freely, including commercially, subject to its terms.
- **Solver executables** (prebuilt in `bin/`) — **PolyForm Noncommercial 1.0.0**
  (`LICENSE-binaries.txt`): free for academic/noncommercial use; commercial use
  requires a separate commercial license. Their CUDA source is not published.

See `NOTICE` for provenance (the fullwave solver lineage).
