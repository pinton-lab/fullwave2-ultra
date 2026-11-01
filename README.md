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

### 3D accuracy correction (2026-07)

The 3D solvers (`bench_3d`, `bench_3d_opt`, and their `_m6` variants) received an
accuracy correction to their internal numerical sound-speed calibration, removing a
small bulk phase-velocity error (up to ~0.4% in water-speed media, ~1.7% in
high-sound-speed media such as bone) present in earlier releases. **3D outputs
therefore differ slightly from previous binaries** (arrival times and refraction at
high-contrast interfaces are more accurate); rerun 3D studies where absolute timing
matters. The **2D solvers are unchanged** (byte-identical outputs).

### `_opt6` accuracy fix (2026-07)

The reduced-precision `bench_*_opt6` variants previously under-attenuated weakly
absorbing media (an fp16 rounding issue in the stored absorption term: weak
per-step attenuation rounded to zero, giving up to several dB of excess amplitude
per slab transit). Fixed: `_opt6` outputs change in attenuating media and now track
the full-precision solvers to within ~0.01 dB. All other solvers are unchanged
(byte-identical outputs).

### `bench_*_m6` variants

`bench_2d_batch_m6`, `bench_2d_aexp_m6`, `bench_3d_opt_m6`, `bench_3d_m6`: alternate
solver builds, same run-dir I/O contract. Outputs are not byte-comparable with the
default `bench_*` set — pick one set per study. Usage is identical, e.g.
`solver.run('<rundir>', name='bench_3d_opt_m6')`.

### SPLAT binaries: frequency-dependent power-law attenuation

`bench_2d_batch_splat` (2D batch) and `bench_3d_opt_splat` (3D) apply a power-law
attenuation `alpha(f) = alpha0 * f^y`, calibrated over a frequency band around the
center frequency. They take an ordinary run dir plus four files:

| File | Contents |
| --- | --- |
| `alpha_coeff.dat` | `alpha0` in dB/(MHz^y cm), float32, one value per cell of the EXTENDED grid (same layout as `c.dat`) |
| `alpha_power.dat` | the exponent `y`, float32, same layout, `0 <= y <= 1.7` |
| `f0.dat` | the center frequency in Hz, one float32 |
| `attenuation_parameters.dat` | a copy of one of the two settings files in `params/` |

`params/attenuation_parameters_centered_v1.dat` calibrates over `f0/sqrt(2)` to
`sqrt(2)*f0` (recommended); `params/attenuation_parameters_author_v1.dat` over `f0` to
`2*f0`. Extend the alpha maps the way the writer extends the medium maps:
`medium.extendMap(map, nbdy + M)` in 2D, an edge pad of `nbdy + M` cells per side in
3D. Write `amap` as zeros, so the classic `Aexp` attenuation does not add on top.

Run the binary in the run dir:

```bash
bin/bench_2d_batch_splat --unlimited-materials    # or bin/bench_3d_opt_splat
```

With `--unlimited-materials` there is no limit on the number of distinct
`(rho, K, beta)` materials, so speckle media with millions of materials run. The
number of distinct `(alpha0, y)` pairs, quantized in 0.1 % steps, is limited to 65535.
Without the flag the binary uses a material table and exits with a `FATAL` message
when the medium has more than 65535 materials. `solver.run` does not pass the flag
yet, so call the binary directly (or through fullwave25).

These two binaries were built later than the rest of `bin/` (see `bin/MANIFEST.txt`).

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
