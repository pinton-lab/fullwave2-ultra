# Solver I/O contract (raw .dat)

The solvers read/write raw little-endian binary `.dat` from the run dir (CWD).
`fullwave2_ultra/io_dat.py` is the reference implementation (write/read for every file
below); `fullwave2_ultra/sim.py` (`write_fullwave_sim`, `write_fullwave_sim_3d`)
assembles a complete run dir.

The same contract serves the **2D batched** solver (`bench_2d_batch`, solo
`bench_2d_aexp`) and the **3D** solver (`bench_3d`, `bench_3d_opt`). 3D adds a `nZ`/`dZ`
axis and an `ncoords_add` scalar; differences are flagged **(3D)** below.

## Scalars (one value per file)
int32: `nX, nY, nT, nTic` (default `nT`), `modT` (default 1), `ncoords, ncoordszero,
ncoordsout, ndmap, nsims` (default 1 if `nsims.dat` absent → single-sim compatible).
float32: `dX, dY, dT, c0`.

**(3D):** add int32 `nZ` and `ncoords_add` (additional-source count; 0 when unused), and
float32 `dZ`. The 3D solver reads `nsims`-free (single-sim), so there is no `nsims.dat`.

## Maps — float32, C-order
- **2D:** `nX*nY`, row-major (`[i,j]` at `i*nY + j`).
- **(3D):** `nX*nY*nZ`, C-order (`[i,j,k]` at `(i*nY + j)*nZ + k`).

Files: `c, rho, K, beta, Aexp` (`Aexp` = the per-cell absorption term). PML arrays
(`kappax/y/u/w`, `apml/bpml{u,w,x,y}{1,2}`) are read if present but **unused** by the
Aexp kernels (a missing PML file only warns). `Aexp` must be present and ≤ 1 (it
attenuates -- `aexp <= 1`).

### `beta` — nonlinearity, sign convention

`beta` is the **textbook coefficient of nonlinearity**, entered directly:

```
beta = 1 + B/(2A)          # water ~3.5, liver ~4.8
```

Tabulated values are usually quoted as `B/A`, so convert first — do **not** put
`B/A` into the map. Set `beta = 0` for a linear run (the overwhelmingly common
case; the shipped fixtures all do).

With this convention the compressional half-cycle propagates faster than `c` and
the dilatational half-cycle slower, by `± beta·p/K` — i.e. a finite-amplitude wave
steepens on its leading edge, and shocks form on compression.

> **Changed 2026-08.** Binaries built before this date evaluated the nonlinear
> term with an inverted sign, realising an effective coefficient of `-beta`:
> harmonic *magnitudes* were correct (they scale with `|beta|`), but the waveform
> asymmetry was reversed — peak-positive vs peak-negative pressure, second-harmonic
> phase, and the direction of shock formation. **`beta = 0` runs are entirely
> unaffected** (the two forms are bit-identical there), so linear users need do
> nothing. If you previously compensated by supplying a *negated* map, remove that
> negation. Output from the older binaries reproduces exactly by negating the map.

`io_dat`: 2D `write_map(path, M(nX,nY))` / `read_map(path, nX, nY)`; 3D
`write_map_3d(path, V(nX,nY,nZ))` / `read_map_3d(path, nX, nY, nZ)`.

## Solver coefficient inputs
The solver binary computes **all** of its internal finite-difference coefficient
inputs at startup, from the sound-speed map (`c.dat`) plus `dT`/`dX`/`c0`. The
Python pipeline does **not** produce or ship any of them: a run directory built
by `fullwave2_ultra.sim` holds the medium maps, coords, source, and scalars, and
the solver derives the rest itself.

## Coords — int32, 0-based, "all-i then all-j (then all-k)"
- **2D:** `[all i ; all j]`. `icc.dat` (`ncoords*2`), `icczero.dat` (`ncoordszero*2`),
  `outc.dat` (`ncoordsout*2`). `io_dat.write_coords(path, i, j)`.
- **(3D):** `[all i ; all j ; all k]`. `icc.dat` (`ncoords*3`), `outc.dat`
  (`ncoordsout*3`). The solver reads `coords[m]=i`, `coords[ncoords+m]=j`,
  `coords[2*ncoords+m]=k`. `io_dat.write_coords_3d(path, i, j, k)` /
  `read_coords_3d(path, ncoords) -> (ncoords, 3)`.

## Sources — `icmat.dat`, float32
- **2D (batched):** `nsims*ncoords*nTic`, **sim-major** then coord-major then time:
  `value(sim s, coord m, time n)` at `((s*ncoords + m)*nTic + n)`. For `nsims=1` this is
  an ordinary single-sim icmat. `io_dat.write_icmat(path, blocks)` takes an iterable of
  `(ncoords, nTic)` arrays (one per sim), written sim-major.
- **(3D):** `ncoords*nTic`, **coord-major** then time (single sim — the 3D solver has no
  batch axis): `value(coord m, time n)` at `m*nTic + n`.

## Additive sources — `icc_add.dat` / `icmat_add.dat` (optional, 2D)
The `icc`/`icmat` channel is **Dirichlet**: the solver hard-sets `p` at the source
coords each step (`n < nTic`) and clamps them to 0 afterwards — dense source regions
therefore act as rigid screens. The additive channel instead **superimposes**
(`p += trace`) so distributed secondary sources (e.g. scatterer injection) stay
acoustically transparent, and is never zero-clamped once its traces run out.
- Scalars: int32 `ncoords_add` (absent/0 → channel disabled, output byte-identical),
  `nTic_add` (default `nTic`).
- `icc_add.dat` (`ncoords_add*2`, `[all i ; all j]`, same convention as `icc.dat`).
  Coords must be **unique** (one add per cell per step).
- `icmat_add.dat`: `nsims*ncoords_add*nTic_add`, sim-major like `icmat.dat`.
- Injection order per step: hard set / zero clamp, then additive.
- Writer: `sim.write_fullwave_sim(..., add_coords=pts, nTic_add=n)` writes the coords
  and scalars; the caller writes `icmat_add.dat` via `io_dat.write_icmat`.
- Gate: `tests/test_2d_additive_source.sh` (superposition, 2D Green's function,
  truncation byte-identity, batch==solo).

## Outputs — genout, float32, frame-major
`nframes = (nT-1)//modT`; a frame is written every `modT` steps.
- **2D (batched):** device-buffered, layout `[frame][sim][pts]`, written per sim (fresh
  each run): sim 0 → `genout.dat` (single-sim compatible), sim `s>0` → `genout_s<s>.dat`,
  each `nframes*ncoordsout` floats. `io_dat.read_genout(path, ncoordsout)` →
  `(nframes, ncoordsout)`; `io_dat.read_genout_batch(rundir, nsims, ncoordsout)`.
- **(3D):** one `genout.dat`, frame-major (`nframes*ncoordsout` floats); each modT frame
  is appended as it is computed. `io_dat.read_genout(path, ncoordsout)`.

## genout_mod — decimated full-field dump (optional, 2D + 3D)
An opt-in alternative to coordinate-based `genout` for outputting the *whole* field (or
a spatially-decimated subgrid) efficiently — no coordinate list, a coalesced per-frame
dump (the spatial analogue of `modT`). Enable by writing the int32 mod scalars (all
absent → disabled, `genout` **byte-unchanged**, so the parity gates stay valid). Per
output (`modT`) frame, the field at every (mod*)-th point of the full grid is appended,
frame-major, C-order, with `nX2 = ceil(nX/modX)` etc.
- **(3D):** scalars `modX.dat`, `modY.dat`, `modZ.dat`; `genout_mod.dat` =
  `nX2*nY2*nZ2` floats/frame, C-order (i,j,k). The 3D solver is single-sim.
  `io_dat.read_genout_mod(path, nX, nY, nZ, modX, modY, modZ)` → `(nframes, nX2, nY2, nZ2)`;
  `sim.write_fullwave_sim_3d(..., genout_mod=(mX,mY,mZ))` writes the scalars.
- **(2D):** scalars `modX.dat`, `modY.dat` (no `modZ`); `nX2*nY2` floats/frame, C-order
  (i,j). The **batched** solver writes one file *per sim* (same naming as `genout`): sim
  0 → `genout_mod.dat`, sim `s>0` → `genout_mod_s<s>.dat` (streamed per frame, not
  device-buffered — per-sim full fields are large). `genout_mod_name(s)` gives the name.
  `io_dat.read_genout_mod_2d(path, nX, nY, modX, modY)` → `(nframes, nX2, nY2)`;
  `io_dat.read_genout_mod_batch_2d(rundir, nsims, nX, nY, modX, modY)` →
  `(nsims, nframes, nX2, nY2)`; `sim.write_fullwave_sim(..., genout_mod=(mX,mY))` writes
  the scalars.

## Solver variants
The distributed solver binaries come in bit-exact and performance-tuned variants
(2D batched / single-sim oracle; 3D base / optimized). Variant behaviour is
documented alongside the binary distribution, not here. `fullwave2_ultra.solver`
selects and preflights a binary at runtime.
