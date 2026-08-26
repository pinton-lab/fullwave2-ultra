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

### Material-LUT eligibility (`USE_MATLUT`) — what actually caps it

The `USE_MATLUT` fast path replaces the fp32 `rho`/`K`/`beta` maps with a u16
index into an exact fp32 lookup table (in 3D it also frees those maps and the
int32 zone map from the device, ~5 GB at 8 ppw). It is declined, falling back to
the general path, when either:

1. the medium holds **more than 65535 distinct `(rho, K, beta)` triples**, or
2. the largest zone index exceeds 65535, i.e. `round(max(c) - min(c)) + 1 > 65536`.

The key is the exact **float32 bit pattern** of the triple. Two consequences
that decide how you build a map:

- **`Aexp` is NOT part of the key.** A graded absorbing ring gives almost every
  boundary cell its own `Aexp`, but that never affects eligibility. The fast
  path is not unreachable in 3D by construction.
- **`K = c^2 * rho`, so `c` enters only through `K`**, and the triple count is
  the number of distinct `(c, rho)` pairs (times distinct `beta`, usually 1
  since `beta = 0` for linear runs). Continuous `c` and `rho` from a CT is what
  trips the cap — for a 682x682x596 skull, immediately.

The fix is to quantize, and to quantize the **source scalar the maps derive
from** rather than each map independently. Quantizing `c` and `rho` separately
to 256 levels each admits 65536 pairs and still declines; quantizing the single
porosity/HU field they are both computed from to 1024 levels yields 1024
triples. `medium.quantize_for_matlut` does this, measures the realized triple
count exactly, and reports the induced error; `medium.count_materials` gives the
count on its own. The solver's decline message names which of the two conditions
fired.

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

## Source formats — what the solvers can represent

A source can be stated three ways in a staggered `p`/`u` scheme. Two are
implemented:

| format | files | 2D | 3D | note |
|---|---|---|---|---|
| **imposed pressure** (Dirichlet) | `icc` / `icmat` | yes | yes | overwrites `p`; the region acts as a rigid screen |
| **added pressure** | `icc_add` / `icmat_add` | yes | **no** | superimposes; the region stays transparent |
| **normal velocity** (piston) | — | no | no | not implemented |

**There is no velocity source.** A pressure source substitutes for a
normal-velocity boundary condition only where `p = rho*c*u_n`, i.e. a plane wave
into a locally homogeneous medium. At a curved or focused aperture, or with the
face on a material contrast, the two differ, and a dipole source has no
pressure-only representation short of a two-cell hack. Do not fake one by
rescaling `icmat`.

**The 3D solvers refuse, rather than ignore, an additive source.** They read
`ncoords_add.dat` optionally and abort with `FATAL: ncoords_add=N but the 3D
solver has no additive source channel` when it is nonzero. Older 3D binaries
read the scalar, reported it, and had no kernel behind it, so a 3D deck carrying
a real additive source ran to completion with those sources silently missing.

## Sources — `icmat.dat`, float32
- **2D (batched):** `nsims*ncoords*nTic`, **sim-major** then coord-major then time:
  `value(sim s, coord m, time n)` at `((s*ncoords + m)*nTic + n)`. For `nsims=1` this is
  an ordinary single-sim icmat. `io_dat.write_icmat(path, blocks)` takes an iterable of
  `(ncoords, nTic)` arrays (one per sim), written sim-major.
- **(3D):** `ncoords*nTic`, **coord-major** then time (single sim — the 3D solver has no
  batch axis): `value(coord m, time n)` at `m*nTic + n`.

### Size limits

`ncoords * nTic` is an **element** count, not a byte count, and the solvers
compute it in 64 bits — a driven-surface source of a few hundred thousand cells
crosses 2^31 elements at realistic `nTic`, and older binaries wrapped there
(a host abort, and on the device a wrapped negative index that injected garbage
instead of the trace). There is no 2^31 cap in the current binaries. What does
bind is memory: the trace array is `ncoords * nTic * 4` bytes on the host AND on
the device, so a 487918-cell shell at `nTic = 5856` is 11.4 GB in each place.
The host allocation is checked and reports the count and the GB; the device
allocation reports a CUDA OOM.

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

## Stability — the CFL condition

The solvers use a staggered leapfrog whose dispersion relation is

```
sin(w*dT/2) = r * Sbar(k*dX, n),        r = c*dT/dX
```

so `w` is real, and the scheme stable, only while `r*Sbar <= 1` everywhere in
the Brillouin zone and over every direction `n`. `Sbar` itself depends on `r`,
so the limit is a root find rather than a division. The worst direction is the
main diagonal.

| dim | M | `r_max` | `1/sqrt(dim)` (plain-leapfrog folklore) |
|---|---|---|---|
| 2 | 8 | **0.5326** | 0.7071 |
| 2 | 6 | **0.6108** | 0.7071 |

Available as `stability.cfl_limit(M=, dim=)` and `stability.CFL_LIMIT`.

**The optimized taps are LESS stable than a plain second-order leapfrog.**
Raising phase accuracy raises `|Sbar|` near the zone corner, which is exactly
what the bound integrates. Never size a time step with `1/sqrt(dim)`.

The 3D limits are not published in this package. Size a 3D time step
conservatively; `stability.check_cfl` still reports the realized ratio there and
warns on the reference-speed trap below, it just has no threshold to compare
against.

### `r` is local; `cfl` is nominal

`r = c*dT/dX` uses the LOCAL sound speed, so the constraint is set by the
**fastest material in the map**. The deck writers take `cfl` and set
`dT = dX/c0*cfl`, referencing it to `c0`, so the realized ratio is

```
r_realized = cfl * max(c) / c0
```

With `c0 = 1540` and cortical bone at 2900 m/s, a requested `cfl = 0.30`
realizes **0.565**. Such a run is clean until the wave reaches the bone and then
goes to NaN across the whole field — nothing about the requested number looks
wrong, only the realized one does. Both deck writers now check this and warn
naming both numbers (`cfl_check="warn"`/`"ignore"` to relax; in 2D, where a
limit is published, exceeding it raises).

Attenuation only damps, so it never loosens the bound, but it does mask a
marginally unstable configuration for many steps before it grows. Nonlinearity
(`beta != 0`) eats margin. Keep real headroom rather than sitting on the limit.

## Solver variants
The distributed solver binaries come in bit-exact and performance-tuned variants
(2D batched / single-sim oracle; 3D base / optimized). Variant behaviour is
documented alongside the binary distribution, not here. `fullwave2_ultra.solver`
selects and preflights a binary at runtime.
