# tests/

The fullwave2-ultra test suite. Three tiers: CPU clean-checkout (always run),
env-gated (run only when pointed at external data/repos), and GPU shell gates
(need a built solver binary + a GPU).

`conftest.py` sets `collect_ignore_glob = ["_*.py"]`: underscore-prefixed files
are dev-only scratch harnesses (figure generators, GPU/out-of-repo experiments),
not collected.

## CPU / clean-checkout (no GPU, no dataset)

Run with `pytest tests/` on a fresh checkout; these pass with only numpy/scipy:

- `test_aexp_sign.py` -- `dbmhzcm2aexp` attenuation-sign invariant + golden values.
- `test_dcmap_clamp.py` -- `io_dat.write_dcmap` clamps the FD-zone map to `[0, ndmap-1]`.
- `test_modt_nyquist.py` -- adaptive `modT` keeps `fs/2 > f0` across ppw (pure numpy).
- `test_io_dat_3d.py` -- 3D `.dat` round-trips: `write/read_map_3d` layout
  `[i,j,k] at (i*nY+j)*nZ+k`, `write/read_coords_3d` `[all i; all j; all k]`,
  3D `write_dcmap` clamp.
- `test_geom.py` -- structural invariants of `make_xdc_c5_2v` (grid, ppw~6,
  coord shape/order, theta symmetry, determinism) + `map_to_coords` /
  `map_to_coords_3d` ordering.
- `test_fixture_parity.py` -- **core parity gate**: regenerates the committed
  `examples/tiny_nsims4` fixture into a tmp dir and asserts every `*.dat` is
  byte-identical (`filecmp.cmp(shallow=False)`) to the committed golden set.
  Proves `sim.write_fullwave_sim` + `medium` + `io_dat` reproduce the golden
  inputs bit-for-bit.

## Env-gated (external data / source repo)

Collected always; **skip green** unless the relevant env vars point at existing
resources:

- `test_parity_vs_fw2b.py` -- byte-for-byte parity of `geom`/`medium` vs the
  original `fw2b` package. Set `FW2U_FW2B_DIR` (or `FW2B_DIR`) to a local `fw2b`
  checkout to run it; **skips** on a clean checkout when neither is set.
- `test_regen_parity.py` -- regen `.dat` vs a reference oracle run dir. Needs
  `FW2U_MEDIUM_WS` + `FW2U_ORACLE_DIR` **and** an importable regen driver. The
  regen driver (`regen_fsa_ppw`) is abdominal-FSA-specific and is not part of
  this repo, so this always skips on a clean checkout.

## GPU shell gates (built solver + GPU required)

Shell scripts (authored separately); not Python, not collected by pytest. Each
runs a built CUDA binary from `bin/` on a fixture/run dir and diffs output:

- `test_batch_eq_solo.sh` -- 2D batch solver == per-element solo runs
  (BIT-EXACT) on the `tiny_nsims4` fixture.
- `test_opt6_batch_offset.sh` -- opt6 batch kernel vs offset batch layout.
- `test_3d_opt_eq_base.sh` -- 3D `bench_3d_opt` vs `bench_3d` base oracle.
  Asserts **numerical equivalence to a tight float tolerance** (relative
  ~2e-7), NOT bit-exactness: the opt kernels restructure the field
  accumulation, so float32 round-off differs (`--fmad=false` does not change
  this). 2D batch==solo, by contrast, IS bit-identical.
