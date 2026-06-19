# GPU solver-correctness gates & run scripts

These shell scripts drive the vendored CUDA solvers in `bin/` to verify the GPU
solver is correct and deterministic. They all **skip cleanly** (print `SKIP` and
`exit 0`) when the binaries are not built or a required fixture is absent, so they
are safe to run in a CPU-only / no-GPU environment.

Obtain the solver binaries (place a prebuilt binary in `bin/` or on PATH, or fetch via `FW2U_BINARY_BASEURL`+`FW2U_BINARY_TAG`+`FW2U_BINARY_DOWNLOAD=1` -- see ../README.md); GPU device is selected by
`$FW2U_GPU` (legacy `$FW2B_GPU` accepted), default `0`. Other paths come from the
`FW2U_*` env (`FW2U_ROOT`, `FW2U_RUN_DIR`, `FW2U_DATA_DIR`, ...); see
`fullwave2_ultra/config.py`. No machine-specific paths are baked in.

## tests/ -- correctness gates (PASS / FAIL / SKIP; exit 1 only on FAIL)

| script | what it asserts |
|---|---|
| `tests/test_batch_eq_solo.sh` | **Primary 2D gate.** Each batched sim's `genout` is **byte-identical** to a solo run of that element (`bench_2d_batch` vs the `bench_2d_aexp` solo oracle), on the committed `examples/tiny_nsims4` fixture. 2D batch==solo is bit-exact (shared kernels). |
| `tests/test_opt6_batch_offset.sh` | **OPT6 sbase regression.** Batched OPT6 sims must differ (`genout.dat != genout_s1.dat`); the old bug made every sim read sim-0's field offset. |
| `tests/test_3d_opt_eq_base.sh` | **3D gate.** `bench_3d_opt` agrees with the `bench_3d` base oracle within a **float32 tolerance** (`max|base-opt|/ptp(base) < 1e-4`), both finite, equal size. Regens the (uncommitted) 3D fixture via `examples/tiny_3d/make_fixture_3d.py`, both the default (constant-memory) path and a `--bigndmap` (global-memory) path. **NOTE:** 3D opt-vs-base is float32-equivalent (~2e-7), **NOT** bit-identical, because the optimized 3D kernels restructure the field accumulation (different summation order) -- unlike 2D, where batch==solo shares kernels and is bit-exact. So this gate uses a tolerance, by design. |

## scripts/ -- larger checks (summaries under `results/`, gitignored)

| script | what it does |
|---|---|
| `scripts/determinism.sh` | **Self-contained** on `examples/tiny_nsims4`: runs `bench_2d_batch` twice and asserts run-to-run bit identity of every per-sim `genout`, plus a width-invariance check (solo-run element 1 with `nsims=1` equals the batch's sim-0 `genout`). Writes `results/determinism.log`. |
| `scripts/scale128.sh` | Large-batch (`nsims=128`) batch==solo spot-check. **Gated behind env:** needs a real 128-element batch input dir at `$FW2U_RUN_DIR/B128` (or `$FW2U_DATA_DIR/B128`); the committed synthetic fixture is only `nsims=4`. Skips with a clear message otherwise. Writes `results/scale128.log`. |

### Intentionally not ported

The original `fw2b` `scripts/marshal_local.sh` (and `determinism.sh` /
`scale128.sh`'s dependence on it) marshalled an **abdominal** medium into 16- and
128-element batch inputs. That marshal step is dataset/domain-specific and is
**intentionally not ported** to this standalone package. `determinism.sh` here is
self-contained on the tiny synthetic fixture; `scale128.sh` is env-gated so it can
consume a pre-built 128-element input if one is supplied.
