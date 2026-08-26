"""Gates for the USE_MATLUT eligibility rule (docs/io_contract.md).

The solver keys its u16 material LUT on the exact float32 BIT PATTERN of the
(rho, K, beta) triple and declines above 65535 distinct triples. Aexp is NOT in
the key -- a graded absorbing ring cannot make the fast path unreachable, which
is the thing callers most often assume. These tests pin the rule so it cannot
drift silently away from src/*.cu.
"""
import numpy as np
import pytest

from fullwave2_ultra.medium import (MATLUT_MAX_MATERIALS, count_materials,
                                    quantize_for_matlut, aexp_from_amap)


def test_key_is_rho_K_beta_and_c_enters_only_through_K():
    n = 32
    rho = np.full((n, n), 1000.0)
    c = np.broadcast_to(np.where(np.arange(n)[:, None] < n // 2,
                                 1540.0, 2900.0), (n, n)).copy()
    assert count_materials(rho, c) == 2                 # two speeds -> two K
    assert count_materials(rho, np.full((n, n), 1540.0)) == 1


def test_graded_Aexp_does_not_affect_eligibility():
    """The single most load-bearing fact: the absorbing ring is not in the key."""
    n = 48
    c = np.full((n, n), 1540.0); rho = np.full((n, n), 1000.0)
    amap = np.zeros((n, n))
    aexp = aexp_from_amap(amap, 1540.0, 2 * np.pi * 1e6, 1e-8, nbdy=12)
    assert np.unique(aexp).size > 2                     # the sponge really is graded
    assert count_materials(rho, c) == 1                 # ... and irrelevant here


def test_independent_maps_multiply_but_correlated_maps_do_not():
    """Why quantizing each map separately is the wrong move."""
    rng = np.random.default_rng(0)
    shape = (64, 64, 40)                 # > 65535 cells: enough to exhaust the cap
    c = rng.uniform(1540, 2900, shape)
    indep = rng.uniform(1000, 1900, shape)
    corr = 1000 + 0.66 * (c - 1540)                     # rho a function of c

    _, _, _, i_ind = quantize_for_matlut(c, indep)
    _, _, _, i_cor = quantize_for_matlut(c, corr)
    assert i_ind["eligible"] and i_cor["eligible"]
    # independent fields exhaust the budget and force a coarse quantization;
    # correlated ones stay fine-grained because the count is MEASURED
    assert i_cor["n_levels"] > i_ind["n_levels"]
    assert i_cor["c_step"] < i_ind["c_step"]


def test_quantize_lands_under_the_cap_and_reports_its_error():
    rng = np.random.default_rng(1)
    shape = (64, 64, 40)
    c = rng.uniform(1540, 2900, shape); rho = rng.uniform(1000, 1900, shape)
    cq, rq, nq, info = quantize_for_matlut(c, rho)
    assert info["eligible"] and info["n_materials"] <= MATLUT_MAX_MATERIALS
    assert count_materials(rq, cq, nq) == info["n_materials"]
    assert np.max(np.abs(c - cq)) <= info["c_max_abs_err"] + 1e-9
    # step/2, plus one float32 ulp: the levels are computed in float64 and
    # stored as float32, which is what the solver keys on.
    ulp = float(np.spacing(np.float32(c.max())))
    assert info["c_max_abs_err"] <= info["c_step"] / 2 + ulp
    assert nq is None


def test_constant_map_is_not_quantized_into_extra_materials():
    n = 24
    c = np.full((n, n), 1540.0); rho = np.full((n, n), 1000.0)
    cq, rq, _, info = quantize_for_matlut(c, rho)
    assert info["n_materials"] == 1
    assert np.all(cq == 1540.0) and np.all(rq == 1000.0)


def test_a_realistic_skull_sized_medium_declines_before_quantization():
    """The reported case: continuous c and rho from a CT overflow immediately."""
    rng = np.random.default_rng(2)
    shape = (64, 64, 40)                                 # 163k cells is plenty
    c = rng.uniform(1540, 2900, shape); rho = rng.uniform(1000, 1900, shape)
    assert count_materials(rho, c) > MATLUT_MAX_MATERIALS
    _, _, _, info = quantize_for_matlut(c, rho)
    assert info["eligible"]
