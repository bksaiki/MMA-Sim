"""Unit tests verifying bit-accurate equivalence between the scalar dot product
functions in ``mmasim.simulator.arithmetic`` and the batched counterparts in
``mmasim.simulator.batch_arithmetic``.

Each test runs the scalar function on every row of an n×k matrix and collects
the results, then runs the batch function on the whole matrix, and asserts
exact (bit-level) equality between the two outputs.
"""

import math

import pytest
import torch

from mmasim.simulator.arithmetic import (
    pairwise_dot,
    fused_dot_add,
    nv_fused_dot_add,
    nv_fused_dot_add_with_block_scale,
    amd_fused_dot_rd_add,
    extract_significand_exponent,
    fused_sum,
    unpack_fp4_tensor,
)
from mmasim.simulator.batch_arithmetic import (
    batch_pairwise_dot,
    batch_fused_dot_add,
    batch_nv_fused_dot_add,
    batch_nv_fused_dot_add_with_block_scale,
    batch_amd_fused_dot_rd_add,
    _extract_sig_exp_batch,
    _fused_sum_batch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fp32(n: int, k: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(n, k, dtype=torch.float32)


def make_fp16(n: int, k: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(n, k, dtype=torch.float16)


def make_bf16(n: int, k: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(n, k, dtype=torch.bfloat16)


# ---------------------------------------------------------------------------
# pairwise_dot / batch_pairwise_dot
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,k", [(1, 4), (3, 4), (4, 8), (7, 16)])
def test_batch_pairwise_dot_basic(n, k):
    """batch_pairwise_dot must match pairwise_dot row-by-row."""
    a = make_fp32(n, k, seed=1)
    b = make_fp32(n, k, seed=2)

    expected = torch.tensor(
        [pairwise_dot(a[i], b[i]) for i in range(n)], dtype=torch.float32
    )
    result = batch_pairwise_dot(a, b)

    assert torch.equal(result, expected), (
        f"batch_pairwise_dot mismatch (n={n}, k={k}):\n"
        f"expected={expected}\nresult={result}"
    )


@pytest.mark.parametrize("n,k", [(2, 4), (4, 8)])
def test_batch_pairwise_dot_flush_denormal(n, k):
    """batch_pairwise_dot with flush_denormal=True must match pairwise_dot row-by-row."""
    a = make_fp32(n, k, seed=3)
    b = make_fp32(n, k, seed=4)

    expected = torch.tensor(
        [pairwise_dot(a[i], b[i], flush_denormal=True) for i in range(n)],
        dtype=torch.float32,
    )
    result = batch_pairwise_dot(a, b, flush_denormal=True)

    assert torch.equal(result, expected), (
        f"batch_pairwise_dot (flush_denormal=True) mismatch (n={n}, k={k}):\n"
        f"expected={expected}\nresult={result}"
    )


def test_batch_pairwise_dot_zeros():
    """Both inputs all-zero: result must be exactly 0.0."""
    n, k = 4, 8
    a = torch.zeros(n, k, dtype=torch.float32)
    b = torch.zeros(n, k, dtype=torch.float32)
    result = batch_pairwise_dot(a, b)
    assert torch.equal(result, torch.zeros(n, dtype=torch.float32))


def test_batch_pairwise_dot_single_row():
    """n=1 batch must equal the scalar result."""
    k = 8
    a = make_fp32(1, k, seed=5)
    b = make_fp32(1, k, seed=6)
    scalar = pairwise_dot(a[0], b[0])
    batched = batch_pairwise_dot(a, b)
    assert batched.shape == (1,)
    assert batched[0].item() == scalar


# ---------------------------------------------------------------------------
# fused_dot_add / batch_fused_dot_add
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,k", [(1, 4), (3, 8), (5, 16)])
def test_batch_fused_dot_add(n, k):
    """batch_fused_dot_add must match fused_dot_add row-by-row."""
    a = make_fp16(n, k, seed=10)
    b = make_fp16(n, k, seed=11)
    c = torch.zeros(n, dtype=torch.float32)
    scale_a = torch.ones(n, dtype=torch.float32)
    scale_b = torch.ones(n, dtype=torch.float32)
    n_frac = 24

    expected = [
        fused_dot_add(a[i], b[i], c[i], n_frac, scale_a[i], scale_b[i])
        for i in range(n)
    ]
    result = batch_fused_dot_add(a, b, c, n_frac, scale_a, scale_b)

    assert len(result) == n
    for i in range(n):
        s_e, exp_e = expected[i]
        s_r, exp_r = result[i]
        assert s_e == s_r and exp_e == exp_r, (
            f"batch_fused_dot_add mismatch at row {i}: "
            f"expected=({s_e}, {exp_e}), got=({s_r}, {exp_r})"
        )


@pytest.mark.parametrize("n,k", [(2, 8), (4, 16)])
def test_batch_fused_dot_add_with_nonunit_scales(n, k):
    """batch_fused_dot_add must match fused_dot_add with non-unit scales."""
    torch.manual_seed(20)
    a = torch.randn(n, k, dtype=torch.float16)
    b = torch.randn(n, k, dtype=torch.float16)
    c = torch.randn(n, dtype=torch.float32)
    scale_a = torch.abs(torch.randn(n, dtype=torch.float32)) + 0.1
    scale_b = torch.abs(torch.randn(n, dtype=torch.float32)) + 0.1
    n_frac = 24

    expected = [
        fused_dot_add(a[i], b[i], c[i], n_frac, scale_a[i], scale_b[i])
        for i in range(n)
    ]
    result = batch_fused_dot_add(a, b, c, n_frac, scale_a, scale_b)

    for i in range(n):
        s_e, exp_e = expected[i]
        s_r, exp_r = result[i]
        assert s_e == s_r and exp_e == exp_r, (
            f"batch_fused_dot_add (non-unit scales) mismatch at row {i}: "
            f"expected=({s_e}, {exp_e}), got=({s_r}, {exp_r})"
        )


# ---------------------------------------------------------------------------
# nv_fused_dot_add / batch_nv_fused_dot_add
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "n,k,output_type",
    [
        (1, 8, "f32"),
        (3, 8, "f32"),
        (4, 16, "f32"),
        (3, 8, "f16"),
        (4, 16, "f16"),
    ],
)
def test_batch_nv_fused_dot_add(n, k, output_type):
    """batch_nv_fused_dot_add must match nv_fused_dot_add row-by-row."""
    a = make_fp16(n, k, seed=30)
    b = make_fp16(n, k, seed=31)
    out_dtype = torch.float16 if output_type == "f16" else torch.float32
    c = torch.zeros(n, dtype=out_dtype)
    n_frac = 24

    expected = torch.stack(
        [nv_fused_dot_add(a[i], b[i], c[i], n_frac, output_type) for i in range(n)]
    )
    result = batch_nv_fused_dot_add(a, b, c, n_frac, output_type)

    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    # bit-accurate: view as integer for NaN-safe comparison
    assert torch.equal(result.view(torch.int16 if out_dtype == torch.float16 else torch.int32),
                       expected.view(torch.int16 if out_dtype == torch.float16 else torch.int32)), (
        f"batch_nv_fused_dot_add mismatch (n={n}, k={k}, output_type={output_type}):\n"
        f"expected={expected}\nresult={result}"
    )


@pytest.mark.parametrize("n,k", [(2, 8), (4, 16)])
def test_batch_nv_fused_dot_add_with_scales(n, k):
    """batch_nv_fused_dot_add must match nv_fused_dot_add with explicit scales."""
    a = make_fp16(n, k, seed=40)
    b = make_fp16(n, k, seed=41)
    c = torch.zeros(n, dtype=torch.float32)
    scale_a = torch.abs(torch.randn(n, dtype=torch.float32)) + 0.1
    scale_b = torch.abs(torch.randn(n, dtype=torch.float32)) + 0.1
    n_frac, output_type = 24, "f32"

    expected = torch.stack(
        [
            nv_fused_dot_add(a[i], b[i], c[i], n_frac, output_type, scale_a[i], scale_b[i])
            for i in range(n)
        ]
    )
    result = batch_nv_fused_dot_add(a, b, c, n_frac, output_type, scale_a, scale_b)

    assert torch.equal(result.view(torch.int32), expected.view(torch.int32)), (
        f"batch_nv_fused_dot_add (with scales) mismatch:\nexpected={expected}\nresult={result}"
    )


def test_batch_nv_fused_dot_add_none_scales():
    """Passing scale_a=None and scale_b=None must give same result as explicit ones=1."""
    n, k = 3, 8
    a = make_fp16(n, k, seed=50)
    b = make_fp16(n, k, seed=51)
    c = torch.zeros(n, dtype=torch.float32)
    n_frac, output_type = 24, "f32"

    result_none = batch_nv_fused_dot_add(a, b, c, n_frac, output_type, None, None)
    scale_one = torch.ones(n, dtype=torch.float32)
    result_ones = batch_nv_fused_dot_add(a, b, c, n_frac, output_type, scale_one, scale_one)

    assert torch.equal(result_none.view(torch.int32), result_ones.view(torch.int32))


# ---------------------------------------------------------------------------
# nv_fused_dot_add_with_block_scale / batch_nv_fused_dot_add_with_block_scale
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,k,n_frac", [(1, 16, 25), (3, 16, 25), (4, 32, 35)])
def test_batch_nv_fused_dot_add_with_block_scale(n, k, n_frac):
    """batch_nv_fused_dot_add_with_block_scale must match scalar version row-by-row."""
    torch.manual_seed(60)
    # Simulate packed fp4 data (uint8), 2 fp4 values per byte → k/2 bytes per row
    a_packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8)
    b_packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8)

    from mmasim.simulator.arithmetic import unpack_fp4_tensor

    # Unpack each row: shape (n, k)
    a = torch.stack([unpack_fp4_tensor(a_packed[i]) for i in range(n)])
    b = torch.stack([unpack_fp4_tensor(b_packed[i]) for i in range(n)])

    c = torch.zeros(n, dtype=torch.float32)
    num_blocks = k // 16
    # Use float8_e8m0fnu scales (E8M0)
    scale_a = torch.ones(n, num_blocks, dtype=torch.float32)
    scale_b = torch.ones(n, num_blocks, dtype=torch.float32)

    expected = torch.stack(
        [
            nv_fused_dot_add_with_block_scale(
                a[i], b[i], c[i], scale_a[i], scale_b[i], n_frac
            )
            for i in range(n)
        ]
    )
    result = batch_nv_fused_dot_add_with_block_scale(a, b, c, scale_a, scale_b, n_frac)

    assert result.shape == expected.shape
    assert torch.equal(result.view(torch.int32), expected.view(torch.int32)), (
        f"batch_nv_fused_dot_add_with_block_scale mismatch (n={n}, k={k}):\n"
        f"expected={expected}\nresult={result}"
    )


# ---------------------------------------------------------------------------
# amd_fused_dot_rd_add / batch_amd_fused_dot_rd_add
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "n,k,is_fp8",
    [
        (1, 4, False),
        (3, 4, False),
        (4, 8, False),
        (2, 16, False),
        (3, 16, True),
    ],
)
def test_batch_amd_fused_dot_rd_add(n, k, is_fp8):
    """batch_amd_fused_dot_rd_add must match amd_fused_dot_rd_add row-by-row."""
    a = make_fp16(n, k, seed=70)
    b = make_fp16(n, k, seed=71)
    c = torch.zeros(n, dtype=torch.float32)
    n_frac = 24

    expected = torch.tensor(
        [
            amd_fused_dot_rd_add(a[i], b[i], c[i], n_frac, is_fp8)
            for i in range(n)
        ],
        dtype=torch.float64,
    )
    result = batch_amd_fused_dot_rd_add(a, b, c, n_frac, is_fp8)

    assert result.shape == expected.shape
    # Bit-accurate comparison via int64 view (handles NaN correctly)
    assert torch.equal(result.view(torch.int64), expected.view(torch.int64)), (
        f"batch_amd_fused_dot_rd_add mismatch (n={n}, k={k}, is_fp8={is_fp8}):\n"
        f"expected={expected}\nresult={result}"
    )


@pytest.mark.parametrize("n,k", [(2, 8), (4, 16)])
def test_batch_amd_fused_dot_rd_add_nonzero_c(n, k):
    """batch_amd_fused_dot_rd_add must match scalar with nonzero accumulators."""
    torch.manual_seed(80)
    a = torch.randn(n, k, dtype=torch.float16)
    b = torch.randn(n, k, dtype=torch.float16)
    c = torch.randn(n, dtype=torch.float32)
    n_frac = 24

    expected = torch.tensor(
        [amd_fused_dot_rd_add(a[i], b[i], c[i], n_frac) for i in range(n)],
        dtype=torch.float64,
    )
    result = batch_amd_fused_dot_rd_add(a, b, c, n_frac)

    assert torch.equal(result.view(torch.int64), expected.view(torch.int64)), (
        f"batch_amd_fused_dot_rd_add (nonzero c) mismatch:\n"
        f"expected={expected}\nresult={result}"
    )


# ---------------------------------------------------------------------------
# Private helpers: _extract_sig_exp_batch and _fused_sum_batch
# ---------------------------------------------------------------------------

def test_extract_sig_exp_batch_normal_float32():
    """_extract_sig_exp_batch must match extract_significand_exponent for every element."""
    torch.manual_seed(100)
    x = torch.randn(5, 8, dtype=torch.float32)
    sig, exp = _extract_sig_exp_batch(x, torch.float32)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            s_ref, e_ref = extract_significand_exponent(x[i, j].item(), torch.float32)
            assert abs(sig[i, j].item() - s_ref) < 1e-14, (
                f"sig mismatch at [{i},{j}]: got {sig[i,j].item()}, expected {s_ref}"
            )
            assert exp[i, j].item() == e_ref, (
                f"exp mismatch at [{i},{j}]: got {exp[i,j].item()}, expected {e_ref}"
            )


def test_extract_sig_exp_batch_zero():
    """_extract_sig_exp_batch must map zero to sig=0 with the canonical exponent -126."""
    x = torch.zeros(3, dtype=torch.float32)
    sig, exp = _extract_sig_exp_batch(x, torch.float32)
    assert torch.all(sig == 0.0), f"Expected all-zero sig, got {sig}"
    assert torch.all(exp == -126), f"Expected all -126 exp, got {exp}"


def test_extract_sig_exp_batch_subnormal_float16():
    """Values below float16 min_exponent (-14) must be subnormalized correctly."""
    # 2^-15 is below the float16 min_exponent of -14 → subnormal in float16 context.
    x = torch.tensor([2.0 ** -15], dtype=torch.float32)
    sig, exp = _extract_sig_exp_batch(x, torch.float16)
    s_ref, e_ref = extract_significand_exponent(2.0 ** -15, torch.float16)
    assert abs(sig[0].item() - s_ref) < 1e-14, (
        f"sig mismatch: got {sig[0].item()}, expected {s_ref}"
    )
    assert exp[0].item() == e_ref, f"exp mismatch: got {exp[0].item()}, expected {e_ref}"


def test_extract_sig_exp_batch_matches_scalar_float16():
    """_extract_sig_exp_batch must match extract_significand_exponent for float16 inputs."""
    torch.manual_seed(101)
    x = torch.randn(4, 6, dtype=torch.float16)
    sig, exp = _extract_sig_exp_batch(x, torch.float16)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            s_ref, e_ref = extract_significand_exponent(x[i, j], torch.float16)
            assert abs(sig[i, j].item() - s_ref) < 1e-14, (
                f"sig mismatch at [{i},{j}]: {sig[i,j].item()} vs {s_ref}"
            )
            assert exp[i, j].item() == e_ref, (
                f"exp mismatch at [{i},{j}]: {exp[i,j].item()} vs {e_ref}"
            )


def test_fused_sum_batch_matches_scalar():
    """_fused_sum_batch must match fused_sum row-by-row for random significands/exponents."""
    torch.manual_seed(102)
    n, m = 4, 8
    sig_t = torch.randn(n, m, dtype=torch.float64)
    exp_t = torch.randint(-20, 20, (n, m), dtype=torch.int32)
    n_frac = 24

    s_batch, e_batch = _fused_sum_batch(sig_t, exp_t, n_frac)

    for i in range(n):
        s_ref, e_ref = fused_sum(sig_t[i].tolist(), exp_t[i].tolist(), n_frac)
        assert abs(s_batch[i].item() - s_ref) < 1e-12, (
            f"sig_sum mismatch at row {i}: {s_batch[i].item()} vs {s_ref}"
        )
        assert e_batch[i].item() == e_ref, (
            f"exp mismatch at row {i}: {e_batch[i].item()} vs {e_ref}"
        )


# ---------------------------------------------------------------------------
# pairwise_dot: edge k sizes
# ---------------------------------------------------------------------------

def test_batch_pairwise_dot_k1():
    """k=1 (tree base case): must match the scalar for every row."""
    torch.manual_seed(110)
    n, k = 5, 1
    a = torch.randn(n, k, dtype=torch.float32)
    b = torch.randn(n, k, dtype=torch.float32)
    expected = torch.tensor(
        [pairwise_dot(a[i], b[i]) for i in range(n)], dtype=torch.float32
    )
    assert torch.equal(batch_pairwise_dot(a, b), expected)


def test_batch_pairwise_dot_k3():
    """k=3 (odd split 1+2): must match the scalar for every row."""
    torch.manual_seed(111)
    n, k = 5, 3
    a = torch.randn(n, k, dtype=torch.float32)
    b = torch.randn(n, k, dtype=torch.float32)
    expected = torch.tensor(
        [pairwise_dot(a[i], b[i]) for i in range(n)], dtype=torch.float32
    )
    assert torch.equal(batch_pairwise_dot(a, b), expected)


# ---------------------------------------------------------------------------
# fused_dot_add: NaN / Inf propagation
# ---------------------------------------------------------------------------

def _check_fused_tuples(result, expected, label=""):
    """Helper: assert (sig, exp) pairs match, handling NaN sig correctly."""
    for i, ((s_r, e_r), (s_e, e_e)) in enumerate(zip(result, expected)):
        if math.isnan(s_e):
            assert math.isnan(s_r), f"{label} row {i}: expected NaN sig, got {s_r}"
        elif math.isinf(s_e):
            assert math.isinf(s_r) and (s_r > 0) == (s_e > 0), (
                f"{label} row {i}: expected {s_e}, got {s_r}"
            )
        else:
            assert s_r == s_e and e_r == e_e, (
                f"{label} row {i}: expected ({s_e}, {e_e}), got ({s_r}, {e_r})"
            )


def test_batch_fused_dot_add_nan_c():
    """A NaN accumulator must propagate: returns (nan, 0) for that row."""
    n, k = 3, 4
    torch.manual_seed(120)
    a = torch.randn(n, k, dtype=torch.float16)
    b = torch.randn(n, k, dtype=torch.float16)
    c = torch.zeros(n, dtype=torch.float32)
    c[1] = float("nan")
    scale_a = torch.ones(n, dtype=torch.float32)
    scale_b = torch.ones(n, dtype=torch.float32)
    n_frac = 24

    expected = [fused_dot_add(a[i], b[i], c[i], n_frac, scale_a[i], scale_b[i]) for i in range(n)]
    result = batch_fused_dot_add(a, b, c, n_frac, scale_a, scale_b)
    _check_fused_tuples(result, expected, "nan_c")


def test_batch_fused_dot_add_inf_scale():
    """An Inf scale must propagate: returns (±inf, 0) for that row."""
    n, k = 3, 4
    torch.manual_seed(121)
    a = torch.randn(n, k, dtype=torch.float16)
    b = torch.randn(n, k, dtype=torch.float16)
    c = torch.zeros(n, dtype=torch.float32)
    scale_a = torch.ones(n, dtype=torch.float32)
    scale_a[0] = float("inf")
    scale_b = torch.ones(n, dtype=torch.float32)
    n_frac = 24

    expected = [fused_dot_add(a[i], b[i], c[i], n_frac, scale_a[i], scale_b[i]) for i in range(n)]
    result = batch_fused_dot_add(a, b, c, n_frac, scale_a, scale_b)
    _check_fused_tuples(result, expected, "inf_scale")


def test_batch_fused_dot_add_mixed_nan_normal():
    """Mixed batch (one NaN row, rest normal): torch.where must not corrupt normal rows."""
    n, k = 4, 8
    torch.manual_seed(122)
    a = torch.randn(n, k, dtype=torch.float16)
    b = torch.randn(n, k, dtype=torch.float16)
    c = torch.randn(n, dtype=torch.float32)
    c[2] = float("nan")
    scale_a = torch.ones(n, dtype=torch.float32)
    scale_b = torch.ones(n, dtype=torch.float32)
    n_frac = 24

    expected = [fused_dot_add(a[i], b[i], c[i], n_frac, scale_a[i], scale_b[i]) for i in range(n)]
    result = batch_fused_dot_add(a, b, c, n_frac, scale_a, scale_b)
    _check_fused_tuples(result, expected, "mixed_nan_normal")


# ---------------------------------------------------------------------------
# nv_fused_dot_add: f32_e8m13 output type and NaN / Inf propagation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,k", [(2, 8), (3, 16)])
def test_batch_nv_fused_dot_add_f32_e8m13(n, k):
    """f32_e8m13 output type (13 mantissa bits, RZ) must match scalar bit-for-bit."""
    a = make_fp16(n, k, seed=130)
    b = make_fp16(n, k, seed=131)
    c = torch.zeros(n, dtype=torch.float32)
    n_frac, output_type = 24, "f32_e8m13"

    expected = torch.stack(
        [nv_fused_dot_add(a[i], b[i], c[i], n_frac, output_type) for i in range(n)]
    )
    result = batch_nv_fused_dot_add(a, b, c, n_frac, output_type)

    assert result.dtype == torch.float32
    assert torch.equal(result.view(torch.int32), expected.view(torch.int32)), (
        f"f32_e8m13 mismatch:\nexpected={expected}\nresult={result}"
    )


def test_batch_nv_fused_dot_add_nan_output_f32():
    """NaN accumulator must produce the quiet-NaN bit pattern 0x7FFFFFFF (f32)."""
    n, k = 3, 4
    torch.manual_seed(140)
    a = torch.randn(n, k, dtype=torch.float16)
    b = torch.randn(n, k, dtype=torch.float16)
    c = torch.zeros(n, dtype=torch.float32)
    c[0] = float("nan")
    n_frac, output_type = 24, "f32"

    expected = torch.stack(
        [nv_fused_dot_add(a[i], b[i], c[i], n_frac, output_type) for i in range(n)]
    )
    result = batch_nv_fused_dot_add(a, b, c, n_frac, output_type)

    assert torch.equal(result.view(torch.int32), expected.view(torch.int32)), (
        f"NaN f32 mismatch:\n"
        f"expected={expected.view(torch.int32)}\nresult={result.view(torch.int32)}"
    )


def test_batch_nv_fused_dot_add_nan_output_f16():
    """NaN accumulator must produce the quiet-NaN bit pattern 0x7FFF (f16)."""
    n, k = 3, 4
    torch.manual_seed(141)
    a = torch.randn(n, k, dtype=torch.float16)
    b = torch.randn(n, k, dtype=torch.float16)
    c = torch.zeros(n, dtype=torch.float16)
    c[1] = float("nan")
    n_frac, output_type = 24, "f16"

    expected = torch.stack(
        [nv_fused_dot_add(a[i], b[i], c[i], n_frac, output_type) for i in range(n)]
    )
    result = batch_nv_fused_dot_add(a, b, c, n_frac, output_type)

    assert torch.equal(result.view(torch.int16), expected.view(torch.int16)), (
        f"NaN f16 mismatch:\n"
        f"expected={expected.view(torch.int16)}\nresult={result.view(torch.int16)}"
    )


def test_batch_nv_fused_dot_add_inf_propagation():
    """Inf accumulator must produce Inf output (not the NaN sentinel)."""
    n, k = 3, 4
    torch.manual_seed(142)
    a = torch.randn(n, k, dtype=torch.float16)
    b = torch.randn(n, k, dtype=torch.float16)
    c = torch.zeros(n, dtype=torch.float32)
    c[2] = float("inf")
    n_frac, output_type = 24, "f32"

    expected = torch.stack(
        [nv_fused_dot_add(a[i], b[i], c[i], n_frac, output_type) for i in range(n)]
    )
    result = batch_nv_fused_dot_add(a, b, c, n_frac, output_type)

    assert torch.equal(result.view(torch.int32), expected.view(torch.int32)), (
        f"Inf propagation mismatch:\nexpected={expected}\nresult={result}"
    )


def test_batch_nv_fused_dot_add_mixed_nan_normal():
    """Mixed batch (NaN and Inf rows alongside normal rows) must produce correct outputs."""
    n, k = 4, 8
    torch.manual_seed(143)
    a = torch.randn(n, k, dtype=torch.float16)
    b = torch.randn(n, k, dtype=torch.float16)
    c = torch.zeros(n, dtype=torch.float32)
    c[1] = float("nan")
    c[3] = float("inf")
    n_frac, output_type = 24, "f32"

    expected = torch.stack(
        [nv_fused_dot_add(a[i], b[i], c[i], n_frac, output_type) for i in range(n)]
    )
    result = batch_nv_fused_dot_add(a, b, c, n_frac, output_type)

    assert torch.equal(result.view(torch.int32), expected.view(torch.int32)), (
        f"mixed NaN/normal mismatch:\n"
        f"expected={expected.view(torch.int32)}\nresult={result.view(torch.int32)}"
    )


# ---------------------------------------------------------------------------
# nv_fused_dot_add_with_block_scale: NaN inputs and non-unit scales
# ---------------------------------------------------------------------------

def _make_fp4_block_inputs(n, k, seed):
    torch.manual_seed(seed)
    a_packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8)
    b_packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8)
    a = torch.stack([unpack_fp4_tensor(a_packed[i]) for i in range(n)])
    b = torch.stack([unpack_fp4_tensor(b_packed[i]) for i in range(n)])
    return a, b


def test_batch_nv_fused_dot_add_with_block_scale_nan_scale():
    """NaN in scale_a must produce the NaN sentinel 0x7FFFFFFF for that row."""
    n, k, n_frac = 3, 16, 25
    a, b = _make_fp4_block_inputs(n, k, seed=150)
    c = torch.zeros(n, dtype=torch.float32)
    num_blocks = k // 16
    scale_a = torch.ones(n, num_blocks, dtype=torch.float32)
    scale_b = torch.ones(n, num_blocks, dtype=torch.float32)
    scale_a[1, 0] = float("nan")

    expected = torch.stack(
        [nv_fused_dot_add_with_block_scale(a[i], b[i], c[i], scale_a[i], scale_b[i], n_frac)
         for i in range(n)]
    )
    result = batch_nv_fused_dot_add_with_block_scale(a, b, c, scale_a, scale_b, n_frac)

    assert torch.equal(result.view(torch.int32), expected.view(torch.int32)), (
        f"NaN scale mismatch:\n"
        f"expected={expected.view(torch.int32)}\nresult={result.view(torch.int32)}"
    )


def test_batch_nv_fused_dot_add_with_block_scale_nan_c():
    """NaN in c must produce the NaN sentinel 0x7FFFFFFF for that row."""
    n, k, n_frac = 3, 16, 25
    a, b = _make_fp4_block_inputs(n, k, seed=151)
    c = torch.zeros(n, dtype=torch.float32)
    c[2] = float("nan")
    num_blocks = k // 16
    scale_a = torch.ones(n, num_blocks, dtype=torch.float32)
    scale_b = torch.ones(n, num_blocks, dtype=torch.float32)

    expected = torch.stack(
        [nv_fused_dot_add_with_block_scale(a[i], b[i], c[i], scale_a[i], scale_b[i], n_frac)
         for i in range(n)]
    )
    result = batch_nv_fused_dot_add_with_block_scale(a, b, c, scale_a, scale_b, n_frac)

    assert torch.equal(result.view(torch.int32), expected.view(torch.int32))


def test_batch_nv_fused_dot_add_with_block_scale_nonunit_scales():
    """Random non-unit block scales must match the scalar implementation bit-for-bit."""
    n, k, n_frac = 4, 32, 25
    a, b = _make_fp4_block_inputs(n, k, seed=152)
    torch.manual_seed(152)
    c = torch.randn(n, dtype=torch.float32)
    num_blocks = k // 16
    scale_a = torch.abs(torch.randn(n, num_blocks, dtype=torch.float32)) + 0.1
    scale_b = torch.abs(torch.randn(n, num_blocks, dtype=torch.float32)) + 0.1

    expected = torch.stack(
        [nv_fused_dot_add_with_block_scale(a[i], b[i], c[i], scale_a[i], scale_b[i], n_frac)
         for i in range(n)]
    )
    result = batch_nv_fused_dot_add_with_block_scale(a, b, c, scale_a, scale_b, n_frac)

    assert torch.equal(result.view(torch.int32), expected.view(torch.int32)), (
        f"non-unit scale mismatch:\nexpected={expected}\nresult={result}"
    )


def test_batch_nv_fused_dot_add_with_block_scale_mixed_nan_normal():
    """Mixed batch (some NaN rows, some normal) must not corrupt normal rows."""
    n, k, n_frac = 4, 16, 25
    a, b = _make_fp4_block_inputs(n, k, seed=153)
    c = torch.zeros(n, dtype=torch.float32)
    num_blocks = k // 16
    scale_a = torch.ones(n, num_blocks, dtype=torch.float32)
    scale_b = torch.ones(n, num_blocks, dtype=torch.float32)
    scale_b[0, 0] = float("nan")   # row 0 → NaN sentinel
    c[3] = float("nan")            # row 3 → NaN sentinel via c

    expected = torch.stack(
        [nv_fused_dot_add_with_block_scale(a[i], b[i], c[i], scale_a[i], scale_b[i], n_frac)
         for i in range(n)]
    )
    result = batch_nv_fused_dot_add_with_block_scale(a, b, c, scale_a, scale_b, n_frac)

    assert torch.equal(result.view(torch.int32), expected.view(torch.int32)), (
        f"mixed NaN/normal mismatch:\n"
        f"expected={expected.view(torch.int32)}\nresult={result.view(torch.int32)}"
    )


# ---------------------------------------------------------------------------
# amd_fused_dot_rd_add: overflow paths (+inf, -inf, NaN, mixed)
# ---------------------------------------------------------------------------

def test_batch_amd_fused_dot_rd_add_positive_overflow():
    """Products exceeding 2^128 must produce +inf; normal rows must be unaffected."""
    flt_max = torch.finfo(torch.float32).max
    n, k = 3, 4
    a = torch.zeros(n, k, dtype=torch.float32)
    b = torch.zeros(n, k, dtype=torch.float32)
    # Row 0: overflow (+inf)
    a[0, 0] = flt_max
    b[0, 0] = flt_max
    # Row 1: normal
    a[1] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b[1] = torch.tensor([1.0, 1.0, 1.0, 1.0])
    # Row 2: normal (zeros → 0)
    c = torch.zeros(n, dtype=torch.float32)
    n_frac = 24

    expected = torch.tensor(
        [amd_fused_dot_rd_add(a[i], b[i], c[i], n_frac) for i in range(n)],
        dtype=torch.float64,
    )
    result = batch_amd_fused_dot_rd_add(a, b, c, n_frac)

    assert math.isinf(result[0].item()) and result[0].item() > 0, (
        f"Row 0 should be +inf, got {result[0].item()}"
    )
    assert torch.equal(result.view(torch.int64), expected.view(torch.int64))


def test_batch_amd_fused_dot_rd_add_negative_overflow():
    """Products below -2^128 must produce -inf; normal rows must be unaffected."""
    flt_max = torch.finfo(torch.float32).max
    n, k = 3, 4
    a = torch.zeros(n, k, dtype=torch.float32)
    b = torch.zeros(n, k, dtype=torch.float32)
    # Row 0: overflow (-inf)
    a[0, 0] = -flt_max
    b[0, 0] = flt_max
    # Row 1: normal
    a[1] = torch.tensor([0.5, -0.5, 0.5, -0.5])
    b[1] = torch.tensor([1.0, 1.0, 1.0, 1.0])
    c = torch.zeros(n, dtype=torch.float32)
    n_frac = 24

    expected = torch.tensor(
        [amd_fused_dot_rd_add(a[i], b[i], c[i], n_frac) for i in range(n)],
        dtype=torch.float64,
    )
    result = batch_amd_fused_dot_rd_add(a, b, c, n_frac)

    assert math.isinf(result[0].item()) and result[0].item() < 0, (
        f"Row 0 should be -inf, got {result[0].item()}"
    )
    assert torch.equal(result.view(torch.int64), expected.view(torch.int64))


def test_batch_amd_fused_dot_rd_add_both_overflow_nan():
    """Both +inf and -inf products in the same row must produce NaN."""
    flt_max = torch.finfo(torch.float32).max
    n, k = 2, 4
    a = torch.zeros(n, k, dtype=torch.float32)
    b = torch.zeros(n, k, dtype=torch.float32)
    # Row 0: +inf and -inf in the same row → NaN
    a[0, 0] = flt_max
    b[0, 0] = flt_max
    a[0, 1] = -flt_max
    b[0, 1] = flt_max
    # Row 1: normal
    a[1] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b[1] = torch.tensor([1.0, 1.0, 1.0, 1.0])
    c = torch.zeros(n, dtype=torch.float32)
    n_frac = 24

    expected = torch.tensor(
        [amd_fused_dot_rd_add(a[i], b[i], c[i], n_frac) for i in range(n)],
        dtype=torch.float64,
    )
    result = batch_amd_fused_dot_rd_add(a, b, c, n_frac)

    assert math.isnan(result[0].item()), f"Row 0 should be NaN, got {result[0].item()}"
    assert torch.equal(result.view(torch.int64), expected.view(torch.int64))


def test_batch_amd_fused_dot_rd_add_mixed_overflow_normal():
    """Multiple overflow rows alongside normal rows must each be handled independently."""
    flt_max = torch.finfo(torch.float32).max
    n, k = 4, 4
    torch.manual_seed(160)
    a = torch.randn(n, k, dtype=torch.float32)
    b = torch.randn(n, k, dtype=torch.float32)
    # Row 0: +inf overflow
    a[0, 0] = flt_max
    b[0, 0] = flt_max
    a[0, 1:] = 0.0
    b[0, 1:] = 0.0
    # Row 2: -inf overflow
    a[2, 0] = -flt_max
    b[2, 0] = flt_max
    a[2, 1:] = 0.0
    b[2, 1:] = 0.0
    c = torch.zeros(n, dtype=torch.float32)
    n_frac = 24

    expected = torch.tensor(
        [amd_fused_dot_rd_add(a[i], b[i], c[i], n_frac) for i in range(n)],
        dtype=torch.float64,
    )
    result = batch_amd_fused_dot_rd_add(a, b, c, n_frac)

    assert torch.equal(result.view(torch.int64), expected.view(torch.int64)), (
        f"mixed overflow mismatch:\nexpected={expected}\nresult={result}"
    )


@pytest.mark.parametrize("n,k", [(2, 16), (4, 16)])
def test_batch_amd_fused_dot_rd_add_fp8_nonzero_c(n, k):
    """is_fp8=True with nonzero accumulators must match scalar bit-for-bit."""
    torch.manual_seed(170)
    a = torch.randn(n, k, dtype=torch.float16)
    b = torch.randn(n, k, dtype=torch.float16)
    c = torch.randn(n, dtype=torch.float32)
    n_frac = 24

    expected = torch.tensor(
        [amd_fused_dot_rd_add(a[i], b[i], c[i], n_frac, is_fp8=True) for i in range(n)],
        dtype=torch.float64,
    )
    result = batch_amd_fused_dot_rd_add(a, b, c, n_frac, is_fp8=True)

    assert torch.equal(result.view(torch.int64), expected.view(torch.int64)), (
        f"fp8 nonzero-c mismatch (n={n}, k={k}):\nexpected={expected}\nresult={result}"
    )


def test_batch_amd_fused_dot_rd_add_fp8_c_below_window():
    """is_fp8=True: accumulator far below product sum must be dropped (sc_adj=0 branch)."""
    torch.manual_seed(171)
    n, k = 3, 16
    a = torch.randn(n, k, dtype=torch.float16)
    b = torch.randn(n, k, dtype=torch.float16)
    # c is extremely small so its exponent will be far below the combined product exponent,
    # triggering the `ec < max_e_all - 25` → sc_adj = 0 branch.
    c = torch.full((n,), 2.0 ** -60, dtype=torch.float32)
    n_frac = 24

    expected = torch.tensor(
        [amd_fused_dot_rd_add(a[i], b[i], c[i], n_frac, is_fp8=True) for i in range(n)],
        dtype=torch.float64,
    )
    result = batch_amd_fused_dot_rd_add(a, b, c, n_frac, is_fp8=True)

    assert torch.equal(result.view(torch.int64), expected.view(torch.int64)), (
        f"fp8 small-c mismatch:\nexpected={expected}\nresult={result}"
    )


# ---------------------------------------------------------------------------
# Non-batched wrappers in batch_arithmetic: equivalence with arithmetic.py
# ---------------------------------------------------------------------------

def test_wrapper_pairwise_dot():
    """batch_arithmetic.pairwise_dot wrapper must match arithmetic.pairwise_dot exactly."""
    import mmasim.simulator.batch_arithmetic as ba
    import mmasim.simulator.arithmetic as ar

    torch.manual_seed(180)
    a = torch.randn(8, dtype=torch.float32)
    b = torch.randn(8, dtype=torch.float32)
    assert ba.pairwise_dot(a, b) == ar.pairwise_dot(a, b)


def test_wrapper_fused_dot_add():
    """batch_arithmetic.fused_dot_add wrapper must match arithmetic.fused_dot_add exactly."""
    import mmasim.simulator.batch_arithmetic as ba
    import mmasim.simulator.arithmetic as ar

    torch.manual_seed(181)
    a = torch.randn(8, dtype=torch.float16)
    b = torch.randn(8, dtype=torch.float16)
    c = torch.tensor(0.5, dtype=torch.float32)
    scale_a = torch.tensor(2.0, dtype=torch.float32)
    scale_b = torch.tensor(0.5, dtype=torch.float32)
    r_ba = ba.fused_dot_add(a, b, c, 24, scale_a, scale_b)
    r_ar = ar.fused_dot_add(a, b, c, 24, scale_a, scale_b)
    assert r_ba == r_ar, f"fused_dot_add wrapper mismatch: {r_ba} vs {r_ar}"


def test_wrapper_nv_fused_dot_add_all_output_types():
    """batch_arithmetic.nv_fused_dot_add wrapper must match arithmetic version for all output types."""
    import mmasim.simulator.batch_arithmetic as ba
    import mmasim.simulator.arithmetic as ar

    torch.manual_seed(182)
    a = torch.randn(8, dtype=torch.float16)
    b = torch.randn(8, dtype=torch.float16)
    c = torch.tensor(0.0, dtype=torch.float32)

    for output_type in ("f16", "f32", "f32_e8m13"):
        r_ba = ba.nv_fused_dot_add(a, b, c, 24, output_type)
        r_ar = ar.nv_fused_dot_add(a, b, c, 24, output_type)
        int_dtype = torch.int16 if output_type == "f16" else torch.int32
        assert torch.equal(r_ba.view(int_dtype), r_ar.view(int_dtype)), (
            f"nv_fused_dot_add wrapper mismatch for output_type={output_type}"
        )


def test_wrapper_amd_fused_dot_rd_add():
    """batch_arithmetic.amd_fused_dot_rd_add wrapper must match arithmetic version."""
    import mmasim.simulator.batch_arithmetic as ba
    import mmasim.simulator.arithmetic as ar

    torch.manual_seed(183)
    a = torch.randn(8, dtype=torch.float16)
    b = torch.randn(8, dtype=torch.float16)
    c = torch.tensor(1.0, dtype=torch.float32)
    for is_fp8 in (False, True):
        r_ba = ba.amd_fused_dot_rd_add(a, b, c, 24, is_fp8)
        r_ar = ar.amd_fused_dot_rd_add(a, b, c, 24, is_fp8)
        assert r_ba == r_ar or (math.isnan(r_ba) and math.isnan(r_ar)), (
            f"amd_fused_dot_rd_add wrapper mismatch for is_fp8={is_fp8}: {r_ba} vs {r_ar}"
        )
