"""Unit tests verifying bit-accurate equivalence between the scalar dot product
functions in ``mmasim.simulator.arithmetic`` and the batched counterparts in
``mmasim.simulator.batch_arithmetic``.

Each test runs the scalar function on every row of an n×k matrix and collects
the results, then runs the batch function on the whole matrix, and asserts
exact (bit-level) equality between the two outputs.
"""

import pytest
import torch

from mmasim.simulator.arithmetic import (
    pairwise_dot,
    fused_dot_add,
    nv_fused_dot_add,
    nv_fused_dot_add_with_block_scale,
    amd_fused_dot_rd_add,
)
from mmasim.simulator.batch_arithmetic import (
    batch_pairwise_dot,
    batch_fused_dot_add,
    batch_nv_fused_dot_add,
    batch_nv_fused_dot_add_with_block_scale,
    batch_amd_fused_dot_rd_add,
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
