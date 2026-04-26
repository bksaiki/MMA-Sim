"""Benchmarks for the NVIDIA PTX MMA simulator.

Run with:
    pytest benchmarks/bench_nv_ptx.py --benchmark-only

Covers mma, wgmma, tcgen05mma, mma_block_scale, and tcgen05mma_block_scale.
Each benchmark instantiates the simulator for a representative qualifier and
measures the time to execute one full matrix-multiply-accumulate operation on
random inputs.
"""

import pytest
import torch

from mmasim.simulator.nv_ptx import (
    mma,
    mma_block_scale,
    wgmma,
    tcgen05mma,
    tcgen05mma_block_scale,
)


# ---------------------------------------------------------------------------
# Representative (arch, qualifier) pairs
# ---------------------------------------------------------------------------

MMA_CASES = [
    # Volta – f16
    ("Volta", "m8n8k4.f32.f16.f16.f32"),
    # Turing – f16
    ("Turing", "m16n8k8.f32.f16.f16.f32"),
    # Ampere – tf32, f16, bf16
    ("Ampere", "m16n8k8.f32.tf32.tf32.f32"),
    ("Ampere", "m16n8k16.f32.f16.f16.f32"),
    ("Ampere", "m16n8k16.f32.bf16.bf16.f32"),
    # Ada Lovelace – fp8
    ("Ada Lovelace", "m16n8k32.f32.e4m3.e4m3.f32"),
    # Hopper – f64
    ("Hopper", "m16n8k16.f64.f64.f64.f64"),
]

MMA_BLOCK_SCALE_CASES = [
    # RTX Blackwell – mxf8f6f4
    ("RTX Blackwell", "m16n8k32.block32.f32.e5m2.e5m2.f32.ue8m0"),
    # RTX Blackwell – mxf4nvf4
    ("RTX Blackwell", "m16n8k64.block32.f32.e2m1.e2m1.f32.ue8m0"),
]

WGMMA_CASES = [
    # Hopper – tf32, f16, bf16, fp8
    ("Hopper", "m64n64k8.f32.tf32.tf32"),
    ("Hopper", "m64n64k16.f32.f16.f16"),
    ("Hopper", "m64n64k16.f32.bf16.bf16"),
    ("Hopper", "m64n64k32.f32.e4m3.e4m3"),
]

TCGEN05MMA_CASES = [
    # Blackwell – tf32, f16, bf16, fp8
    ("Blackwell", "m64n64k8.f32.tf32.tf32"),
    ("Blackwell", "m64n64k16.f32.f16.f16"),
    ("Blackwell", "m64n64k16.f32.bf16.bf16"),
    ("Blackwell", "m64n64k32.f32.e4m3.e4m3"),
]

TCGEN05MMA_BLOCK_SCALE_CASES = [
    # Blackwell – mxf8f6f4
    ("Blackwell", "m128n64k32.block32.f32.e5m2.e5m2.ue8m0"),
    # Blackwell – mxf4nvf4
    ("Blackwell", "m128n64k64.block32.f32.e2m1.e2m1.ue8m0"),
]


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def _rand(shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    """Return a random tensor of the given shape and dtype."""
    if dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2fnuz,
        torch.float8_e8m0fnu,
    ):
        return torch.rand(shape, dtype=torch.float32).to(dtype)
    if dtype == torch.uint8:
        # fp4 packed: each byte holds two fp4 values
        return torch.randint(0, 256, shape, dtype=torch.uint8)
    return torch.randn(shape, dtype=dtype)


def _make_mma_inputs(
    instr: mma,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    m, n, k = instr.m, instr.n, instr.k
    A = _rand((m, k), instr.a_type)
    B = _rand((k, n), instr.b_type)
    C = torch.randn(m, n, dtype=instr.c_type)
    return A, B, C


def _make_block_scale_inputs(
    instr,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    m, n, k = instr.m, instr.n, instr.k
    packing = instr.packing
    block_size = instr.block_size
    A = _rand((m, k // packing), instr.a_type)
    B = _rand((k // packing, n), instr.b_type)
    C = torch.randn(m, n, dtype=instr.c_type)
    scale_A = _rand((m, k // block_size), instr.s_type)
    scale_B = _rand((k // block_size, n), instr.s_type)
    return A, B, C, scale_A, scale_B


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch,qualifier", MMA_CASES)
def test_bench_mma(benchmark, arch, qualifier):
    """Benchmark one call to the mma simulator."""
    instr = mma(arch, qualifier)
    A, B, C = _make_mma_inputs(instr)
    benchmark(instr, A, B, C)


@pytest.mark.parametrize("arch,qualifier", MMA_BLOCK_SCALE_CASES)
def test_bench_mma_block_scale(benchmark, arch, qualifier):
    """Benchmark one call to the mma_block_scale simulator."""
    instr = mma_block_scale(arch, qualifier)
    A, B, C, scale_A, scale_B = _make_block_scale_inputs(instr)
    benchmark(instr, A, B, C, scale_A, scale_B)


@pytest.mark.parametrize("arch,qualifier", WGMMA_CASES)
def test_bench_wgmma(benchmark, arch, qualifier):
    """Benchmark one call to the wgmma simulator."""
    instr = wgmma(arch, qualifier)
    A, B, C = _make_mma_inputs(instr)
    benchmark(instr, A, B, C)


@pytest.mark.parametrize("arch,qualifier", TCGEN05MMA_CASES)
def test_bench_tcgen05mma(benchmark, arch, qualifier):
    """Benchmark one call to the tcgen05mma simulator."""
    instr = tcgen05mma(arch, qualifier)
    A, B, C = _make_mma_inputs(instr)
    benchmark(instr, A, B, C)


@pytest.mark.parametrize("arch,qualifier", TCGEN05MMA_BLOCK_SCALE_CASES)
def test_bench_tcgen05mma_block_scale(benchmark, arch, qualifier):
    """Benchmark one call to the tcgen05mma_block_scale simulator."""
    instr = tcgen05mma_block_scale(arch, qualifier)
    A, B, C, scale_A, scale_B = _make_block_scale_inputs(instr)
    benchmark(instr, A, B, C, scale_A, scale_B)
