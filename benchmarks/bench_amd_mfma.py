"""Benchmarks for the AMD mfma simulator.

Run with:
    pytest benchmarks/bench_amd_mfma.py --benchmark-only

Each benchmark instantiates the simulator for a representative qualifier and
measures the time to execute one full matrix-multiply-accumulate operation on
random inputs.
"""

import pytest
import torch

from mmasim.simulator.amd import mfma


# ---------------------------------------------------------------------------
# Representative (arch, qualifier) pairs – one or two per ISA generation
# ---------------------------------------------------------------------------

CDNA1_CASES = [
    ("CDNA1", "f32_32x32x2f32"),    # f32
    ("CDNA1", "f32_32x32x8f16"),    # f16
    ("CDNA1", "f32_32x32x4bf16"),   # bf16
]

CDNA2_CASES = [
    ("CDNA2", "f64_16x16x4f64"),        # f64
    ("CDNA2", "f32_32x32x8bf16_1k"),    # bf16 (1k variant)
]

CDNA3_CASES = [
    ("CDNA3", "f32_32x32x2_f32"),       # f32
    ("CDNA3", "f32_32x32x8_f16"),       # f16
    ("CDNA3", "f32_16x16x32_fp8_fp8"),  # fp8
]

ALL_CASES = CDNA1_CASES + CDNA2_CASES + CDNA3_CASES


def _make_inputs(instr: mfma) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return random (A, B, C) tensors matching the instruction's shapes and dtypes."""
    torch.manual_seed(0)
    m, n, k = instr.m, instr.n, instr.k
    if instr.a_type in (torch.float8_e4m3fnuz, torch.float8_e5m2fnuz):
        # fp8 types: generate float32 values in [-1, 1] and cast
        A = torch.rand(m, k, dtype=torch.float32).to(instr.a_type)
        B = torch.rand(k, n, dtype=torch.float32).to(instr.b_type)
    else:
        A = torch.randn(m, k, dtype=instr.a_type)
        B = torch.randn(k, n, dtype=instr.b_type)
    C = torch.randn(m, n, dtype=instr.c_type)
    return A, B, C


@pytest.mark.parametrize("arch,qualifier", ALL_CASES)
def test_bench_mfma(benchmark, arch, qualifier):
    """Benchmark one call to the mfma simulator."""
    instr = mfma(arch, qualifier)
    A, B, C = _make_inputs(instr)
    benchmark(instr, A, B, C)
