"""Unit tests for the utility functions in ``mmasim.simulator.utils``.

Covers ``fma``, ``truncate_to_tf32``, ``unpack_fp4_tensor``,
``flush_denormal``, and ``dtype_min_exponent``.
"""

import pytest
import torch

from mmasim.simulator.utils import (
    dtype_min_exponent,
    flush_denormal,
    fma,
    truncate_to_tf32,
    unpack_fp4_tensor,
)


# ---------------------------------------------------------------------------
# fma
# ---------------------------------------------------------------------------

def test_fma_f32_basic():
    """fma(a, b, c) == a*b+c for ordinary float32 values."""
    a = torch.tensor(2.0, dtype=torch.float32)
    b = torch.tensor(3.0, dtype=torch.float32)
    c = torch.tensor(1.0, dtype=torch.float32)
    result = fma(a, b, c)
    assert result.dtype == torch.float32
    assert result.item() == 7.0


def test_fma_f32_zero_addend():
    """fma(a, b, 0) == a*b for float32."""
    a = torch.tensor(1.5, dtype=torch.float32)
    b = torch.tensor(4.0, dtype=torch.float32)
    c = torch.tensor(0.0, dtype=torch.float32)
    result = fma(a, b, c)
    assert result.dtype == torch.float32
    assert result.item() == 6.0


def test_fma_f64_basic():
    """fma(a, b, c) == a*b+c for ordinary float64 values."""
    a = torch.tensor(2.0, dtype=torch.float64)
    b = torch.tensor(3.0, dtype=torch.float64)
    c = torch.tensor(1.0, dtype=torch.float64)
    result = fma(a, b, c)
    assert result.dtype == torch.float64
    assert result.item() == 7.0


def test_fma_f32_negative():
    """fma with negative values."""
    a = torch.tensor(-2.0, dtype=torch.float32)
    b = torch.tensor(3.0, dtype=torch.float32)
    c = torch.tensor(1.0, dtype=torch.float32)
    result = fma(a, b, c)
    assert result.item() == -5.0


def test_fma_unsupported_dtype():
    """fma raises ValueError for unsupported dtypes."""
    a = torch.tensor(1.0, dtype=torch.float16)
    b = torch.tensor(1.0, dtype=torch.float16)
    c = torch.tensor(1.0, dtype=torch.float16)
    with pytest.raises(ValueError, match="Unsupported dtype"):
        fma(a, b, c)


# ---------------------------------------------------------------------------
# truncate_to_tf32
# ---------------------------------------------------------------------------

def test_truncate_to_tf32_exact_value():
    """A value already representable in TF32 is unchanged."""
    # 1.0 has zero mantissa bits, so no truncation occurs.
    x = torch.tensor(1.0, dtype=torch.float32)
    result = truncate_to_tf32(x)
    assert result.item() == 1.0


def test_truncate_to_tf32_clears_lower_bits():
    """The lower 13 bits of the float32 mantissa are zeroed."""
    # Build a float32 value with all mantissa bits set:
    # sign=0, exponent=127 (value 1.0 range), mantissa=all ones
    raw = torch.tensor(0x3FFF_FFFF, dtype=torch.int32).view(torch.float32)
    result = truncate_to_tf32(raw)
    result_bits = result.view(torch.int32).item()
    # The lower 13 mantissa bits must be zero
    assert (result_bits & 0x1FFF) == 0


def test_truncate_to_tf32_preserves_sign_and_exponent():
    """sign and exponent bits survive truncation."""
    # Use -2.0: sign=1, exponent=128, mantissa=0 — already TF32-exact.
    x = torch.tensor(-2.0, dtype=torch.float32)
    result = truncate_to_tf32(x)
    assert result.item() == -2.0


def test_truncate_to_tf32_requires_float32():
    """truncate_to_tf32 asserts on non-float32 input."""
    x = torch.tensor(1.0, dtype=torch.float64)
    with pytest.raises(AssertionError):
        truncate_to_tf32(x)


# ---------------------------------------------------------------------------
# unpack_fp4_tensor
# ---------------------------------------------------------------------------

# All 16 positive FP4 codes and their expected float values.
_FP4_POSITIVE = [
    (0b0000, 0.0),
    (0b0001, 0.5),
    (0b0010, 1.0),
    (0b0011, 1.5),
    (0b0100, 2.0),
    (0b0101, 3.0),
    (0b0110, 4.0),
    (0b0111, 6.0),
]

# All 8 negative FP4 codes (sign bit set) and their expected float values.
_FP4_NEGATIVE = [(code | 0b1000, -val) for code, val in _FP4_POSITIVE]


@pytest.mark.parametrize("code,expected", _FP4_POSITIVE)
def test_unpack_fp4_tensor_positive_values(code, expected):
    """Low nibble positive FP4 codes decode correctly."""
    # Pack the code into the low nibble of a single byte; high nibble = 0 (→ 0.0).
    packed = torch.tensor([code & 0xFF], dtype=torch.uint8)
    result = unpack_fp4_tensor(packed)
    assert result.numel() == 2
    assert result[0].item() == expected, f"code={code:#06b}: expected {expected}, got {result[0].item()}"
    assert result[1].item() == 0.0  # high nibble is 0b0000


@pytest.mark.parametrize("code,expected", _FP4_NEGATIVE)
def test_unpack_fp4_tensor_negative_values(code, expected):
    """Low nibble negative FP4 codes decode correctly."""
    packed = torch.tensor([code & 0xFF], dtype=torch.uint8)
    result = unpack_fp4_tensor(packed)
    assert result[0].item() == expected, f"code={code:#06b}: expected {expected}, got {result[0].item()}"


def test_unpack_fp4_tensor_high_nibble():
    """High nibble is decoded correctly."""
    # Pack 0b0110 (4.0) into the high nibble, 0 in the low nibble.
    packed = torch.tensor([0b0110_0000], dtype=torch.uint8)
    result = unpack_fp4_tensor(packed)
    assert result[0].item() == 0.0   # low nibble
    assert result[1].item() == 4.0   # high nibble


def test_unpack_fp4_tensor_multi_byte():
    """Multi-byte packed tensor produces 2*n values."""
    n = 4
    packed = torch.zeros(n, dtype=torch.uint8)
    result = unpack_fp4_tensor(packed)
    assert result.numel() == 2 * n
    assert torch.all(result == 0.0)


def test_unpack_fp4_tensor_output_dtype():
    """Output dtype is always float32."""
    packed = torch.tensor([0x12], dtype=torch.uint8)
    result = unpack_fp4_tensor(packed)
    assert result.dtype == torch.float32


# ---------------------------------------------------------------------------
# flush_denormal
# ---------------------------------------------------------------------------

def test_flush_denormal_f32_denormal_becomes_zero():
    """Denormal float32 values (abs < 2^-126) are flushed to zero."""
    denormal = torch.tensor(2.0**-127, dtype=torch.float32)
    result = flush_denormal(denormal.clone())
    assert result.item() == 0.0


def test_flush_denormal_f32_normal_unchanged():
    """Normal float32 values at or above the minimum exponent are not flushed."""
    normal = torch.tensor(2.0**-126, dtype=torch.float32)
    result = flush_denormal(normal.clone())
    assert result.item() == 2.0**-126


def test_flush_denormal_f32_keep_sign_positive():
    """keep_sign=True flushes positive denormal to +0."""
    denormal = torch.tensor(2.0**-127, dtype=torch.float32)
    result = flush_denormal(denormal.clone(), keep_sign=True)
    # +denormal * 0.0 → +0.0 in IEEE 754
    assert result.item() == 0.0
    assert not result.item() < 0.0


def test_flush_denormal_f32_keep_sign_negative():
    """keep_sign=True flushes negative denormal to -0."""
    denormal = torch.tensor(-2.0**-127, dtype=torch.float32)
    result = flush_denormal(denormal.clone(), keep_sign=True)
    # -denormal * 0.0 → -0.0: value is 0 but sign bit is set
    bits = result.view(torch.int32).item()
    assert result.item() == 0.0
    assert bits < 0  # sign bit set → negative zero


def test_flush_denormal_f32_keep_sign_false_negative():
    """keep_sign=False replaces negative denormal with +0 (not -0)."""
    denormal = torch.tensor(-2.0**-127, dtype=torch.float32)
    result = flush_denormal(denormal.clone(), keep_sign=False)
    bits = result.view(torch.int32).item()
    assert result.item() == 0.0
    assert bits == 0  # positive zero, sign bit clear


def test_flush_denormal_f16_denormal_becomes_zero():
    """Denormal float16 values (abs < 2^-14) are flushed to zero."""
    denormal = torch.tensor(2.0**-15, dtype=torch.float16)
    result = flush_denormal(denormal.clone())
    assert result.item() == 0.0


def test_flush_denormal_f16_min_normal_unchanged():
    """The smallest normal float16 value (2^-14) is not flushed."""
    min_normal = torch.tensor(2.0**-14, dtype=torch.float16)
    result = flush_denormal(min_normal.clone())
    assert result.item() == 2.0**-14


def test_flush_denormal_vector():
    """flush_denormal works element-wise on a vector."""
    x = torch.tensor(
        [2.0**-127, 1.0, -2.0**-127, -1.0], dtype=torch.float32
    )
    result = flush_denormal(x.clone())
    assert result[0].item() == 0.0    # denormal flushed
    assert result[1].item() == 1.0    # normal untouched
    assert result[2].item() == 0.0    # negative denormal flushed
    assert result[3].item() == -1.0   # normal untouched


# ---------------------------------------------------------------------------
# dtype_min_exponent
# ---------------------------------------------------------------------------

_EXPECTED_MIN_EXPONENTS = [
    (torch.float64,      -1022),
    (torch.float32,       -126),
    (torch.float16,        -14),
    (torch.bfloat16,      -126),
    (torch.float8_e8m0fnu,-127),
    (torch.float8_e5m2,    -14),
    (torch.float8_e4m3fn,   -6),
    (torch.float8_e5m2fnuz,-15),
    (torch.float8_e4m3fnuz, -7),
]


@pytest.mark.parametrize("dtype,expected", _EXPECTED_MIN_EXPONENTS)
def test_dtype_min_exponent_values(dtype, expected):
    """dtype_min_exponent contains the correct minimum exponent for each dtype."""
    assert dtype in dtype_min_exponent, f"{dtype} not in dtype_min_exponent"
    assert dtype_min_exponent[dtype] == expected, (
        f"dtype_min_exponent[{dtype}]: expected {expected}, "
        f"got {dtype_min_exponent[dtype]}"
    )


def test_dtype_min_exponent_completeness():
    """dtype_min_exponent covers all expected dtypes."""
    expected_dtypes = {dtype for dtype, _ in _EXPECTED_MIN_EXPONENTS}
    assert expected_dtypes.issubset(set(dtype_min_exponent.keys()))
