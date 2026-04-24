import ctypes

import torch

libm = ctypes.CDLL("libm.so.6")
libm.fmaf.argtypes = [ctypes.c_float] * 3
libm.fmaf.restype = ctypes.c_float
libm.fma.argtypes = [ctypes.c_double] * 3
libm.fma.restype = ctypes.c_double


def fma(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    assert a.dtype == b.dtype == c.dtype
    if a.dtype == torch.float32:
        res = libm.fmaf(a.item(), b.item(), c.item())
        return torch.tensor(res, dtype=torch.float32)
    elif a.dtype == torch.float64:
        res = libm.fma(a.item(), b.item(), c.item())
        return torch.tensor(res, dtype=torch.float64)
    else:
        raise ValueError(f"Unsupported dtype: {a.dtype}")


def truncate_to_tf32(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float32
    x = x.view(torch.int32)  # uint32 operations are not supported by pytorch
    x = x & ~0x1FFF  # truncate to tf32 by masking lower 13 bits
    return x.view(torch.float32)


# Global decoding table for FP4 values.  Index == 4-bit FP4 code (0–15).
# Positive codes occupy indices 0–7; negative (sign bit set) occupy 8–15.
_FP4_DECODE = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)
_FP4_DECODE_TENSOR = torch.tensor(_FP4_DECODE, dtype=torch.float32)


def unpack_fp4_tensor(packed: torch.Tensor) -> torch.Tensor:
    low = (packed & 0x0F).long()
    high = (packed >> 4).long()
    # Interleave low/high nibbles: [low[0], high[0], low[1], high[1], ...]
    indices = torch.stack([low, high], dim=1).flatten()
    return _FP4_DECODE_TENSOR[indices]


dtype_min_exponent = {
    torch.float64: -1022,
    torch.float32: -126,
    torch.float16: -14,
    torch.bfloat16: -126,
    torch.float8_e8m0fnu: -127,
    torch.float8_e5m2: -14,
    torch.float8_e4m3fn: -6,
    torch.float8_e5m2fnuz: -15,
    torch.float8_e4m3fnuz: -7,
}


def flush_denormal(x: torch.Tensor, keep_sign: bool = False) -> torch.Tensor:
    min_exponent = dtype_min_exponent[x.dtype]
    if keep_sign:
        x[x.abs() < 2.0**min_exponent] *= 0.0
    else:
        x[x.abs() < 2.0**min_exponent] = 0.0
    return x
