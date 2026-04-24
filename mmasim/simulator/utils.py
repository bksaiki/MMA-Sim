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
    x = x >> 13 << 13  # truncate to tf32
    return x.view(torch.float32)


def unpack_fp4_tensor(packed: torch.Tensor) -> torch.Tensor:
    n = packed.numel()
    low = packed & 0x0F
    high = packed >> 4
    unpacked = torch.zeros(n * 2, dtype=torch.float32)
    decoding = {
        0b0000: 0.0,
        0b0001: 0.5,
        0b0010: 1.0,
        0b0011: 1.5,
        0b0100: 2.0,
        0b0101: 3.0,
        0b0110: 4.0,
        0b0111: 6.0,
    }
    decoding |= {x + 0b1000: -y for x, y in decoding.items()}
    for i in range(n):
        unpacked[i * 2] = decoding[int(low[i].item())]
        unpacked[i * 2 + 1] = decoding[int(high[i].item())]
    return unpacked


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
