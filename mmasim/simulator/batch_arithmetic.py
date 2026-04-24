import torch

from .utils import dtype_min_exponent


# ---------------------------------------------------------------------------
# Private helpers – vectorized building blocks
# ---------------------------------------------------------------------------

def _pairwise_sum_batch(t: torch.Tensor) -> torch.Tensor:
    """Tree-structured pairwise reduction over dim=1, matching pairwise_dot's recursion.

    Performs the exact same binary-tree split as the scalar pairwise_dot (split at
    m = k // 2, recurse on each half, then add) but operates on all n rows at once.
    Because every addition is identical to the scalar fmaf(left, 1.0, right) = left+right,
    the result is bit-accurate with the scalar for every row.

    Args:
        t: (n, k) float32 tensor.

    Returns:
        (n,) float32 tensor.
    """
    if t.shape[1] == 1:
        return t[:, 0]
    m = t.shape[1] // 2
    left = _pairwise_sum_batch(t[:, :m])
    right = _pairwise_sum_batch(t[:, m:])
    return left + right


def _extract_sig_exp_batch(
    x: torch.Tensor,
    dtype_for_min_exp: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized extract_significand_exponent for every element of x.

    Mirrors arithmetic.extract_significand_exponent exactly, operating on the
    whole tensor at once via torch.frexp instead of a Python loop.

    Args:
        x: Float tensor of any shape.
        dtype_for_min_exp: Dtype whose min_exponent is used for subnormal detection
            (should be the logical dtype of x's values, e.g. torch.float16 for fp16
            inputs even after they have been converted to float64).

    Returns:
        significands: float64 tensor, same shape as x.
        exponents:    int32  tensor, same shape as x.
    """
    min_exp = dtype_min_exponent[dtype_for_min_exp]
    x_f64 = x.double()
    # torch.frexp gives mantissa in [0.5, 1) and integer exponent, matching math.frexp
    mantissa, exp = torch.frexp(x_f64)
    sig = mantissa * 2.0   # renormalize to [1, 2)
    exp = exp - 1

    # Subnormal: value is smaller than 2^min_exp in the target dtype
    is_subnormal = exp < min_exp
    exp_diff = (exp - min_exp).to(torch.float64)
    sig = torch.where(is_subnormal, sig * torch.pow(2.0, exp_diff), sig)
    exp = torch.where(is_subnormal, torch.full_like(exp, min_exp), exp)

    # Zero: keep sig=0 but set the canonical exponent used by the scalar helper
    exp = torch.where(sig == 0.0, torch.full_like(exp, -126), exp)

    return sig, exp


def _fused_sum_batch(
    sig: torch.Tensor,
    exp: torch.Tensor,
    n_frac: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized fused_sum over the last dimension.

    Mirrors arithmetic.fused_sum exactly: aligns all significands to the row's
    maximum exponent, truncates each to n_frac fractional bits, then sums.

    Args:
        sig: (n, m) float64 significands.
        exp: (n, m) int32  exponents.
        n_frac: Number of fractional bits for the fixed-point accumulator.

    Returns:
        sig_sums: (n,) float64 – sum of truncated significands.
        max_exps: (n,) int32  – per-row maximum exponent.
    """
    max_exp = exp.max(dim=1, keepdim=True).values          # (n, 1) int32
    diff = (n_frac + exp - max_exp).to(torch.float64)      # (n, m) float64
    rounded = torch.trunc(sig * torch.pow(2.0, diff)) / (2.0 ** n_frac)
    sig_sums = rounded.sum(dim=1)                          # (n,) float64
    return sig_sums, max_exp.squeeze(1)


# ---------------------------------------------------------------------------
# Batch implementations
# ---------------------------------------------------------------------------

def batch_pairwise_dot(
    a: torch.Tensor, b: torch.Tensor, flush_denormal: bool = False
) -> torch.Tensor:
    """Batched pairwise dot product over n k-length vectors.

    Internally vectorized: computes all n dot products simultaneously using a
    batched tree reduction over the k dimension, with no per-row Python loop.

    Args:
        a: Tensor of shape (n, k), dtype float32.
        b: Tensor of shape (n, k), dtype float32.
        flush_denormal: Whether to flush denormal results to zero.

    Returns:
        Tensor of shape (n,) containing the dot product for each row pair.
    """
    assert a.dtype == b.dtype == torch.float32
    assert a.dim() == 2 and b.dim() == 2
    assert a.shape == b.shape
    # Elementwise products: fmaf(a[i,j], b[i,j], 0) == a[i,j]*b[i,j] in float32
    ab = a * b
    result = _pairwise_sum_batch(ab)
    if flush_denormal:
        result = result.clone()
        result[result.abs() < 2.0 ** -126] *= 0.0
    return result


def batch_fused_dot_add(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    n_fractional_bits: int,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> list[tuple[float, int]]:
    """Batched fused dot-add over n k-length vectors.

    Internally vectorized: extracts significand/exponent for all (n, k) elements
    at once and runs a batched fused_sum, with no per-row Python loop over k.

    Args:
        a: Tensor of shape (n, k).
        b: Tensor of shape (n, k).
        c: Tensor of shape (n,) – one accumulator per row.
        n_fractional_bits: Number of fractional bits for the fused sum.
        scale_a: Tensor of shape (n,) – one scale per row for a.
        scale_b: Tensor of shape (n,) – one scale per row for b.

    Returns:
        List of n (significand, exponent) tuples, one per row.
    """
    assert a.dim() == 2 and b.dim() == 2
    assert a.shape == b.shape
    n = a.shape[0]

    # NaN / inf check (vectorized over all rows at once)
    fp64_sum = (
        a.double() * b.double()
        * scale_a.double().unsqueeze(1)
        * scale_b.double().unsqueeze(1)
    ).sum(dim=1) + c.double()                                         # (n,)
    nan_or_inf = torch.isnan(fp64_sum) | torch.isinf(fp64_sum)       # (n,) bool

    # Extract sig/exp for every element of a, b (n, k) and the scalars (n,)
    sa, ea = _extract_sig_exp_batch(a, a.dtype)                       # (n, k)
    sb, eb = _extract_sig_exp_batch(b, b.dtype)                       # (n, k)
    _, esa = _extract_sig_exp_batch(scale_a, scale_a.dtype)           # (n,)
    _, esb = _extract_sig_exp_batch(scale_b, scale_b.dtype)           # (n,)
    sc, ec = _extract_sig_exp_batch(c, c.dtype)                       # (n,)

    prod_sig = sa * sb                                                 # (n, k)
    prod_exp = ea + eb + esa.unsqueeze(1) + esb.unsqueeze(1)          # (n, k)

    # Prepend the accumulator c as the first column
    all_sig = torch.cat([sc.unsqueeze(1), prod_sig], dim=1)           # (n, k+1)
    all_exp = torch.cat([ec.unsqueeze(1), prod_exp], dim=1)           # (n, k+1)

    sig_sums, max_exps = _fused_sum_batch(all_sig, all_exp, n_fractional_bits)

    # Assemble output – only a thin O(n) loop, no heavy math
    results: list[tuple[float, int]] = []
    for i in range(n):
        if nan_or_inf[i]:
            results.append((fp64_sum[i].item(), 0))
        else:
            results.append((sig_sums[i].item(), max_exps[i].item()))
    return results


def batch_nv_fused_dot_add(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    n_fractional_bits: int,
    output_type: str,
    scale_a: torch.Tensor | None = None,
    scale_b: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batched NV fused dot-add over n k-length vectors.

    Internally vectorized: reuses _extract_sig_exp_batch / _fused_sum_batch for
    the heavy computation and applies the NV-specific output normalization with
    torch.where, with no per-row Python loop.

    Args:
        a: Tensor of shape (n, k).
        b: Tensor of shape (n, k).
        c: Tensor of shape (n,) – one accumulator per row.
        n_fractional_bits: Number of fractional bits for the fused sum.
        output_type: One of "f16", "f32", or "f32_e8m13".
        scale_a: Optional tensor of shape (n,) – one scale per row for a.
        scale_b: Optional tensor of shape (n,) – one scale per row for b.

    Returns:
        Tensor of shape (n,) containing one output element per row.
    """
    assert a.dim() == 2 and b.dim() == 2
    assert a.shape == b.shape
    n = a.shape[0]

    if scale_a is None or scale_b is None:
        scale_a = torch.ones(n, dtype=torch.float32)
        scale_b = torch.ones(n, dtype=torch.float32)

    fp64_sum = (
        a.double() * b.double()
        * scale_a.double().unsqueeze(1)
        * scale_b.double().unsqueeze(1)
    ).sum(dim=1) + c.double()
    nan_or_inf = torch.isnan(fp64_sum) | torch.isinf(fp64_sum)

    sa, ea = _extract_sig_exp_batch(a, a.dtype)
    sb, eb = _extract_sig_exp_batch(b, b.dtype)
    _, esa = _extract_sig_exp_batch(scale_a, scale_a.dtype)
    _, esb = _extract_sig_exp_batch(scale_b, scale_b.dtype)
    sc, ec = _extract_sig_exp_batch(c, c.dtype)

    prod_sig = sa * sb
    prod_exp = ea + eb + esa.unsqueeze(1) + esb.unsqueeze(1)
    all_sig = torch.cat([sc.unsqueeze(1), prod_sig], dim=1)
    all_exp = torch.cat([ec.unsqueeze(1), prod_exp], dim=1)

    sig_sums, max_exps = _fused_sum_batch(all_sig, all_exp, n_fractional_bits)

    # Merge NaN/inf rows: replace sig_sums with fp64_sum so that NaN/inf
    # propagates correctly through the normalization below.
    sig_sums = torch.where(nan_or_inf, fp64_sum, sig_sums)
    max_exps = torch.where(nan_or_inf, torch.zeros_like(max_exps), max_exps)

    nan_mask = torch.isnan(sig_sums)
    inf_mask = ~nan_mask & torch.isinf(sig_sums)

    if output_type == "f16":
        # Specific quiet-NaN bit pattern used by the scalar implementation
        nan_val = torch.tensor(0x7FFF, dtype=torch.int16).view(torch.float16)
        inf_out = sig_sums.to(torch.float16)

        val = sig_sums * torch.pow(2.0, max_exps.to(torch.float64))
        sig_v, exp_v = _extract_sig_exp_batch(val, torch.float16)
        sig_rne = torch.round(sig_v * (2.0 ** 10)) / (2.0 ** 10)
        normal_out = (sig_rne * torch.pow(2.0, exp_v.to(torch.float64))).to(torch.float16)

        out = torch.where(inf_mask, inf_out, normal_out)
        out = torch.where(nan_mask, nan_val.expand_as(out), out)
        return out
    else:  # "f32" or "f32_e8m13"
        n_frac_out = 13 if output_type == "f32_e8m13" else 23
        nan_val = torch.tensor(0x7FFF_FFFF, dtype=torch.int32).view(torch.float32)
        inf_out = sig_sums.to(torch.float32)

        val = sig_sums * torch.pow(2.0, max_exps.to(torch.float64))
        sig_v, exp_v = _extract_sig_exp_batch(val, torch.float32)
        sig_rz = torch.trunc(sig_v * (2.0 ** n_frac_out)) / (2.0 ** n_frac_out)
        normal_out = (sig_rz * torch.pow(2.0, exp_v.to(torch.float64))).to(torch.float32)

        out = torch.where(inf_mask, inf_out, normal_out)
        out = torch.where(nan_mask, nan_val.expand_as(out), out)
        return out


def batch_nv_fused_dot_add_with_block_scale(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    n_fractional_bits: int,
) -> torch.Tensor:
    """Batched NV fused dot-add with block scaling over n k-length vectors.

    Internally vectorized: reshapes a and b into (n, num_blocks, block_size) and
    reduces with sum(dim=2) to get per-block dot products for all rows at once,
    then applies a batched fused_sum and final RZ normalization.

    Args:
        a: Tensor of shape (n, k).
        b: Tensor of shape (n, k).
        c: Tensor of shape (n,) – one accumulator per row.
        scale_a: Tensor of shape (n, num_blocks) – block scales for a.
        scale_b: Tensor of shape (n, num_blocks) – block scales for b.
        n_fractional_bits: Number of fractional bits for the fused sum.

    Returns:
        Tensor of shape (n,) containing one output element per row.
    """
    assert a.dim() == 2 and b.dim() == 2
    assert a.shape == b.shape
    n, k = a.shape
    num_blocks = scale_a.shape[1]

    # Per-row NaN check (mirrors the scalar's any() check)
    nan_mask = (
        torch.isnan(scale_a).any(dim=1)
        | torch.isnan(scale_b).any(dim=1)
        | torch.isnan(c)
    )

    if scale_a.dtype == torch.float8_e4m3fn:  # ue4m3: use magnitude only
        scale_a = scale_a.abs()
        scale_b = scale_b.abs()

    # Block dot products: (n, num_blocks) – same float32 reduction as the scalar loop
    block_size = k // num_blocks
    a_blocks = a.view(n, num_blocks, block_size)   # (n, num_blocks, block_size)
    b_blocks = b.view(n, num_blocks, block_size)
    block_dots = (a_blocks * b_blocks).sum(dim=2)  # (n, num_blocks) float32

    # Extract sig/exp for block scales
    sa, ea = _extract_sig_exp_batch(scale_a, scale_a.dtype)   # (n, num_blocks)
    sb, eb = _extract_sig_exp_batch(scale_b, scale_b.dtype)   # (n, num_blocks)
    sc, ec = _extract_sig_exp_batch(c, c.dtype)               # (n,)

    # Block contribution sig = block_dot * scale_sig_a * scale_sig_b;
    # exp = scale_exp_a + scale_exp_b (block_dot is folded into the significand).
    block_sig = block_dots.double() * sa * sb   # (n, num_blocks) float64
    block_exp = ea + eb                         # (n, num_blocks) int32

    all_sig = torch.cat([sc.unsqueeze(1), block_sig], dim=1)   # (n, 1+num_blocks)
    all_exp = torch.cat([ec.unsqueeze(1), block_exp], dim=1)   # (n, 1+num_blocks)

    sig_sums, max_exps = _fused_sum_batch(all_sig, all_exp, n_fractional_bits)

    # Final RZ normalization to float32
    val = sig_sums * torch.pow(2.0, max_exps.to(torch.float64))
    sig_v, exp_v = _extract_sig_exp_batch(val, torch.float32)
    sig_rz = torch.trunc(sig_v * (2.0 ** 23)) / (2.0 ** 23)
    result = (sig_rz * torch.pow(2.0, exp_v.to(torch.float64))).to(torch.float32)

    nan_val = torch.tensor(0x7FFF_FFFF, dtype=torch.int32).view(torch.float32)
    return torch.where(nan_mask, nan_val.expand_as(result), result)


def batch_amd_fused_dot_rd_add(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    n_fractional_bits: int,
    is_fp8: bool = False,
) -> torch.Tensor:
    """Batched AMD fused dot-round-down-add over n k-length vectors.

    Internally vectorized: all per-product and per-block operations are performed
    over the full (n, k) tensors at once, including both the is_fp8 and normal paths.

    Args:
        a: Tensor of shape (n, k).
        b: Tensor of shape (n, k).
        c: Tensor of shape (n,) – one accumulator per row.
        n_fractional_bits: Number of fractional bits for the fused sum.
        is_fp8: Whether the inputs use fp8 two-stage accumulation.

    Returns:
        Tensor of shape (n,) (float64) containing one output element per row.
    """
    assert a.dim() == 2 and b.dim() == 2
    assert a.shape == b.shape

    products = a.double() * b.double()                                    # (n, k)
    fp64_sum = products.sum(dim=1) + c.double()                           # (n,)
    nan_or_inf = torch.isnan(fp64_sum) | torch.isinf(fp64_sum)           # (n,) bool

    # Overflow detection (product exceeds float64-representable range of the dtype)
    p_inf_mask = (products >= 2.0 ** 128).any(dim=1)                     # (n,) bool
    n_inf_mask = (products <= -(2.0 ** 128)).any(dim=1)                  # (n,) bool

    # Extract sig/exp for products and accumulator
    sa, ea = _extract_sig_exp_batch(a, a.dtype)   # (n, k)
    sb, eb = _extract_sig_exp_batch(b, b.dtype)   # (n, k)
    sc, ec = _extract_sig_exp_batch(c, c.dtype)   # (n,)

    prod_sig = sa * sb   # (n, k) float64
    prod_exp = ea + eb   # (n, k) int32

    if is_fp8:
        # Two-stage accumulation: even and odd indexed products summed separately
        sig0, e0 = _fused_sum_batch(prod_sig[:, 0::2], prod_exp[:, 0::2], n_fractional_bits)
        sig1, e1 = _fused_sum_batch(prod_sig[:, 1::2], prod_exp[:, 1::2], n_fractional_bits)

        max_e01 = torch.maximum(e0, e1)                                       # (n,) int32
        diff0 = (n_fractional_bits + e0 - max_e01).to(torch.float64)
        diff1 = (n_fractional_bits + e1 - max_e01).to(torch.float64)
        s0f = torch.floor(sig0 * torch.pow(2.0, diff0)) / (2.0 ** n_fractional_bits)
        s1f = torch.floor(sig1 * torch.pow(2.0, diff1)) / (2.0 ** n_fractional_bits)
        s_combined = s0f + s1f                                                # (n,) float64
        e_combined = max_e01                                                  # (n,) int32
    else:
        s_combined, e_combined = _fused_sum_batch(prod_sig, prod_exp, n_fractional_bits)

    # Combine product sum with accumulator c
    max_e_all = torch.maximum(e_combined, ec)                                 # (n,) int32

    diff_s = (31 + e_combined - max_e_all).to(torch.float64)
    s_final = torch.floor(s_combined * torch.pow(2.0, diff_s)) / (2.0 ** 31)

    diff_sc = (24 + ec - max_e_all).to(torch.float64)
    sc_adj = torch.floor(sc * torch.pow(2.0, diff_sc)) / (2.0 ** 24)

    if is_fp8:
        # c is only included when it is within 25 exponent steps of the product sum
        sc_adj = torch.where(ec >= max_e_all - 25, sc_adj, torch.zeros_like(sc_adj))

    result = (s_final + sc_adj) * torch.pow(2.0, max_e_all.to(torch.float64))

    # Apply per-row overflow results (p_inf / n_inf)
    has_overflow = p_inf_mask | n_inf_mask
    result = torch.where(
        has_overflow & p_inf_mask & n_inf_mask,
        torch.full_like(result, float("nan")), result,
    )
    result = torch.where(
        has_overflow & p_inf_mask & ~n_inf_mask,
        torch.full_like(result, float("inf")), result,
    )
    result = torch.where(
        has_overflow & ~p_inf_mask & n_inf_mask,
        torch.full_like(result, float("-inf")), result,
    )

    # NaN/inf in fp64_sum takes precedence (checked first in the scalar)
    return torch.where(nan_or_inf, fp64_sum, result)


# ---------------------------------------------------------------------------
# Non-batched wrappers – compatibility shims so that batch_arithmetic.py can
# be used as a drop-in for arithmetic.py.  Each wrapper calls the batched
# version with a single-row batch (n=1) and unwraps the result.
# ---------------------------------------------------------------------------

def pairwise_dot(
    a: torch.Tensor, b: torch.Tensor, flush_denormal: bool = False
) -> float:
    return batch_pairwise_dot(a.unsqueeze(0), b.unsqueeze(0), flush_denormal)[0].item()


def fused_dot_add(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    n_fractional_bits: int,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> tuple[float, int]:
    return batch_fused_dot_add(
        a.unsqueeze(0),
        b.unsqueeze(0),
        c.unsqueeze(0),
        n_fractional_bits,
        scale_a.unsqueeze(0),
        scale_b.unsqueeze(0),
    )[0]


def nv_fused_dot_add(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    n_fractional_bits: int,
    output_type: str,
    scale_a: torch.Tensor | None = None,
    scale_b: torch.Tensor | None = None,
) -> torch.Tensor:
    sa = scale_a.unsqueeze(0) if scale_a is not None else None
    sb = scale_b.unsqueeze(0) if scale_b is not None else None
    return batch_nv_fused_dot_add(
        a.unsqueeze(0), b.unsqueeze(0), c.unsqueeze(0), n_fractional_bits, output_type, sa, sb
    )[0]


def nv_fused_dot_add_with_block_scale(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    n_fractional_bits: int,
) -> torch.Tensor:
    return batch_nv_fused_dot_add_with_block_scale(
        a.unsqueeze(0),
        b.unsqueeze(0),
        c.unsqueeze(0),
        scale_a.unsqueeze(0),
        scale_b.unsqueeze(0),
        n_fractional_bits,
    )[0]


def amd_fused_dot_rd_add(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    n_fractional_bits: int,
    is_fp8: bool = False,
) -> float:
    return batch_amd_fused_dot_rd_add(
        a.unsqueeze(0), b.unsqueeze(0), c.unsqueeze(0), n_fractional_bits, is_fp8
    )[0].item()
