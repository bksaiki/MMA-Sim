import torch

from .arithmetic import (
    pairwise_dot,
    fused_dot_add,
    nv_fused_dot_add,
    nv_fused_dot_add_with_block_scale,
    amd_fused_dot_rd_add,
)


def batch_pairwise_dot(
    a: torch.Tensor, b: torch.Tensor, flush_denormal: bool = False
) -> torch.Tensor:
    """Batched pairwise dot product over n k-length vectors.

    Args:
        a: Tensor of shape (n, k), dtype float32.
        b: Tensor of shape (n, k), dtype float32.
        flush_denormal: Whether to flush denormal values to zero.

    Returns:
        Tensor of shape (n,) containing the dot product for each row pair.
    """
    assert a.dtype == b.dtype == torch.float32
    assert a.dim() == 2 and b.dim() == 2
    assert a.shape == b.shape
    n = a.shape[0]
    results = torch.zeros(n, dtype=torch.float32)
    for i in range(n):
        results[i] = pairwise_dot(a[i], b[i], flush_denormal)
    return results


def batch_fused_dot_add(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    n_fractional_bits: int,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> list[tuple[float, int]]:
    """Batched fused dot-add over n k-length vectors.

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
    results = []
    for i in range(n):
        results.append(
            fused_dot_add(a[i], b[i], c[i], n_fractional_bits, scale_a[i], scale_b[i])
        )
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
    out_dtype = torch.float16 if output_type == "f16" else torch.float32
    results = torch.zeros(n, dtype=out_dtype)
    for i in range(n):
        sa = scale_a[i] if scale_a is not None else None
        sb = scale_b[i] if scale_b is not None else None
        results[i] = nv_fused_dot_add(
            a[i], b[i], c[i], n_fractional_bits, output_type, sa, sb
        )
    return results


def batch_nv_fused_dot_add_with_block_scale(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    n_fractional_bits: int,
) -> torch.Tensor:
    """Batched NV fused dot-add with block scaling over n k-length vectors.

    Args:
        a: Tensor of shape (n, k).
        b: Tensor of shape (n, k).
        c: Tensor of shape (n,) – one accumulator per row.
        scale_a: Tensor of shape (n, num_blocks_a) – block scales for a.
        scale_b: Tensor of shape (n, num_blocks_b) – block scales for b.
        n_fractional_bits: Number of fractional bits for the fused sum.

    Returns:
        Tensor of shape (n,) containing one output element per row.
    """
    assert a.dim() == 2 and b.dim() == 2
    assert a.shape == b.shape
    n = a.shape[0]
    results = torch.zeros(n, dtype=torch.float32)
    for i in range(n):
        results[i] = nv_fused_dot_add_with_block_scale(
            a[i], b[i], c[i], scale_a[i], scale_b[i], n_fractional_bits
        )
    return results


def batch_amd_fused_dot_rd_add(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    n_fractional_bits: int,
    is_fp8: bool = False,
) -> torch.Tensor:
    """Batched AMD fused dot-round-down-add over n k-length vectors.

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
    n = a.shape[0]
    results = torch.zeros(n, dtype=torch.float64)
    for i in range(n):
        results[i] = amd_fused_dot_rd_add(a[i], b[i], c[i], n_fractional_bits, is_fp8)
    return results
