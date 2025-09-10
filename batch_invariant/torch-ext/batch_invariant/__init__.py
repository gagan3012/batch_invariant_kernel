import torch
from ._ops import ops


def matmul_persistent(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor = None
) -> torch.Tensor:
    """
    Persistent matrix multiplication with optional bias.

    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)
        bias: Optional bias tensor of shape (N,)

    Returns:
        Output tensor of shape (M, N)
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    assert bias is None or bias.dim() == 1, "Bias must be 1D"

    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    ops.matmul_persistent(a, b, c, bias)

    return c


def log_softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute log_softmax using custom CUDA kernel.

    Args:
        input: Input tensor
        dim: Dimension along which to compute log_softmax (only -1 supported)

    Returns:
        Tensor with log_softmax applied
    """
    if dim != -1 and dim != input.ndim - 1:
        raise ValueError(
            "This implementation only supports log_softmax along the last dimension"
        )

    output = torch.empty_like(input)
    ops.log_softmax(input, output)

    return output


def mean_dim(
    input: torch.Tensor, dim: int, keepdim: bool = False, dtype: torch.dtype = None
) -> torch.Tensor:
    """
    Compute mean along a single dimension.

    Args:
        input: Input tensor
        dim: Single dimension along which to compute mean
        keepdim: Whether to keep the reduced dimension
        dtype: Output dtype

    Returns:
        Tensor with mean values along specified dimension
    """
    assert input.is_cuda, "Input must be a CUDA tensor"
    assert -input.ndim <= dim < input.ndim, f"Invalid dimension {dim}"

    if dim < 0:
        dim = dim + input.ndim

    if dtype is None:
        if input.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            dtype = torch.float32
        else:
            dtype = input.dtype

    if input.dtype != dtype:
        input = input.to(dtype)

    shape = list(input.shape)

    if keepdim:
        output_shape = shape.copy()
        output_shape[dim] = 1
    else:
        output_shape = shape[:dim] + shape[dim + 1 :]

    output = torch.empty(output_shape, dtype=dtype, device=input.device)
    ops.mean_dim(input, output, dim)

    return output


# Batch invariant mode functionality (if you still want the mode switching)
def mm_batch_invariant(a, b):
    return matmul_persistent(a, b)


def addmm_batch_invariant(bias, a, b):
    return matmul_persistent(a, b, bias=bias)


def _log_softmax_batch_invariant(input, dim, _half_to_float):
    assert not _half_to_float, "not implemented"
    return log_softmax(input, dim=dim)


def mean_batch_invariant(input, dim, keepdim=False, dtype: torch.dtype = None):
    if len(dim) == 1:
        return mean_dim(input, dim[0], keepdim=keepdim, dtype=dtype)
    else:
        # Multi-dimensional mean fallback
        n_elems = 1
        for d in dim:
            n_elems *= input.shape[d]
        return torch.sum(input, dim=dim, keepdim=keepdim, dtype=torch.float32) / n_elems
